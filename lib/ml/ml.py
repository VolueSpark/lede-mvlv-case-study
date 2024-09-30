from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import MinMaxScaler
import os, yaml, torch, re, json, socket
from torcheval.metrics import R2Score, MeanSquaredError
from scipy.stats import linregress
from datetime import datetime
from typing import Tuple
from torch import nn
import polars as pl
import numpy as np

from lib import logger
from lib.ml.dataloader import DataLoader
from lib.price.insight import fetch_hist_spot
from lib.price.valuta import fetch_hist_valuta
from lib.weather.weather import fetch_hist_weather

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

PATH = os.path.dirname(__file__)

with open(os.path.join(PATH,'config.yaml')) as fp:
    config = yaml.safe_load(fp)

class Log:
    def __init__(self, log_dir: str, window=100):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

        self.window = window
        self.train_loss = []
        self.val_loss = []
        self.t = -1

    def graph(self, model, inputs: Tuple[torch.Tensor, ...]):
        self.writer.add_graph(model, inputs)
        self.writer.flush()

    def output(self, epoch: int, batch:int,  train_loss: float, val_loss:  float, train_acc:  float, val_acc:  float)->bool:
        self.t += 1
        self.writer.add_scalars(
            'Loss',
            {
                'Train': train_loss,
                'Val': val_loss
            },
            self.t
        )
        self.writer.add_scalars(
            'Accuracy',
            {
                'Train': train_acc,
                'Val': val_acc
            },
            self.t
        )

        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)

        logger.info(f'[Epoch: {epoch}| Batch: {batch}] Loss: Train: {train_loss}; Val: {val_loss}')

        if len(self.train_loss) > self.window:
            self.train_loss.pop(0)
            self.val_loss.pop(0)

            train_slope = linregress(np.arange(1, self.window+1), self.train_loss).slope
            val_slope = linregress(np.arange(1, self.window+1), self.val_loss).slope

            self.writer.add_scalars(
                'Loss slope',
                {
                    'train': train_slope,
                    'val': val_slope
                },
                self.t
            )
            # terminate when validation slope becomes positive, ie., overfitting
            return bool(val_slope)
        return False


class Ml:

    def __init__(
            self,
            uuid: str,
            work_dir: str
    ):
        self.name = uuid
        self.path = os.path.join(work_dir, uuid)

        self.bronze_data_path = os.path.join(self.path, 'data/bronze')
        self.silver_data_path = os.path.join(self.path, 'data/silver')

        os.makedirs(self.bronze_data_path, exist_ok=True)
        os.makedirs(self.silver_data_path, exist_ok=True)

        self.prepare_bronze_data()
        self.prepare_silver_data()

        self.log = Log(log_dir=os.path.join(work_dir, f'runs/{uuid}/{datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}_{socket.gethostname()}'))


    def prepare_bronze_data(self):

        if not os.path.exists(os.path.join(self.bronze_data_path, 'meter.parquet')):
            with open( os.path.join( config['data']['topology'], f'{self.name}'), 'r') as fp:
                data = json.load(fp)
            df = pl.DataFrame()
            for meter in data['load']:
                df = df.vstack(pl.read_parquet(os.path.join(config['data']['meter'], meter['meter_id'])))
            df.rename({'datetime':'timestamp'}).write_parquet(os.path.join(self.bronze_data_path, 'meter.parquet'))

        df = pl.read_parquet(os.path.join(self.bronze_data_path, 'meter.parquet'))

        from_time =df['timestamp'].min()
        to_time =df['timestamp'].max()

        with open( os.path.join( config['data']['geojson'], 'lede.geojson'), 'r') as fp:
            data = json.load(fp)
            for feature in data['features']:
                if feature['properties']['objecttype'] == 'ConformLoad':
                    longitude, latitude = feature['geometry']['coordinates']
                    break

        if not os.path.exists(os.path.join(self.bronze_data_path, 'weather.parquet')):
            (fetch_hist_weather(
                latitude=latitude,
                longitude=longitude,
                date_from=from_time,
                date_to=to_time)
             .write_parquet(os.path.join(self.bronze_data_path, 'weather.parquet')))

        if not os.path.exists(os.path.join(self.bronze_data_path, 'price.parquet')):
            spot = fetch_hist_spot(
                from_time=from_time,
                to_time=to_time,
                latitude=latitude,
                longitude=longitude
            )
            valuta = fetch_hist_valuta(
                from_time=from_time,
                to_time=to_time
            )
            (spot
             .join(valuta, on='timestamp', validate='1:1')
             .with_columns((pl.col('euro_mwh')/1000.0*pl.col('nok_euro'))
                           .alias(f'nok_kwh'))
             .write_parquet(os.path.join(self.bronze_data_path, 'price.parquet')))

    def prepare_silver_data(self):
        if not os.path.exists(os.path.join(self.silver_data_path, 'data.parquet')):
            meter = (
                pl.read_parquet(os.path.join(self.bronze_data_path, 'meter.parquet'))
                .with_columns(
                    p_kwh=(pl.col('p_kwh_out') - pl.col('p_kwh_in')),
                    q_kvarh=(pl.col('q_kvarh_out') - pl.col('q_kvarh_in')),

                )
                .drop('p_kwh_in', 'p_kwh_out', 'q_kvarh_in', 'q_kvarh_out')
                .pivot(
                    columns='meter_id',
                    index='timestamp',
                    values=['p_kwh', 'q_kvarh']
                )
                .fill_nan(None)
                .sort(by='timestamp', descending=False)
            ).upsample(time_column="timestamp", every="1h").fill_null(strategy="forward")

            price = pl.read_parquet(os.path.join(self.bronze_data_path, 'price.parquet'))
            weather = pl.read_parquet(os.path.join(self.bronze_data_path, 'weather.parquet'))

            data = (
                meter
                .join(
                    price
                    .join(
                        weather,
                        on='timestamp', validate='1:1'),
                    on='timestamp', validate='1:1')
            )

            def rbf_transform(x: pl.Series, period: int, input_range: Tuple[int, int]) -> pl.Series:
                x_normalized = (x - input_range[0]) / (input_range[1] - input_range[0]) * period
                return np.exp(-0.5 * ((x_normalized - period / 2) / 1.0) ** 2)

            data = (data
                    .with_columns(pl.col('timestamp').map_batches(lambda x: rbf_transform(x=x.dt.hour(), period=24, input_range=(0,23))).alias('rbf_hour'),
                                  pl.col('timestamp').map_batches(lambda x: rbf_transform(x=x.dt.weekday(), period=7, input_range=(0,6))).alias('rbf_weekday') ))

            (data
             .rename({feature: (f't_{feature}' if feature == 'timestamp' else "X_{0}{1}_y".format(feature[0].upper(), re.match(r'^.*(\d{18})$', feature).group(1))
            if bool(re.match(r'^.*(\d{18})$', feature)) else f'X_{feature}') for i, feature in enumerate(data.columns)})
             .write_parquet(os.path.join(self.silver_data_path, 'data.parquet')))

    def load_data(self, data_path: str, params: dict):
        # purge non-features from dataframe
        data = pl.read_parquet(data_path).drop('t_timestamp')

        n = data.shape[0]
        train = data[:int(n * params['split'][0])]
        test = data[int(n * params['split'][1]):]
        val = data[int(n * params['split'][0]):int(n * params['split'][1])]

        logger.info(f"Split data on a training:validation:test ratio of "
                    f"{int(n * params['split'][0])}:"
                    f"{int(n * (params['split'][1] - params['split'][0]))}:"
                    f"{int(n * (1 - params['split'][1]))}")

        scaler = MinMaxScaler()
        scaler.fit(train)

        train_scaled = pl.DataFrame(scaler.transform(train), data.schema)
        test_scaled = pl.DataFrame(scaler.transform(test), data.schema)
        val_scaled = pl.DataFrame(scaler.transform(val), data.schema)

        train_loader = DataLoader(train_scaled, params=params['dataloader'], name='train')
        test_loader = DataLoader(test_scaled , params=params['dataloader'], name='test')
        val_loader = DataLoader(val_scaled, params=params['dataloader'], name='val')

        return train_loader, test_loader, val_loader

    def create(self):
        model_class = config['ml']['variant']
        library_path = f"lib.ml.model.{model_class.lower()}"
        Network = __import__(library_path, fromlist=[model_class])

        logger.info(f"Loading {model_class} model at working direction {self.path}")

        self.params = config['params'][model_class.lower()]

        # load the data
        self.train_loader, self.test_loader, self.val_loader = self.load_data(
            data_path=os.path.join(self.silver_data_path, 'data.parquet'),
            params=self.params
        )

        # create the dnn
        self.model = getattr(Network, model_class)(
            work_dir=self.path,
            inputs_shape=self.train_loader.inputs_shape,
            inputs_exo_shape=self.train_loader.inputs_exo_shape,
            targets_shape = self.train_loader.targets_shape
        )

        # tensor graph visualization
        (_, inputs, inputs_exo, targets) = next(iter(self.val_loader)).values()
        self.log.graph(model=self.model, inputs=(inputs, inputs_exo))

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['learning_rate'])
        self.loss_fn = nn.HuberLoss(reduction='mean')
        self.acc_fn = R2Score()

    def train(self):

        for epoch_i in range(self.params['num_epochs']):
            for batch_i, data_i in enumerate(self.train_loader):

                # training
                self.model.train()
                (_, inputs, inputs_exo, targets) = data_i.values()

                self.optimizer.zero_grad()

                outputs = self.model.forward(
                    inputs=inputs,
                    inputs_exo=inputs_exo
                )

                train_loss = self.loss_fn(outputs, targets)
                train_acc = self.acc_fn.update(outputs.view(outputs.shape[1],outputs.shape[0]*outputs.shape[2]),
                                         targets.view(targets.shape[1],targets.shape[0]*targets.shape[2])).compute()

                train_loss.backward()

                self.optimizer.step()

                # validate on one sample
                self.model.eval()

                with torch.no_grad():
                    (_, inputs, inputs_exo, targets) = next(iter(self.val_loader)).values()

                    outputs = self.model(inputs, inputs_exo)

                    val_loss = self.loss_fn(outputs, targets)
                    val_acc = self.acc_fn.update(outputs.view(outputs.shape[1],outputs.shape[0]*outputs.shape[2]),
                                             targets.view(targets.shape[1],targets.shape[0]*targets.shape[2])).compute()

                if self.log.output(epoch=epoch_i, batch=batch_i, train_loss=train_loss.item(), val_loss=val_loss.item(), train_acc=train_acc.item(), val_acc=val_acc.item()):
                    continue;

















