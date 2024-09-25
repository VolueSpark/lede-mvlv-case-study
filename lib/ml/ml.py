from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import MinMaxScaler
import os, yaml, torch, re, json, socket
from datetime import datetime
from typing import Tuple
from tqdm import tqdm
from torch import nn
import polars as pl
import numpy as np

from lib import logger
from lib.ml.dataloader import DataLoader
from lib.price.insight import fetch_hist_spot
from lib.price.valuta import fetch_hist_valuta
from lib.weather.weather import fetch_hist_weather

PATH = os.path.dirname(__file__)

with open(os.path.join(PATH,'config.yaml')) as fp:
    config = yaml.safe_load(fp)

class Ml:

    def __init__(
            self,
            uuid: str,
            work_dir: str
    ):
        self.name = uuid
        self.path = os.path.join(work_dir, uuid)

        self.log_dir = os.path.join(work_dir, f'runs/{uuid}/{datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}_{socket.gethostname()}')
        self.bronze_data_path = os.path.join(self.path, 'data/bronze')
        self.silver_data_path = os.path.join(self.path, 'data/silver')

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.bronze_data_path, exist_ok=True)
        os.makedirs(self.silver_data_path, exist_ok=True)

        self.prepare_bronze_data()
        self.prepare_silver_data()

        self.writer = SummaryWriter(log_dir=self.log_dir)


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

        train_loader = DataLoader( train_scaled, params=params['dataloader'], name='train')
        test_loader = DataLoader( test_scaled , params=params['dataloader'], name='test')
        val_loader = DataLoader(val_scaled, params=params['dataloader'], name='val')

        return train_loader, test_loader, val_loader

    def create(self):
        model_class = config['ml']['variant']
        library_path = f"lib.ml.model.{model_class.lower()}"
        Network = __import__(library_path, fromlist=[model_class])

        logger.info(f"Loading {model_class} model at working direction {self.path}")

        self.params = config['params'][model_class.lower()]

        self.train_loader, self.test_loader, self.val_loader = self.load_data(
            data_path=os.path.join(self.silver_data_path, 'data.parquet'),
            params=self.params
        )

        self.model = getattr(Network, model_class)(
            work_dir=self.path,
            inputs_shape=self.train_loader.inputs_shape,
            inputs_exo_shape=self.train_loader.inputs_exo_shape,
            targets_shape = self.train_loader.targets_shape
        )

        self.loss_fn = nn.HuberLoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['learning_rate'])

    def train_one_epoch(self, epoch_index):
        running_loss = 0.
        last_loss = 0.

        for batch_i, data in enumerate(self.train_loader):

            (inputs, inputs_exo, targets) = data.values()

            self.optimizer.zero_grad()

            outputs = self.model.forward(
                inputs=inputs,
                inputs_exo=inputs_exo
            )

            loss = self.loss_fn(outputs, targets)
            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()
            if batch_i % 10 == 9:
                last_loss = running_loss / 10 # loss per batch
                tb_x = epoch_index * len(self.train_loader) + batch_i + 1
                self.writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.
                logger.info(f"[{batch_i + 1}] LOSS: Train {last_loss:.4f}")

        return last_loss
            

    def train(self):

        best_vloss = 1_000_000.

        for epoch_i in range(self.params['num_epochs']):

            self.model.train()
            avg_loss = self.train_one_epoch(epoch_i)

            running_vloss = 0.0
            self.model.eval()

            with torch.no_grad():
                for i, vdata in enumerate(self.val_loader):
                    (vinputs, vinputs_exo, vlabels) = vdata.values()
                    voutputs = self.model(vinputs, vinputs_exo)
                    vloss = self.loss_fn(voutputs, vlabels)
                    running_vloss += vloss
                    logger.info(f"[{i}] LOSS: Train {avg_loss:.4f}; Validation {vloss:.4f}")

            avg_vloss = running_vloss / (i + 1)

            self.writer.add_scalars('Training vs. Validation Loss',
                               { 'Training' : avg_loss, 'Validation' : avg_vloss },
                                    epoch_i + 1)
            self.writer.flush()

            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                #model_path = 'model_{}_{}'.format(timestamp, epoch_i)
                #torch.save(self.model.state_dict(), model_path)
















