from torch.profiler import profile, record_function, ProfilerActivity
import os, yaml, torch, re, json, socket, torcheval
from torch.utils.tensorboard import SummaryWriter
import polars.selectors as cs
from torcheval import metrics
from datetime import datetime
from typing import Tuple
from torch import nn
import polars as pl
import numpy as np

from lib.ml import Split, Scaler, decorate_train, decorator_epoch, device

from lib.ml.dataloader import DataLoader
from lib.price.insight import fetch_hist_spot
from lib.price.valuta import fetch_hist_valuta
from lib.weather.weather import fetch_hist_weather

TIME_FORMAT = '%Y-%m-%dT%H:%M:%S'
PATH = os.path.dirname(__file__)

with open(os.path.join(PATH,'config.yaml')) as fp:
    CONFIG = yaml.safe_load(fp)
    MODEL_CLASS = CONFIG['ml']['variant']
    LIBRARY_PATH = f"lib.ml.model.{MODEL_CLASS.lower()}"
    NETWORK = __import__(LIBRARY_PATH, fromlist=[MODEL_CLASS])
    PARAMS = CONFIG['params']

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Optimizer:
    def __new__(cls, *args, **kwargs):
        if 'params' not in kwargs:
            raise ValueError('No model params specified')
        optimizer_class = PARAMS['optimizer']['algorithm']
        params = PARAMS['optimizer'][optimizer_class]['params']
        return getattr(torch.optim, optimizer_class)(kwargs['params'], **params)


class Loss:
    def __init__(self, *args, **kwargs):
        loss_class = PARAMS['loss']['function']
        params = PARAMS['loss'][loss_class]['params']
        self.obj = getattr(nn, loss_class)(**params)

    def eval(self, x: torch.Tensor, y: torch.Tensor):
        return self.obj(x, y)


class Metric:
    def __init__(self, *args, **kwargs):
        metric_class = PARAMS['metric']['function']
        params = PARAMS['metric'][metric_class]['params']
        self.obj =  getattr(torcheval.metrics, metric_class)(**params).to(device)

    def eval(self, x: torch.Tensor, y: torch.Tensor):
        return self.obj.update(x.view(x.shape[1],x.shape[0]*x.shape[2]),
                           y.view(y.shape[1],y.shape[0]*y.shape[2])).compute()


class Ml:

    def __init__(
            self,
            root: str,
            uuid: str
    ):
        self.name = uuid
        self.path = os.path.join(root, uuid)

        self.logs = os.path.join(self.path, 'logs')
        self.tensorboard_path = os.path.join(self.logs, f'tensorboard/{datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}_{socket.gethostname()}')

        self.artifacts_path = os.path.join(self.path, 'artifacts')
        self.bronze_data_path = os.path.join(self.path, 'data/bronze')
        self.silver_data_path = os.path.join(self.path, 'data/silver')
        self.gold_data_path = os.path.join(self.path, 'data/gold')

        os.makedirs(self.tensorboard_path, exist_ok=True)
        os.makedirs(self.artifacts_path, exist_ok=True)
        os.makedirs(self.bronze_data_path, exist_ok=True)
        os.makedirs(self.silver_data_path, exist_ok=True)
        os.makedirs(self.gold_data_path, exist_ok=True)

        self.prepare_bronze_data()
        self.prepare_silver_data()

        self.writer = SummaryWriter(log_dir=self.tensorboard_path )

        # load the data
        self.train_loader, self.val_loader = self.load_data(data_path=os.path.join(self.silver_data_path, 'data.parquet'))

        # create the dnn
        self.model = getattr(NETWORK, MODEL_CLASS)(
            input_shape=self.train_loader.input_shape,
            target_shape=self.train_loader.target_shape
        ).to(device)


        # tensor graph visualization
        (_, input, target) = next(iter(self.val_loader)).values()
        self.writer.add_graph(model=self.model, input_to_model=(input))

        self.epoch_cnt = PARAMS['num_epochs']
        self.optimizer = Optimizer(params=self.model.parameters())
        self.loss = Loss()
        self.metric = Metric()

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("model_inference"):
                self.model(input)

        profile_analysis = prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=100)
        with open(os.path.join(self.artifacts_path, 'profiling.txt'), 'w') as fp:
            fp.write(profile_analysis)


    def prepare_bronze_data(self):

        if not os.path.exists(os.path.join(self.bronze_data_path, 'meter.parquet')):
            with open( os.path.join( CONFIG['data']['topology'], f'{self.name}'), 'r') as fp:
                data = json.load(fp)
            df = pl.DataFrame()
            for meter in data['load']:
                df = df.vstack(pl.read_parquet(os.path.join(CONFIG['data']['meter'], meter['meter_id'])))
            df.rename({'datetime':'timestamp'}).write_parquet(os.path.join(self.bronze_data_path, 'meter.parquet'))

        df = pl.read_parquet(os.path.join(self.bronze_data_path, 'meter.parquet'))

        from_time =df['timestamp'].min()
        to_time =df['timestamp'].max()

        with open( os.path.join( CONFIG['data']['geojson'], 'lede.geojson'), 'r') as fp:
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
                .sort(by='timestamp', descending=False)
            ).upsample(time_column="timestamp", every="1h").fill_null(strategy="forward").fill_null(strategy="backward")

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

            def rbf_transform(value: int, period: int):
                return np.exp(-(value/period) ** 2)


            data = (
                data
                .with_columns(
                    pl.col('timestamp').map_elements(lambda x: rbf_transform(
                        value=x.hour + 1,
                        period=24), return_dtype=pl.Float64).alias('rbf_hour'),
                    pl.col('timestamp').map_elements(lambda x: rbf_transform(
                        value=x.weekday() + 1,
                        period=7), return_dtype=pl.Float64).alias('rbf_weekday')
                )
            )

            #data = data.select([col for col in data.columns if not bool(re.match(r'^.*(\d{18})$', col)) ] + ['p_kwh_707057500041377889', 'q_kvarh_707057500041377889']) TODO remove

            data = (data
             .rename({feature: (f't_{feature}' if feature == 'timestamp' else "X_{0}{1}_y".format(feature[0].upper(), re.match(r'^.*(\d{18})$', feature).group(1))
            if bool(re.match(r'^.*(\d{18})$', feature)) else f'X_{feature}') for i, feature in enumerate(data.columns)}))

            data = data.drop(['X_euro_mwh',
                              'X_nok_euro',
                              'X_cloudcovermean',
                              'X_cloudcover',
                              'X_dewmean',
                              'X_feelslikemean',
                              'X_feelslikemin',
                              'X_precipprobmean',
                              'X_precipprob',
                              'X_pressuremean',
                              'X_solarenergymean',
                              'X_solarradiationmean',
                              'X_solarradiation',
                              'X_solarradiationmean',
                              'X_sunrise',
                              'X_sunset',
                              'X_tempmax',
                              'X_tempmean',
                              'X_tempmin',
                              'X_windspeedmax',
                              'X_windspeedmin'])

            data.write_parquet(os.path.join(self.silver_data_path, 'data.parquet'))
            data.drop(cs.starts_with('X_P')).write_parquet(os.path.join(self.silver_data_path, 'reactive.parquet'))
            data.drop(cs.starts_with('X_Q')).write_parquet(os.path.join(self.silver_data_path, 'active.parquet'))

    def load_data(self, data_path):
        data = pl.read_parquet(data_path)

        # split data. features excluding time frame, test untouched
        features, train, val, test = Split(data=data, split=PARAMS['dataloader']['split'])

        # scale train and validation data
        scaler = Scaler()
        scaler.fit(train, features)

        train_scaled = scaler.transform(train)
        val_scaled = scaler.transform(val)

        train_loader = DataLoader(
            data=train_scaled,
            features=features,
            input_width=PARAMS['dataloader']['window']['input_width'],
            label_width=PARAMS['dataloader']['window']['label_width'],
            batch_size=PARAMS['dataloader']['batch_size'],
            shuffle=PARAMS['dataloader']['shuffle'],
            name='train_loader'
        )

        val_loader = DataLoader(
            data=val_scaled,
            features=features,
            input_width=PARAMS['dataloader']['window']['input_width'],
            label_width=PARAMS['dataloader']['window']['label_width'],
            batch_size=PARAMS['dataloader']['batch_size'],
            shuffle=PARAMS['dataloader']['shuffle'],
            name='val_loader'
        )

        # save artifacts
        scaler.save(pickle_path=self.artifacts_path)
        test.write_parquet(os.path.join(self.gold_data_path, 'test.parquet'))

        return train_loader, val_loader

    @decorator_epoch
    def epoch_training(
            self,
            epoch_i: int
    ) -> Tuple[str, int, float, float]:
        self.model.train()
        for i, data in enumerate(self.train_loader):
            (_, input, target) = data.values()

            self.optimizer.zero_grad()

            output = self.model.forward( input=input )

            loss = self.loss.eval(output, target )
            acc = self.metric.eval(output, target )

            loss.backward()

            self.optimizer.step()

            #print(f'epoch_training: index={epoch_i*self.train_loader.meta.loader_depth + i}, loss={loss.item()}, acc={acc.item()}')

            yield (
                epoch_i*self.train_loader.meta.loader_depth + i,
                loss,
                acc
            )

    @decorator_epoch
    def epoch_validation(
            self,
            epoch_i: int
    ) -> Tuple[str, int, float, float]:
        self.model.eval()

        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                (_, input, target) = data.values()

                output = self.model(input)

                loss = self.loss.eval(output, target)
                acc = self.metric.eval(output, target)

                yield (
                    epoch_i*self.val_loader.meta.loader_depth + i,
                    loss,
                    acc
                )

    @decorate_train
    def train(self):

        for epoch_i in range(self.epoch_cnt):

            (ave_loss, ave_acc) = self.epoch_training(epoch_i=epoch_i)

            (ave_vloss, ave_vacc) = self.epoch_validation(epoch_i=epoch_i)

            yield (epoch_i, ave_loss, ave_acc, ave_vloss, ave_vacc)











