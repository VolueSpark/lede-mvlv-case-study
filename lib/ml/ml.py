from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import MinMaxScaler
from torcheval.metrics import MeanSquaredError
import os, yaml, torch, re, json, socket
from datetime import datetime
from typing import Tuple
from torch import nn
import pandas as pd
import polars as pl
import numpy as np
import joblib
import time

from lib import logger
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
    PARAMS = CONFIG['params'][MODEL_CLASS.lower()]

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

        self.prepare_bronze_data()
        self.prepare_silver_data()

        self.writer = SummaryWriter(log_dir=self.tensorboard_path )

        os.makedirs(self.tensorboard_path, exist_ok=True)
        os.makedirs(self.artifacts_path, exist_ok=True)
        os.makedirs(self.bronze_data_path, exist_ok=True)
        os.makedirs(self.silver_data_path, exist_ok=True)

        self.create()

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

    def load_data(self, data_path):
        # purge non-features from dataframe
        data = pl.read_parquet(data_path)

        n = data.shape[0]
        split_n = [ int(n*split_i)for split_i in PARAMS['split']]

        train = data[:split_n[0]].drop('t_timestamp')
        val = data[split_n[0]:split_n[1]].drop('t_timestamp')

        data[split_n[1]:].write_parquet(os.path.join(self.artifacts_path, 'test_dataset.parquet'))

        logger.info(f"Split data on a training:validation ratio of {split_n[0]}:{split_n[1]-split_n[0]}:{n-split_n[1]}")

        scaler = MinMaxScaler()
        scaler.fit(train)
        joblib.dump(scaler, os.path.join(self.artifacts_path, 'minmaxScaler.pkl'))

        train_scaled = pd.DataFrame(data=scaler.transform(train), columns=train.columns)
        val_scaled = pd.DataFrame(data=scaler.transform(val), columns=val.columns)

        train_loader = DataLoader(train_scaled, params=PARAMS['dataloader'], name='train')
        val_loader = DataLoader(val_scaled, params=PARAMS['dataloader'], name='val')

        return train_loader, val_loader

    def create(self):

        logger.info(f"Loading {MODEL_CLASS} model at working direction {self.path}")

        # load the data
        self.train_loader, self.val_loader = self.load_data(data_path=os.path.join(self.silver_data_path, 'data.parquet'))

        # create the dnn
        self.model = getattr(NETWORK, MODEL_CLASS)(
            work_dir=self.path,
            inputs_shape=self.train_loader.inputs_shape,
            inputs_exo_shape=self.train_loader.inputs_exo_shape,
            targets_shape = self.train_loader.targets_shape
        )

        # tensor graph visualization
        (_, inputs, inputs_exo, targets) = next(iter(self.val_loader)).values()
        self.writer.add_graph(model=self.model, input_to_model=(inputs, inputs_exo))

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=PARAMS['learning_rate'])
        self.loss_fn = nn.HuberLoss(reduction='mean')
        self.acc_fn = MeanSquaredError()

    def train(self):

        train_idx = 0
        val_idx = 0

        epoch_ave_vloss = np.nan
        epoch_ave_vacc = np.nan

        for epoch_i in range(PARAMS['num_epochs']):
            t0_epoch = time.time()
            self.model.train()

            epoch_ave_loss = 0
            epoch_ave_acc = 0

            t0 = time.time()

            for train_i, train_data in enumerate(self.train_loader):

                (_, inputs, inputs_exo, targets) = train_data.values()

                self.optimizer.zero_grad()

                outputs = self.model.forward(
                    inputs=inputs,
                    inputs_exo=inputs_exo
                )

                loss = self.loss_fn(outputs, targets)
                acc = self.acc_fn.update(outputs.view(outputs.shape[1],outputs.shape[0]*outputs.shape[2]),
                                         targets.view(targets.shape[1],targets.shape[0]*targets.shape[2])).compute()

                loss.backward()

                self.optimizer.step()

                epoch_ave_loss = (loss.item() + train_i*epoch_ave_loss)/(train_i+1)
                epoch_ave_acc = (acc.item() + train_i*epoch_ave_acc)/(train_i+1)

                # TODO some verbose / logging to be improved
                self.writer.add_scalar('Loss/train', loss.item(), train_idx)
                self.writer.add_scalar('Acc/train', acc.item(), train_idx)
                train_idx +=1

                train_it_per_sec = train_i/(time.time()-t0)
                logger.info(f"\033[F\rEPOCH {(epoch_i+1)/PARAMS['num_epochs']*100:.0f}% ({epoch_i+1}/{PARAMS['num_epochs']}) [{time.time() - t0_epoch:.3f} sec]|TRAIN {(train_i+1)/self.train_loader.iter_cnt*100:.0f}% ({train_i+1}/{self.train_loader.iter_cnt}): loss={epoch_ave_loss:.4f} - acc={epoch_ave_acc:.4f} [{train_it_per_sec:.4f} it/sec]|VAL {0}%: val_loss={epoch_ave_vloss:.4f} - val_acc={epoch_ave_vacc:.4f} [- it/sec]",end='', color=logger.BLUE)
                # TODO end

            self.model.eval()

            epoch_ave_vloss = 0
            epoch_ave_vacc = 0

            with torch.no_grad():
                t0 = time.time()
                for val_i, val_data in enumerate(self.val_loader):
                    (_, inputs, inputs_exo, targets) = val_data.values()

                    outputs = self.model(inputs, inputs_exo)

                    loss = self.loss_fn(outputs, targets)
                    acc = self.acc_fn.update(outputs.view(outputs.shape[1],outputs.shape[0]*outputs.shape[2]),
                                             targets.view(targets.shape[1],targets.shape[0]*targets.shape[2])).compute()

                    epoch_ave_vloss = (loss.item() + val_i*epoch_ave_vloss)/(val_i+1)
                    epoch_ave_vacc = (acc.item() + val_i*epoch_ave_vacc)/(val_i+1)

                    # TODO some verbose / logging to be improved
                    self.writer.add_scalar('Loss/val', loss.item(), val_idx)
                    self.writer.add_scalar('Acc/val', acc.item(), val_idx)
                    val_idx +=1

                    val_it_per_sec = val_i/(time.time()-t0)
                    logger.info(f"\033[F\rEPOCH {(epoch_i+1)/PARAMS['num_epochs']*100:.0f}% ({epoch_i+1}/{PARAMS['num_epochs']}) [{time.time() - t0_epoch:.3f} sec]|TRAIN {(train_i+1)/self.train_loader.iter_cnt*100:.0f}% ({train_i+1}/{self.train_loader.iter_cnt}): loss={epoch_ave_loss:.4f} - acc={epoch_ave_acc:.4f} [{train_it_per_sec:.4f} it/sec]|VAL {(val_i+1)/self.val_loader.iter_cnt*100:.0f}% ({val_i+1}/{self.val_loader.iter_cnt}): val_loss={epoch_ave_vloss:.4f} - val_acc={epoch_ave_vacc:.4f} [{val_it_per_sec:.4f} it/sec]",end='', color=logger.BLUE)

                logger.info('')

            self.writer.add_scalars('Loss/epoch', {'train': epoch_ave_loss, 'val': epoch_ave_vloss}, epoch_i)
            self.writer.add_scalars('Acc/epoch', {'train': epoch_ave_acc, 'val': epoch_ave_vacc}, epoch_i)
            # TODO end

        model_path = os.path.join(self.artifacts_path, f'__{self.model.name.lower()}__.pytorch')
        torch.save(self.model.state_dict(), model_path)

        logger.info(f'Pytorch model saved to: {model_path}')




















