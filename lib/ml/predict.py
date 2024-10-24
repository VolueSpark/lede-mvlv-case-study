
from sklearn.metrics import root_mean_squared_error
from lib.ml.ml import DataLoader
from sklearn.preprocessing import MinMaxScaler
import os, torch, json, importlib

from lib.ml.analysis import decorator_confidence

import polars as pl
import pandas as pd

import re

PATH = os.path.dirname(__file__)

from lib import decorator_timer, logger
from lib.ml import Scaler, device

import numpy as np
import scipy.stats

def annotate_metrics(ax, metrics: dict):
    ax.annotate(
        '\n'.join([f'{key}={value}' for key, value in metrics.items()]),
        xy=(0, 1), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize='medium', verticalalignment='top', fontfamily='serif',
        bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0)
    )


def confidence_interval(data:pl.DataFrame, confidence: float) ->pl.DataFrame:
        ci_analysis = []
        def ci(data:pl.Series, confidence):
            a = 1.0 * np.array(data)
            n = len(a)
            m, se = np.mean(a, axis=0), scipy.stats.sem(a)
            h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)

            return m, m-h, m+h

        prediction_range = range(data.filter(pl.col('type')=='target')['offset'].min(), data.filter(pl.col('type')=='target')['offset'].max()+1)
        for power_type in ['P', 'Q']:
            wildcard = f'^X_{power_type}.*$'
            scaler = MinMaxScaler()
            scaler.fit(data.filter(pl.col('type')=='target').select(wildcard))
            scale_coef = sum([ max(val, abs(scaler.data_min_[i])) for i, val in enumerate(scaler.data_max_)])
            for k in prediction_range:
                pred_k = data.filter((pl.col('offset') == k) & (pl.col('type') == 'forecast')).select(wildcard)
                real_k = data.filter((pl.col('offset') == k) & (pl.col('type') == 'target')).select(wildcard)

                # absolute sum % error on maximum sum target value
                error_k = abs((pred_k - real_k).to_numpy()).sum(axis=1)/scale_coef*100 # absolute error [power unit/meter]
                mean_k, ci_lb_k, ci_ub_k = ci(data=error_k, confidence=confidence)
                ci_analysis.append(
                    {
                        'type': power_type,
                        'k': k - prediction_range[0],
                        'mean': mean_k,
                        'ci_lb': ci_lb_k,
                        'ci_ub': ci_ub_k}
                )
        return pl.from_dicts(ci_analysis)



class Predict:
    def __init__(
            self,
            root: str,
            uuid: str
    ):
        self.name = uuid
        self.path = os.path.join(root, uuid)

        if not os.path.exists(os.path.join(self.path, 'artifacts')):
            raise Exception(f'Need to first train {uuid} prior to predictions')

        with open(os.path.join(self.path, 'artifacts', 'meta.json')) as fp:
            self.meta = json.load(fp)

        self.scaler = Scaler().load(pickle_path=os.path.join(self.path, 'artifacts'))
        self.test = pl.read_parquet(os.path.join(self.path, 'data/gold/test.parquet'))

        self.test_loader = DataLoader(
            data=self.scaler.transform(self.test.drop('t_timestamp')),
            features=self.scaler.features,
            input_width=self.meta['input_width'],
            label_width=self.meta['label_width'],
            batch_size=1,
            shuffle=False,
            name='test_loader'
        )

        module = importlib.import_module(self.meta['library'])
        self.model = getattr(module, self.meta['class'])(
            input_shape=self.meta['shape']['input'],
            target_shape=self.meta['shape']['target']
        )
        self.model.load_state_dict(torch.load(os.path.join(os.path.join(os.path.join(self.path, 'artifacts'), f'state_dict.pth')), weights_only=True))
        self.model.to(device)

    def parse( self, index: int, output: np.ndarray) -> pl.DataFrame:
        timestamp = self.test.select('t_timestamp')[int(index)].item()
        columns = ['base', 'offset', 'type']+list(self.meta['features']['target'].keys())

        x = self.test.select(list(self.meta['features']['target'].keys()))[index:index+self.meta['input_width']].with_columns(
            type=pl.lit('intput'),
            base=pl.lit(timestamp)).with_row_index(name='offset')

        y = self.test.select(list(self.meta['features']['target'].keys()))[index+self.meta['input_width']:index+self.meta['input_width']+self.meta['label_width']].with_columns(
            type=pl.lit('target'),
            base=pl.lit(timestamp)).with_row_index(offset=x['offset'].max()+1,name='offset')

        y_hat = (
            pl.from_pandas(
                pd.DataFrame(
                    self.scaler.inverse_transform(data=output.transpose(),
                                                  features=self.meta['features']['target']),
                    columns=self.meta['features']['target'].keys()
                )
            ).with_columns(
                type=pl.lit('forecast'),
                base=pl.Series(y.select('base'))).with_row_index(offset=x['offset'].max()+1, name='offset')
        )

        return x.select(columns).vstack(y.select(columns)).vstack(y_hat.select(columns))

    #@decorator_confidence

    def predict(self):

        data =pl.DataFrame()

        logger.info(f"Run predict on test dataset {os.path.join(self.path, 'data/gold/test.parquet')}")
        with torch.no_grad():
            for data_i in self.test_loader:
                (index, input, _) = data_i.values()

                output = self.model(input)

                data.vstack(
                    self.parse(
                        index=index.cpu().numpy()[0],
                        output=output.cpu().numpy()[0]
                    ), in_place=True
                )
        data.write_parquet(os.path.join(self.path, 'data/gold/data.parquet'))
        logger.info(f"Prediction on test dataset saved at {os.path.join(self.path, 'data/gold/data.parquet')}")

        logger.info(f"Compile predict metadata on test dataset {os.path.join(self.path, 'data/gold/test.parquet')}")
        pred_range = range(data.filter(pl.col('type')=='target')['offset'].min(), data.filter(pl.col('type')=='target')['offset'].max()+1)
        metadata = []

        forecast_data = data.filter(pl.col('type')=='forecast')
        target_data = data.filter(pl.col('type')=='target')

        for power_type in ['P', 'Q']:
            wildcard = f'^X_{power_type}.*$'
            for meter in data.select(wildcard).columns:

                y_max = round(target_data.select(meter).max().item(),2)
                y_min = round(target_data.select(meter).min().item(),2)
                y_range = round(y_max-y_min,2)

                meta = {
                    'id':re.search(f'(?<=X_{power_type})\\d+(?=_y)', meter).group(),
                    'type': power_type,
                    'max': y_max,
                    'min': y_min,
                    'range': y_range
                }
                if meta['id'] == '707057500041377841':
                    print('')
                for k in pred_range:
                    pred_k = forecast_data.filter(pl.col('offset') == k).select(meter).to_numpy()
                    real_k = target_data.filter(pl.col('offset') == k).select(meter).to_numpy()


                    rmse_k = root_mean_squared_error(pred_k, real_k)/y_range*100

                    meta |= {
                        f'rmse_{k-min(pred_range)+1}': round(rmse_k,2) # absolute error % over max real value over entire time horizon for prediction index k,
                    }
                metadata.append(meta)
        pl.from_dicts(metadata).write_parquet(os.path.join(self.path, 'data/gold/metadata.parquet'))
        logger.info(f"Prediction metadata on test dataset saved at {os.path.join(self.path, 'data/gold/metadata.parquet')}")






