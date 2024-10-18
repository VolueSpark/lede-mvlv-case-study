import matplotlib.pyplot as plt
import numpy as np
import random

from lib.ml.ml import DataLoader
import os, torch, json, importlib

from typing import Tuple
import polars as pl
import pandas as pd

PATH = os.path.dirname(__file__)

from lib import decorator_timer, logger
from lib.ml import Scaler, device



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
        self.model.load_state_dict(torch.load(os.path.join(os.path.join(os.path.join(self.path, 'artifacts'), f'state_dict.pth')),weights_only=True))
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

    @decorator_timer
    def predict(self):

        predictions =pl.DataFrame()
        logger.info(f"Run predict on test dataset {os.path.join(self.path, 'data/gold/test.parquet')}")
        with torch.no_grad():
            for data_i in self.test_loader:
                (index, input, _) = data_i.values()

                output = self.model(input)

                predictions.vstack(
                    self.parse(
                        index=index.cpu().numpy()[0],
                        output=output.cpu().numpy()[0]
                    ), in_place=True
                )
        predictions.write_parquet(os.path.join(self.path, 'data/gold/predictions.parquet'))
        logger.info(f"Prediction on test dataset saved at {os.path.join(self.path, 'data/gold/test.parquet')}")

    def inspect(self, prediction_index: int=0, show_pct:int =0.1):



        predictions = pl.read_parquet(os.path.join(self.path, 'data/gold/predictions.parquet'))
        prediction_range = (predictions.filter(pl.col('type')=='target')['offset'].min(), predictions.filter(pl.col('type')=='target')['offset'].max())
        offset =prediction_index+min(prediction_range)

        if len(predictions.select(pl.col(r'^X_P.*$')).columns):
            p = predictions.select(['base', 'offset', 'type']+predictions.select(r'^X_P.*$').columns).sort(by='base', descending=False)
            p = p.with_columns(p_sum=p.select(r'^X_P.*$').sum_horizontal())
            y_p = p.filter(pl.col('offset')==offset).filter( (pl.col('type')=='forecast') | (pl.col('type')=='target') )

        if len(predictions.select(pl.col(r'^X_Q.*$')).columns):
            q = predictions.select(['base', 'offset', 'type']+predictions.select(r'^X_Q.*$').columns).sort(by='base', descending=False)
            q = q.with_columns(q_sum=q.select(r'^X_Q.*$').sum_horizontal())
            y_q = q.filter(pl.col('offset')==offset).filter( (pl.col('type')=='forecast') | (pl.col('type')=='target') )

        fig, axs = plt.subplots(2,1, sharex=True, figsize=(15,10))

        plot_range = 168
        n = y_p['base'].unique().shape[0]
        plot_index = random.randint(0, n-plot_range)
        from_index = plot_index
        to_index = plot_index+plot_range

        t = y_p.filter(pl.col('type')=='target')['base'][from_index:to_index]
        y_p_real = y_p.filter(pl.col('type')=='target')['p_sum'][from_index:to_index]
        y_p_pred = y_p.filter(pl.col('type')=='forecast')['p_sum'][from_index:to_index]

        axs[0].plot(t, y_p_real, label='$P^{real}_{kWh}$', color='#088F8F', linewidth=1, marker='o', linestyle='dashed')
        axs[0].plot(t, y_p_pred, label='$P^{pred}_{kWh}$', color='#0000FF', linewidth=1, marker='x')
        axs[0].set_title(f"Predicted aggregated active power")
        axs[0].set_ylabel('$P^{sum}_{kWh}$')
        axs[0].legend(loc='lower left')

        t = y_q.filter(pl.col('type')=='target')['base'][from_index:to_index]
        y_q_real = y_q.filter(pl.col('type')=='target')['q_sum'][from_index:to_index]
        y_q_pred = y_q.filter(pl.col('type')=='forecast')['q_sum'][from_index:to_index]

        axs[1].plot(t, y_q_real, label='$Q^{real}_{kVArh}$', color='#088F8F', linewidth=1, marker='o', linestyle='dashed')
        axs[1].plot(t, y_q_pred, label='$Q^{pred}_{kVArh}$', color='#0000FF', linewidth=1, marker='x')
        axs[1].set_title(f"Predicted aggregated reactive power]")
        axs[1].set_ylabel('$Q^{sum}_{kWh}$')
        axs[1].legend(loc='lower left')

        fig.text(0.5, 0.04, 'time', ha='center')
        plt.suptitle(f"Aggregated load profile at T=t+{prediction_index} (t={t[0].strftime('%Y-%m-%d %H')}):\nNeighborhood {self.name}")

        plt.xticks(rotation=25)
        plt.tight_layout()
        plt.show()