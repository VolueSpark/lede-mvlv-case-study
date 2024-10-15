import numpy as np

from lib.ml.ml import DataLoader
import os, torch, json, importlib

from typing import Tuple
import polars as pl
import pandas as pd

PATH = os.path.dirname(__file__)

from lib.ml import Scaler



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

    def parse( self, index: int, output: np.ndarray) -> Tuple[pl.DataFrame,...]:

        x = self.test.select(['t_timestamp']+list(self.meta['features']['target'].keys()))[index:index+self.meta['input_width']]
        y = self.test.select(['t_timestamp']+list(self.meta['features']['target'].keys()))[index+self.meta['input_width']:index+self.meta['input_width']+self.meta['label_width']]

        y_hat = (
            pl.from_pandas(
                pd.DataFrame(
                    self.scaler.inverse_transform(data=output.transpose(),
                                                  features=self.meta['features']['target']),
                    columns=self.meta['features']['target'].keys()
                )
            ).with_columns(t_timestamp=pl.Series(y.select('t_timestamp')))
        ).select(['t_timestamp']+list(self.meta['features']['target'].keys()))

        return (x,y,y_hat)

    def predict(self) -> Tuple[pl.DataFrame, ...]:

        with torch.no_grad():
            for data_i in self.test_loader:
                (index, input, _) = data_i.values()

                output = self.model(input)

                yield self.parse(
                    index=index.numpy()[0],
                    output=output.numpy()[0]
                )
