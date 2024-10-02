from lib.ml.ml import DataLoader, PARAMS, NETWORK, MODEL_CLASS
import matplotlib.pyplot as plt
import joblib, yaml, os, torch
from typing import Any
import numpy as np
import polars as pl
import pandas as pd

PATH = os.path.dirname(__file__)


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

        self.scaler = joblib.load(os.path.join(self.path, 'artifacts/minmaxScaler.pkl'))

        self.data = pl.read_parquet(os.path.join(self.path, 'artifacts/test_dataset.parquet')).drop('t_timestamp')
        self.data_scaled = pd.DataFrame(data=self.scaler.transform(self.data), columns=self.data.columns)

        abs(self.scaler.inverse_transform(self.scaler.transform(self.data)) - self.data.to_numpy()).max()

        self.test_loader = DataLoader(data=self.data_scaled, params=PARAMS['dataloader'], name='test')

        self.model = getattr(NETWORK, MODEL_CLASS)(
            work_dir=self.path,
            inputs_shape=self.test_loader.inputs_shape,
            inputs_exo_shape=self.test_loader.inputs_exo_shape,
            targets_shape = self.test_loader.targets_shape
        )

        self.model.load_state_dict(torch.load(os.path.join(os.path.join(os.path.join(self.path, 'artifacts'), f'__{self.model.name.lower()}__.pytorch')), weights_only=True))

    def predict(self, inputs: Any=None):
        if inputs is None:
            for i, data in enumerate(self.test_loader):

                (index, inputs, inputs_exo, targets) = data.values()

                with torch.no_grad():
                    outputs = self.model(inputs, inputs_exo)

                self.plot(index, inputs, inputs_exo, outputs, targets)

    def plot(self,index: Any, inputs: Any, inputs_exo: Any, outputs: Any, targets: Any):
        schema={feature:pl.Float64 for feature in self.test_loader.dataset.inputs}

        x_ = self.data[0:48]
        y_ = self.data[48:48+24]

        x = pd.DataFrame(data=self.scaler.inverse_transform(inputs[0].numpy().transpose()), columns=self.test_loader.dataset.inputs.keys())
        y = pl.DataFrame(self.scaler.inverse_transform(torch.cat([targets, inputs_exo], dim=1)[0].numpy().transpose()), schema=schema)
        y_hat = pl.DataFrame(self.scaler.inverse_transform(torch.cat([outputs, inputs_exo], dim=1)[0].numpy().transpose()), schema=schema)

'''
class Predict:
    def __init__(self, uuid):
        self.uuid = uuid


    def predict(self):

        model_class = config['ml']['variant']
        library_path = f"lib.ml.model.{model_class.lower()}"
        Network = __import__(library_path, fromlist=[model_class])

        params={
            'window':
                {
                    'input_width': 48,
                    'label_width': 24,
                },
            'batch_size': 1,
            'shuffle': False
        }

        scaler = joblib.load(os.path.join(self.path, 'minmax_scaler.pkl'))
        test_loader = DataLoader(
            pl.read_parquet(os.path.join(self.path, 'test_scaled.parquet')),
            params=params,
            name='test'
        )

        model = getattr(Network, model_class)(
            work_dir=self.path,
            inputs_shape=test_loader.inputs_shape,
            inputs_exo_shape=test_loader.inputs_exo_shape,
            targets_shape = test_loader.targets_shape)
        model.load_state_dict(torch.load(os.path.join(self.path, 'model.pytorch'), weights_only=True))

        model.eval()

        with torch.no_grad():
            for i, data in enumerate((pbar := tqdm(test_loader, position=0, leave=True))):
                (_, inputs, inputs_exo, targets) = data.values()

                outputs = model(inputs, inputs_exo)
'''