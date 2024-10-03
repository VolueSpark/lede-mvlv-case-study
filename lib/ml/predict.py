from lib.ml.ml import DataLoader, PARAMS, NETWORK, MODEL_CLASS
import joblib, yaml, os, torch
from typing import Any
import numpy as np
import polars as pl
import pandas as pd

PATH = os.path.dirname(__file__)

from lib.ml import Scaler
from lib.ml.plotting import plot_aggregate


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

        self.scaler = Scaler.load(pickle_path=os.path.join(self.path, 'artifacts/dataset_scaler.pkl'))
        self.data = pl.read_parquet(os.path.join(self.path, 'artifacts/dataset_test.parquet'))

        test = self.data.drop('t_timestamp').to_numpy()
        test_scaled = self.scaler.transform(test)

        abs(self.scaler.inverse_transform(self.scaler.transform(test))-test).max()

        self.test_loader = DataLoader(
            data=test_scaled,
            features=self.scaler.features,
            input_width=PARAMS['dataloader']['window']['input_width'],
            label_width=PARAMS['dataloader']['window']['label_width'],
            batch_size=1,
            shuffle=False,
            name='test_loader'
        )

        self.model = getattr(NETWORK, MODEL_CLASS)(
            work_dir=self.path,
            inputs_shape=self.test_loader.inputs_shape,
            inputs_exo_shape=self.test_loader.inputs_exo_shape,
            targets_shape = self.test_loader.targets_shape
        )

        self.model.load_state_dict(
            torch.load(os.path.join(os.path.join(os.path.join(self.path, 'artifacts'), f'{self.model.name.lower()}.pytorch')),
                       weights_only=True)
        )

    def predict(self, inputs: Any=None):
        if inputs is None:
            for i, data in enumerate(self.test_loader):

                (index, inputs, inputs_exo, targets) = data.values()

                with torch.no_grad():
                    outputs = self.model(inputs, inputs_exo)

                x = self.data.select(['t_timestamp']+list(self.test_loader.meta.target_features.keys()))[index.item():index.item()+self.test_loader.meta.input_shape[2]]
                y = self.data.select(['t_timestamp']+list(self.test_loader.meta.target_features.keys()))[index.item()+self.test_loader.meta.input_shape[2]:index.item()+self.test_loader.meta.input_shape[2]+self.test_loader.meta.target_shape[2]]
                y_hat = (
                    pl.from_pandas(
                        pd.DataFrame(
                            self.scaler.inverse_transform(data=outputs[0].numpy().transpose(),
                                                          features=self.test_loader.meta.target_features),
                            columns=self.test_loader.meta.target_features.keys()
                        )
                    ).with_columns(t_timestamp=pl.Series(y.select('t_timestamp')))
                ).select(['t_timestamp']+list(self.test_loader.meta.target_features.keys()))

                plot_aggregate(i=index.item(), x=x, y_hat=y_hat, y=y)

                #inputs_data = self.data.drop('t_timestamp')[index.item():index.item()+self.test_loader.inputs_shape[2]]
                #target_data = self.data.drop('t_timestamp')[index.item()+self.test_loader.inputs_shape[2]:index.item()+self.test_loader.inputs_shape[2]+self.test_loader.inputs_exo_shape[2]]

                #abs(self.scaler.transform(inputs_data.to_numpy(), self.test_loader.meta.input_features).transpose()-inputs[0].numpy()).max()
                #abs(self.scaler.transform(y.to_numpy(), self.test_loader.meta.target_features).transpose()-targets[0].numpy()).max()

                #target_dnn = torch.cat([targets, inputs_exo], dim=1)[0]

                #abs(self.scaler.transform(target_data.to_numpy()).transpose()-target_dnn.numpy()).max()





