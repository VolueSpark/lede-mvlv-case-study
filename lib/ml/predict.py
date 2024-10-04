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

        self.scaler = Scaler().load(pickle_path=os.path.join(self.path, 'artifacts'))
        with open(os.path.join(self.path, 'artifacts', 'meta.json')) as fp:
            self.meta = json.load(fp)

        module = importlib.import_module(self.meta['library'])
        self.model = getattr(module, self.meta['class'])(
            inputs_shape=self.meta['shape']['input'],
            inputs_exo_shape=self.meta['shape']['exo'],
            targets_shape=self.meta['shape']['target']
        )
        self.model.load_state_dict(torch.load(os.path.join(os.path.join(os.path.join(self.path, 'artifacts'), f'state_dict.pth')),weights_only=True))

    def parse(self, i: int, data: torch.Tensor) -> Tuple[pl.DataFrame,...]:
        x = self.data.select(['t_timestamp']+list(self.meta['features']['target'].keys()))[i:i+self.meta['shape']['input'][2]]
        y = self.data.select(['t_timestamp']+list(self.meta['features']['target'].keys()))[i+self.meta['shape']['input'][2]:i+self.meta['shape']['input'][2]+self.meta['shape']['target'][2]]
        y_hat = (
            pl.from_pandas(
                pd.DataFrame(
                    self.scaler.inverse_transform(data=data[0].numpy().transpose(),
                                                  features=self.meta['features']['target']),
                    columns=self.meta['features']['target'].keys()
                )
            ).with_columns(t_timestamp=pl.Series(y.select('t_timestamp')))
        ).select(['t_timestamp']+list(self.meta['features']['target'].keys()))
        return (x,y,y_hat)

    def load_data(self,data: pl.DataFrame) -> DataLoader:
        window_length = self.meta['shape']['input'][2]+self.meta['shape']['target'][2]
        assert data.shape[0] >= window_length, f"Data input dimension should satisfy ({window_length},{len(self.scaler.features)+1}) inclusive of feature t_timestamp"
        assert abs((self.scaler.inverse_transform(self.scaler.transform(data[0:window_length].drop('t_timestamp').to_numpy())) - data[0:window_length].drop('t_timestamp').to_numpy())).max() < 1e-7, f'Scaler coefficients are corrupted'

        self.data = data

        return DataLoader(
            data=self.scaler.transform(data.drop('t_timestamp').to_numpy()),
            features=self.scaler.features,
            input_width=self.meta['shape']['input'][2],
            label_width=self.meta['shape']['target'][2],
            batch_size=1,
            shuffle=False,
            name='data_loader'
        )


    def predict(self, data: pl.DataFrame) -> Tuple[pl.DataFrame, ...]:

        data_loader = self.load_data(data)

        for i, data_i in enumerate(data_loader):

            (index, inputs, inputs_exo, targets) = data_i.values()

            with torch.no_grad():
                outputs = self.model(inputs, inputs_exo)
                yield self.parse(i=i, data=outputs)



                #plot_aggregate(i=index.item(), x=x, y_hat=y_hat, y=y)

                #inputs_data = self.data.drop('t_timestamp')[index.item():index.item()+self.test_loader.inputs_shape[2]]
                #target_data = self.data.drop('t_timestamp')[index.item()+self.test_loader.inputs_shape[2]:index.item()+self.test_loader.inputs_shape[2]+self.test_loader.inputs_exo_shape[2]]

                #abs(self.scaler.transform(inputs_data.to_numpy(), self.test_loader.meta.input_features).transpose()-inputs[0].numpy()).max()
                #abs(self.scaler.transform(y.to_numpy(), self.test_loader.meta.target_features).transpose()-targets[0].numpy()).max()

                #target_dnn = torch.cat([targets, inputs_exo], dim=1)[0]

                #abs(self.scaler.transform(target_data.to_numpy()).transpose()-target_dnn.numpy()).max()





