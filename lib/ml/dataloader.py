from sklearn.preprocessing import MinMaxScaler
from typing import List
import polars as pl
import torch, os, json
import numpy as np

from lib import logger

class WindowGenerator:
    def __init__(
            self,
            prediction_horizon: int,
            historical_horizon: int
    ):
        pass

    def split_window(self, window: torch.Tensor):
        # Inputs prior to input_width
        inputs_prior = window[:, :self.input_width, :]

        # Inputs posterior and selecting relevant targets
        inputs_posterior = torch.stack([window[:, self.input_width:,
                                        index if feature_name in self.targets.keys()
                                        else index - self.label_width]
                                        for index, feature_name in enumerate(self.features)], dim=-1)

        # Concatenating inputs
        inputs = torch.cat([inputs_prior, inputs_posterior], dim=1)

        # Selecting targets
        targets = torch.stack([window[:, self.input_width:, index]
                               for index, feature_name in enumerate(self.features)
                               if feature_name in self.targets.keys()], dim=-1)

        return inputs, targets

    def make_dataset(self, data: pl.DataFrame, shuffle=True):
        data = data.to_numpy()

        # Generating windowed dataset
        dataset = TimeseriesDataset(data, self.total_window_size)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

        return loader


class TimeseriesDataset(torch.utils.data.Dataset):
    def __init__(self, data: np.ndarray, window_size: int):
        self.data = data
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size + 1

    def __getitem__(self, idx):
        window = self.data[idx:idx + self.window_size]
        return torch.tensor(window, dtype=torch.float32)


class DataLoader(WindowGenerator):
    def __init__( self, data_path: str, params: dict):
        self.data = pl.read_parquet(os.path.join(data_path, 'data.parquet'))

        n = self.data.shape[0]
        self.train = self.data[:int(n * params['split'][0])]
        self.test = self.data[int(n * params['split'][1]):]
        self.val = self.data[int(n * params['split'][0]):int(n * params['split'][1])]


        logger.info(f"Split data on a training:validation:test ratio of "
              f"{int(n * params['split'][0])}:"
              f"{int(n * (params['split'][1] - params['split'][0]))}:"
              f"{int(n * (1 - params['split'][1]))}")

        self.scaler = MinMaxScaler()
        self.scaler.fit(self.train)

        self.train_scl = pl.DataFrame(self.scaler.transform(self.train), self.data.schema)
        self.test_scl = pl.DataFrame(self.scaler.transform(self.test), self.data.schema)
        self.val_scl = pl.DataFrame(self.scaler.transform(self.val), self.data.schema)


        with open(os.path.join(data_path, 'data/gold/inputs.json'), 'rb') as fp:
            self.inputs = json.load(fp)
        with open(os.path.join(data_path, 'data/gold/labels.json'), 'rb') as fp:
            self.labels = json.load(fp)

    def make_train_dataset(self):

        return inputs, exogenous, targets