from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from torch import nn
import polars as pl
import numpy as np
import os, torch, json

from lib import logger


class TimeseriesDataset(Dataset):
    def __init__(self, data: pl.DataFrame, params: dict):

        self.inputs = {feature:i for i, feature in enumerate(data.columns) if feature[0:2]=='X_'}
        self.exo_inputs = {feature:i for i, feature in enumerate(data.columns) if feature[0:2]=='X_' and feature[-2:]!='_y'}
        self.targets = {feature:i for i, feature in enumerate(data.columns) if feature[-2:]=='_y'}

        self.features = dict(sorted((self.inputs | self.exo_inputs | self.targets).items(), key=lambda item: item[1], reverse=False))

        self.data = data.select(self.features.keys()).to_numpy()
        self.params = params

        self.seq_len = params['dataloader']['sequence_length']
        self.pred_hor = params['dataloader']['prediction_horizon']
        self.win_len = params['dataloader']['sequence_length'] + params['dataloader']['prediction_horizon']

    def __len__(self):
        return len(self.data) - self.win_len + 1

    def __getitem__(self, idx):
        window = self.data[idx:idx + self.win_len]

        x = torch.cat( [torch.tensor(window[0:self.seq_len, col_idx], dtype=torch.float32).unsqueeze(-1) for feature, col_idx in self.inputs.items()], dim=-1)
        x_exo = torch.cat( [torch.tensor(window[self.seq_len:self.win_len, col_idx], dtype=torch.float32).unsqueeze(-1) for feature, col_idx in self.exo_inputs.items()], dim=-1)
        y = torch.cat( [torch.tensor(window[self.seq_len:self.win_len, col_idx], dtype=torch.float32).unsqueeze(-1) for feature, col_idx in self.targets.items()], dim=-1)

        return {'x':x, 'x_exo':x_exo, 'y':y}

class SparkNet(nn.Module):
    def __init__( self, work_dir: str, params: dict):
        super().__init__()
        self.work_dir = work_dir
        self.params = params

        self.load_data()
        self.create_model()

    def load_data(self):

        # purge non-features from dataframe
        self.data = pl.read_parquet(os.path.join(self.work_dir, 'data/silver/data.parquet'))
        self.date = self.data.select(pl.col(r'^t_.*$'))
        self.data = self.data.drop(self.date.columns)

        n = self.data.shape[0]
        self.train = self.data[:int(n * self.params['split'][0])]
        self.test = self.data[int(n * self.params['split'][1]):]
        self.val = self.data[int(n * self.params['split'][0]):int(n * self.params['split'][1])]

        logger.info(f"Split data on a training:validation:test ratio of "
                    f"{int(n * self.params['split'][0])}:"
                    f"{int(n * (self.params['split'][1] - self.params['split'][0]))}:"
                    f"{int(n * (1 - self.params['split'][1]))}")

        self.scaler = MinMaxScaler()
        self.scaler.fit(self.train)

        self.train_scl = pl.DataFrame(self.scaler.transform(self.train), self.data.schema)
        self.test_scl = pl.DataFrame(self.scaler.transform(self.test), self.data.schema)
        self.val_scl = pl.DataFrame(self.scaler.transform(self.val), self.data.schema)

        self.train_dataset = TimeseriesDataset(self.train_scl, self.params)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.params['dataloader']['batch_size'], shuffle=False)

        for i_batch, sample_batched in enumerate(self.train_dataloader):
            print(i_batch, sample_batched['x'].size(),sample_batched['x_exo'].size(), sample_batched['y'].size())

            assert abs(self.train_scl[i_batch*self.params['dataloader']['batch_size']:i_batch*self.params['dataloader']['batch_size']+self.params['dataloader']['sequence_length']].to_numpy() - sample_batched['x'][0].numpy()).max()<1e-7, f'Batch {i_batch} does not match with data source'


    def crate_model(self):
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    '''
    def train_model(self):

        if self.train_loader is None:
            ValueError("Dataset not defined!")

        self.train()
        for i, (x, exog, y) in enumerate(tqdm(self.train_loader, leave=False, desc="Training")):
            pass
    '''

