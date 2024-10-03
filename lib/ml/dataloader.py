from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import List, Tuple
import torch

from lib import logger


class DataloaderMetaData(BaseModel):
    name: str
    sample_cnt: int
    loader_depth: int
    input_shape: List[int]
    input_exo_shape: List[int]
    target_shape: List[int]
    input_features: dict
    input_exo_features: dict
    target_features: dict


class WidowGenerator(torch.utils.data.Dataset):
    def __init__(
            self, data: np.ndarray,
            features: List[str],
            input_width: int,
            label_width: int
    ):
        super(WidowGenerator, self).__init__()
        self.data = data

        self.inputs = {feature:features[feature] for feature in features if feature[0:2] == 'X_'}
        self.inputs_exo = {feature:features[feature] for feature in features if feature[0:2] == 'X_' and feature[-2:]!='_y'}
        self.targets = {feature:features[feature] for feature in features if feature[-2:] == '_y'}

        self.input_width = input_width
        self.total_window_length = input_width + label_width

    def __len__(self):
        return len(self.data) - self.total_window_length +1

    def __getitem__(self, idx):
        window = self.data[idx:idx + self.total_window_length]

        inputs = torch.cat(
            [torch.tensor(window[0:self.input_width, col_idx], dtype=torch.float32).unsqueeze(-1)
             for feature, col_idx in self.inputs.items()],
            dim=-1
        ).transpose(0, 1)
        inputs_exo = torch.cat(
            [torch.tensor(window[self.input_width:self.total_window_length, col_idx], dtype=torch.float32).unsqueeze(-1)
             for feature, col_idx in self.inputs_exo.items()],
            dim=-1
        ).transpose(0, 1)
        targets = torch.cat(
            [torch.tensor(window[self.input_width:self.total_window_length, col_idx], dtype=torch.float32).unsqueeze(-1)
             for feature, col_idx in self.targets.items()],
            dim=-1
        ).transpose(0, 1)
        return {
            'index': idx,               # data shuffle index
            'inputs': inputs,         # features inputs based on historical data
            'inputs_exo': inputs_exo, # feature inputs based on exogenous forecast data
            'targets': targets        # target labels
        }


class DataLoader(torch.utils.data.DataLoader):

    def __init__(
            self,
            data: np.ndarray,
            features: List[str],
            input_width: int,
            label_width: int,
            batch_size: int = 32,
            shuffle=True,
            name: str = 'Loader',
            drop_last: bool = True
    ):
        dataset = WidowGenerator(
            data=data,
            features=features,
            input_width=input_width,
            label_width=label_width
        )
        super(DataLoader, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last
        )

        sample_batched = next(iter(self))
        idx = sample_batched['index'][0].numpy().flat[0]

        inputs_start = idx
        inputs_end = inputs_start + input_width
        inputs_data = np.column_stack([data[inputs_start:inputs_end,col_idx] for feature, col_idx in dataset.inputs.items()]).transpose()

        inputs_exo_start = idx + input_width
        inputs_exo_end = inputs_exo_start+label_width
        inputs_exo_data = np.column_stack([data[inputs_exo_start:inputs_exo_end,col_idx] for feature, col_idx in dataset.inputs_exo.items()]).transpose()

        targets_start = idx + input_width
        targets_end = targets_start+label_width
        targets_data = np.column_stack([data[targets_start:targets_end,col_idx] for feature, col_idx in dataset.targets.items()]).transpose()

        assert abs(inputs_data - sample_batched['inputs'][0].numpy()).max()<1e-7, f'Validation valued for inputs data'
        assert abs(inputs_exo_data - sample_batched['inputs_exo'][0].numpy()).max()<1e-7, f'Validation valued for exo inputs data'
        assert abs(targets_data - sample_batched['targets'][0].numpy()).max()<1e-7, f'Validation valued for targets data'

        self.inputs_shape = sample_batched['inputs'].size()
        self.inputs_exo_shape = sample_batched['inputs_exo'].size()
        self.targets_shape = sample_batched['targets'].size()

        self.meta = DataloaderMetaData(
            name=name,
            sample_cnt=data.shape[0],
            loader_depth=self.dataset.__len__()//(batch_size + 0 if drop_last else 1),
            input_shape=list(sample_batched['inputs'].size()),
            input_exo_shape=list(sample_batched['inputs_exo'].size()),
            target_shape=list(sample_batched['targets'].size()),
            input_features=dataset.inputs,
            input_exo_features=dataset.inputs_exo,
            target_features=dataset.targets,
        )

        logger.info(f"{self.meta.name.upper():} ({self.meta.sample_cnt} samples, {self.meta.loader_depth} loader depth): "
                    f"group:[#batch, #channels, #sequence] -> (inputs:[{self.meta.input_shape}], inputs_exo[{self.meta.input_exo_shape}], targets:[{self.meta.target_shape}])")

