from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import List, Tuple
import torch

from lib import logger
from lib.ml import device



class DataloaderMetaData(BaseModel):
    name: str
    sample_cnt: int
    loader_depth: int
    input_width: int
    label_width: int
    input_shape: List[int]
    target_shape: List[int]
    input_features: dict
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
        self.label_width = label_width
        self.total_window_length = input_width + label_width

    def __len__(self):
        return len(self.data) - self.total_window_length + 1

    def __getitem__(self, idx):
        window = self.data[idx:idx + self.total_window_length]

        '''
        input = torch.hstack(
            (
                torch.cat(
                    [torch.tensor(window[0:self.input_width, col_idx], dtype=torch.float32).unsqueeze(-1)
                     for feature, col_idx in self.inputs.items()],
                    dim=-1
                ).transpose(0, 1)
                , torch.vstack(
                    (
                        torch.cat(
                            [torch.tensor(window[self.input_width - self.label_width:self.total_window_length - self.label_width, col_idx], dtype=torch.float32).unsqueeze(-1)
                             for feature, col_idx in self.targets.items()],
                            dim=-1
                        ).transpose(0, 1),
                        torch.cat(
                            [torch.tensor(window[self.input_width:self.total_window_length, col_idx], dtype=torch.float32).unsqueeze(-1)
                             for feature, col_idx in self.inputs_exo.items()],
                            dim=-1
                        ).transpose(0, 1)
                    )
                )
            )
        )
        '''
        input = torch.vstack(
            (
                torch.cat(
                    [torch.tensor(window[0:self.input_width, col_idx], dtype=torch.float32).unsqueeze(-1)
                     for feature, col_idx in self.targets.items()],
                    dim=-1
                ).transpose(0, 1),
                torch.cat(
                    [torch.tensor(window[self.total_window_length-self.input_width:self.total_window_length, col_idx], dtype=torch.float32).unsqueeze(-1)
                     for feature, col_idx in self.inputs_exo.items()],
                    dim=-1
                ).transpose(0, 1)
            )
        )

        target = torch.cat(
            [torch.tensor(window[self.input_width:self.total_window_length, col_idx], dtype=torch.float32).unsqueeze(-1)
             for feature, col_idx in self.targets.items()],
            dim=-1
        ).transpose(0, 1)

        return {
            'index': idx,               # data shuffle index
            'input': input.to(device),         # features inputs based on historical data
            'target': target.to(device)        # target labels
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

        # sample batch data at index for inspection
        window_length = input_width + label_width

        sample_batched = next(iter(self))
        idx = sample_batched['index'][0].numpy().flat[0]

        input = np.vstack(
            (
                np.vstack([data[idx:idx + input_width, col_idx] for feature, col_idx in dataset.targets.items()]),
                np.vstack([data[idx + window_length - input_width:idx + window_length, col_idx] for feature, col_idx in dataset.inputs_exo.items()])
            )
        )

        '''
        input = np.hstack(
            (
                np.vstack([data[idx:idx + input_width, col_idx] for feature, col_idx in dataset.inputs.items()]),
                np.vstack(
                    (
                        np.vstack([data[idx + input_width - label_width:idx + input_width, col_idx] for feature, col_idx in dataset.targets.items()]),
                        np.vstack([data[idx + input_width:idx + input_width + label_width, col_idx] for feature, col_idx in dataset.inputs_exo.items()])
                    )
                )
            )
        )
        '''

        error = abs(input - sample_batched['input'][0].cpu().numpy()).max()
        assert error<1e-7, f'Validation failed for input data'

        target = np.vstack([data[idx+input_width:idx+window_length, col_idx] for feature, col_idx in dataset.targets.items()])

        error = abs(target - sample_batched['target'][0].cpu().numpy()).max()
        assert error<1e-7, f'Validation failed for target data'

        self.input_shape = sample_batched['input'].size()
        self.target_shape = sample_batched['target'].size()

        self.meta = DataloaderMetaData(
            name=name,
            sample_cnt=data.shape[0],
            loader_depth=self.dataset.__len__()//(batch_size + 0 if drop_last else 1),
            input_width=input_width,
            label_width=label_width,
            input_shape=list(sample_batched['input'].size()),
            target_shape=list(sample_batched['target'].size()),
            input_features=dataset.inputs,
            target_features=dataset.targets,
        )

        logger.info(f"{self.meta.name.upper():} ({self.meta.sample_cnt} samples, {self.meta.loader_depth} loader depth): "
                    f"group:[#batch, #channels, #sequence] -> (input:[{self.meta.input_shape}], target:[{self.meta.target_shape}])")

