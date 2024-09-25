import polars as pl
import torch

from lib import logger

class WidowGenerator(torch.utils.data.Dataset):
    def __init__(self, data: pl.DataFrame, params: dict):
        super(WidowGenerator, self).__init__()

        self.inputs = {feature:i for i, feature in enumerate(data.columns) if feature[0:2]=='X_'}
        self.inputs_exo = {feature:i for i, feature in enumerate(data.columns) if feature[0:2]=='X_' and feature[-2:]!='_y'}
        self.targets = {feature:i for i, feature in enumerate(data.columns) if feature[-2:]=='_y'}

        self.feature = dict(sorted((self.inputs | self.inputs_exo | self.targets).items(), key=lambda item: item[1], reverse=False))

        self.data = data.select(self.feature.keys()).to_numpy()
        self.params = params

        self.input_width = params['input_width']
        self.label_width = params['label_width']
        self.total_window_legnth = params['input_width'] + params['label_width']

    def __len__(self):
        return len(self.data) - self.total_window_legnth +1

    def __getitem__(self, idx):
        window = self.data[idx:idx + self.total_window_legnth]

        inputs = torch.cat([torch.tensor(window[0:self.input_width, col_idx], dtype=torch.float32).unsqueeze(-1)
                       for feature, col_idx in self.inputs.items()],
                      dim=-1
                      ).transpose(0,1)
        inputs_exo = torch.cat([torch.tensor(window[self.input_width:self.total_window_legnth, col_idx], dtype=torch.float32).unsqueeze(-1)
                           for feature, col_idx in self.inputs_exo.items()],
                          dim=-1
                          ).transpose(0,1)
        targets = torch.cat([torch.tensor(window[self.input_width:self.total_window_legnth, col_idx], dtype=torch.float32).unsqueeze(-1)
                       for feature, col_idx in self.targets.items()],
                      dim=-1
                      ).transpose(0,1)
        return {'inputs':inputs,
                'inputs_exo':inputs_exo,
                'targets':targets}


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, data: pl.DataFrame, params: dict, name:str= 'Loader', drop_last:bool=True):
        dataset = WidowGenerator(data, params['window'])
        super(DataLoader, self).__init__(
            dataset,
            batch_size=params['batch_size'],
            shuffle=params['shuffle'],
            drop_last=drop_last
        )

        inputs_start = 0
        inputs_end = inputs_start + params['window']['input_width']
        inputs_data = data.select(dataset.inputs.keys())[inputs_start:inputs_end].to_numpy().transpose()

        inputs_exo_start = params['window']['input_width']
        inputs_exo_end = inputs_exo_start+params['window']['label_width']
        inputs_exo_data = data.select(dataset.inputs_exo.keys())[inputs_exo_start:inputs_exo_end].to_numpy().transpose()

        targets_start = params['window']['input_width']
        targets_end = targets_start+params['window']['label_width']
        targets_data = data.select(dataset.targets.keys())[targets_start:targets_end].to_numpy().transpose()

        sample_batched = next(iter(self))

        assert abs(inputs_data - sample_batched['inputs'][0].numpy()).max()<1e-7, f'Validation valued for inputs data'
        assert abs(inputs_exo_data - sample_batched['inputs_exo'][0].numpy()).max()<1e-7, f'Validation valued for exo inputs data'
        assert abs(targets_data - sample_batched['targets'][0].numpy()).max()<1e-7, f'Validation valued for targets data'

        self.inputs_shape = sample_batched['inputs'].size()
        self.inputs_exo_shape = sample_batched['inputs_exo'].size()
        self.targets_shape = sample_batched['targets'].size()

        logger.info(f"({data.shape[0]} {name} samples): (group:[#batch, #channels, #sequence]) = (x:[{self.inputs_shape}], x_exo[{self.inputs_exo_shape}], y:[{self.targets_shape}])")

