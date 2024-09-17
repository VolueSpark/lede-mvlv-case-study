import polars as pl
import torch

from lib import logger

class WidowGenerator(torch.utils.data.Dataset):
    def __init__(self, data: pl.DataFrame, params: dict):
        super(WidowGenerator, self).__init__()

        self.inputs = {feature:i for i, feature in enumerate(data.columns) if feature[0:2]=='X_'}
        self.exo_inputs = {feature:i for i, feature in enumerate(data.columns) if feature[0:2]=='X_' and feature[-2:]!='_y'}
        self.targets = {feature:i for i, feature in enumerate(data.columns) if feature[-2:]=='_y'}

        self.features = dict(sorted((self.inputs | self.exo_inputs | self.targets).items(), key=lambda item: item[1], reverse=False))

        self.data = data.select(self.features.keys()).to_numpy()
        self.params = params

        self.input_width = params['input_width']
        self.label_width = params['label_width']
        self.total_window_legnth = params['input_width'] + params['label_width']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        window = self.data[idx:idx + self.total_window_legnth]

        x = torch.cat([torch.tensor(window[0:self.input_width, col_idx], dtype=torch.float32).unsqueeze(-1)
                       for feature, col_idx in self.inputs.items()],
                      dim=-1
                      ).transpose(0,1)
        x_exo = torch.cat([torch.tensor(window[self.input_width:self.total_window_legnth, col_idx], dtype=torch.float32).unsqueeze(-1)
                           for feature, col_idx in self.exo_inputs.items()],
                          dim=-1
                          ).transpose(0,1)
        y = torch.cat([torch.tensor(window[self.input_width:self.total_window_legnth, col_idx], dtype=torch.float32).unsqueeze(-1)
                       for feature, col_idx in self.targets.items()],
                      dim=-1
                      ).transpose(0,1)
        return {'x':x, 'x_exo':x_exo, 'y':y}


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, data: pl.DataFrame, params: dict, name='Loader'):
        dataset = WidowGenerator(data, params['window'])
        super(DataLoader, self).__init__(dataset, batch_size=params['batch_size'], shuffle=params['shuffle'])

        sample_batched = next(iter(self))

        logger.info(f"{name} ({data.shape[0]} samples): (name:[#batch, #channels, #sequence]) = (x:[{sample_batched['x'].size()}], x_exo[{sample_batched['x_exo'].size()}], y:[{sample_batched['y'].size()}])")

        input_start = 0
        input_end = input_start + params['window']['input_width']
        input_data = data.select(dataset.inputs.keys())[input_start:input_end].to_numpy().transpose()

        exo_input_start = params['window']['input_width']
        exo_input_end = exo_input_start+params['window']['label_width']
        exo_input_data = data.select(dataset.exo_inputs.keys())[exo_input_start:exo_input_end].to_numpy().transpose()

        target_start = params['window']['input_width']
        target_end = target_start+params['window']['label_width']
        target_data = data.select(dataset.targets.keys())[target_start:target_end].to_numpy().transpose()

        assert abs(input_data - sample_batched['x'][0].numpy()).max()<1e-7, f'Validation valued for input data'
        assert abs(exo_input_data - sample_batched['x_exo'][0].numpy()).max()<1e-7, f'Validation valued for exo input data'
        assert abs(target_data - sample_batched['y'][0].numpy()).max()<1e-7, f'Validation valued for target data'
