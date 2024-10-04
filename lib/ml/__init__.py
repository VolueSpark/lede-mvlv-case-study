from sklearn.preprocessing import MinMaxScaler
from typing import List
import numpy as np
import polars as pl
import pickle, os, json
import torch, time
from pydantic import BaseModel, Field

from lib import logger


class Scaler(MinMaxScaler):

    def __init__(self):
        super().__init__()

    def fit(self, data: np.array, features: dict):
        self.features=features
        super().fit(data)

    def transform(self, data: np.array, features: dict = None):
        if features is None:
            return super().transform(data)
        # partial transformation based in feature set
        scale_ = [self.scale_[index] for feature, index in features.items()]
        min_ = [self.min_[index] for feature, index in features.items()]
        return data * scale_ + min_

    def inverse_transform(self, data: np.array, features: dict = None):
        if features is None:
            return super().inverse_transform(data)
        # partial transformation based in feature set
        scale_ = [self.scale_[index] for feature, index in features.items()]
        min_ = [self.min_[index] for feature, index in features.items()]
        return (data - min_)/scale_

    def load(self, pickle_path: str):
        with open(os.path.join(pickle_path, 'scaler.pkl'), 'rb') as fp:
            self = pickle.load(fp)
            return self

    def save(self, pickle_path: str):
        with open(os.path.join(pickle_path, 'scaler.pkl'), 'wb') as fp:
            pickle.dump(self, fp, pickle.HIGHEST_PROTOCOL)



class Split:
    def __new__(self, data: pl.DataFrame, split: List[float]):
        n = data.shape[0]
        split_n = [ int(n*split_i)for split_i in split]

        train = data[:split_n[0]]
        val = data[split_n[0]:split_n[1]]
        test = data[split_n[1]:]

        features = {feature:index for index, feature in enumerate(train.drop('t_timestamp').columns)}

        return features, train.drop('t_timestamp').to_numpy(), val.drop('t_timestamp').to_numpy(), test

class Ascii:
    RETURN = '\033[F\r'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    END = '\033[0m'

class OuterProgress(BaseModel):
    elapsed_time: float=Field(default=0.0)
    progress_pct: float=Field(default=0.0)
    loss: float=Field(default=0.0)
    acc: float=Field(default=0.0)
    vloss: float=Field(default=0.0)
    vacc: float=Field(default=0.0)

class InnerProgress(BaseModel):
    it_per_sec:  float=Field(default=0.0)
    progress_pct: float=Field(default=0.0)
    loss: float=Field(default=0.0)
    acc: float=Field(default=0.0)

class ProgressBar:
    def __init__(self):
        self.train = OuterProgress()
        self.epoch_train = InnerProgress()
        self.epoch_val = InnerProgress()

    def update(
            self,
            tag: str,
            elapsed_time: float,
            it_per_sec: float,
            progress_pct: float,
            loss: float,
            acc: float,
            vloss: float = None,
            vacc: float = None
    ):
        if tag == 'epoch':
            print(f"{Ascii.RETURN}{Ascii.BLUE}EPOCH ({progress_pct:3.0f}%): loss={loss:.2e}; acc={acc:.2e}; vloss={vloss:.2e}; vacc={vacc:.2e}  [{elapsed_time:6.2f} elapsed time]{Ascii.END} "
                                f"{Ascii.GREEN}TRAIN ({self.epoch_train.progress_pct:3.0f}%): loss={self.epoch_train.loss:.2e}; acc={self.epoch_train.acc:.2e} [{self.epoch_train.it_per_sec:5.2f} it/sec]{Ascii.END} "
                                f"{Ascii.YELLOW}VAL ({self.epoch_val.progress_pct:3.0f}%): loss={self.epoch_val.loss:.2e}; acc={self.epoch_val.acc:.2e} [{self.epoch_val.it_per_sec:5.2f} it/sec]{Ascii.END}", )

        else:
            if tag == 'train':
                self.epoch_train = InnerProgress(
                    it_per_sec=it_per_sec,
                    progress_pct=progress_pct,
                    loss=loss,
                    acc=acc,
                )
            else:
                self.epoch_val = InnerProgress(
                    it_per_sec=it_per_sec,
                    progress_pct=progress_pct,
                    loss=loss,
                    acc=acc,
                )
            print(f"{Ascii.RETURN}{Ascii.GREEN}TRAIN ({self.epoch_train.progress_pct:3.0f}%): loss={self.epoch_train.loss:.2e}; acc={self.epoch_train.acc:.2e} [{self.epoch_train.it_per_sec:5.2f} it/sec]{Ascii.END} "
                          f"{Ascii.YELLOW}VAL ({self.epoch_val.progress_pct:3.0f}%): loss={self.epoch_val.loss:.2e}; acc={self.epoch_val.acc:.2e} [{self.epoch_val.it_per_sec:5.2f} it/sec]{Ascii.END}", end='')

pbar = ProgressBar()


def decorator_epoch(func):
    def inner(self, *args, **kwargs):

        ave_loss = 0
        ave_acc = 0

        t0 = time.time()
        for result in func(self, *args, **kwargs):

            (global_index, loss, acc) = result

            elapsed_time = (time.time()-t0)

            if func.__name__ == 'epoch_training':
                tag = 'train'
                local_index = (global_index%(self.train_loader.meta.loader_depth))+1
                it_per_sec = local_index/elapsed_time
                progress_pct = local_index/self.train_loader.meta.loader_depth*100
            elif func.__name__ == 'epoch_validation':
                tag = 'val'
                local_index = (global_index%(self.val_loader.meta.loader_depth))+1
                it_per_sec = local_index/elapsed_time
                progress_pct = local_index/self.val_loader.meta.loader_depth*100
            else:
                raise NotImplementedError(f'{func.__name__} does not support progress bar evaluation')

            pbar.update(
                tag=tag,
                elapsed_time=elapsed_time,
                it_per_sec=it_per_sec,
                progress_pct=progress_pct,
                loss=loss,
                acc=acc
            )

            self.writer.add_scalar(f'Loss/{tag}', loss, global_index)
            self.writer.add_scalar(f'Acc/{tag}', acc, global_index)

            ave_loss = (loss + global_index*ave_loss)/(global_index+1)
            ave_acc = (acc + global_index*ave_acc)/(global_index+1)
        return (ave_loss, ave_acc)
    return inner


def decorate_train(func):

    def inner(self, *args, **kwargs):

        t0 = time.time()
        for result in func(self, *args, **kwargs):

            (global_index, loss, acc, vloss, vacc) = result

            elapsed_time = (time.time()-t0)

            if func.__name__ == 'train':
                local_index = global_index+1
                tag = 'epoch'
                it_per_sec = elapsed_time/local_index
                progress_pct = local_index/self.epoch_cnt*100
            else:
                raise NotImplementedError(f'{func.__name__} does not support progress bar evaluation')

            pbar.update(
                tag=tag,
                elapsed_time=elapsed_time,
                it_per_sec=it_per_sec,
                progress_pct=progress_pct,
                loss=loss,
                acc=acc,
                vloss=vloss,
                vacc=vacc
            )

            self.writer.add_scalars(f'Loss/{tag}', {'train': loss, 'val': vloss}, global_index)
            self.writer.add_scalars(f'Acc/{tag}', {'train': acc, 'val': vacc}, global_index)

        model_path = os.path.join(self.artifacts_path, f'state_dict.pth')
        torch.save(self.model.state_dict(), model_path)
        with open(os.path.join(self.artifacts_path, f'meta.json'), 'w') as fp:
            json.dump({
                'library': self.model.__module__,
                'class': self.model.__class__.__name__,
                'shape':
                    {
                        'input': self.train_loader.meta.input_shape,
                        'exo': self.train_loader.meta.input_exo_shape,
                        'target': self.train_loader.meta.target_shape
                    },
                'features':
                    {
                        'input': self.train_loader.meta.input_features,
                        'exo': self.train_loader.meta.input_exo_features,
                        'target': self.train_loader.meta.target_features
                    }
            }, fp)

        logger.info(f'Pytorch model saved to: {model_path}')

    return inner
