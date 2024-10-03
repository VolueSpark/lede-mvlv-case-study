from sklearn.preprocessing import MinMaxScaler
from typing import List, Tuple
import numpy as np
import polars as pl
import pickle


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


    @staticmethod
    def load(pickle_path: str):
        with open(pickle_path, 'rb') as fp:
           return pickle.load(fp)

    def save(self, pickle_path: str):
        with open(pickle_path, 'wb') as fp:
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
