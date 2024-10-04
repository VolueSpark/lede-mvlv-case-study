from lib.ml.plotting import plot_aggregate
from lib.ml.predict import Predict
import polars as pl
import os

PATH = os.path.dirname(__file__)
WORK_PATH = os.path.join(PATH, 'ml')

UUID = '59f5db7a-a41d-5166-90d8-207ca87fecc6'
if __name__ == "__main__":
    data = pl.read_parquet(os.path.join(WORK_PATH, UUID, 'data/gold/test.parquet'))

    (x,y,y_hat) = next(Predict(
        root=WORK_PATH,
        uuid=UUID
    ).predict(data=data[0:72]))

    plot_aggregate(x, y, y_hat)


