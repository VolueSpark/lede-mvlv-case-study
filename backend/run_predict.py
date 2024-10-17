from lib.ml.plotting import plot
from lib.ml.predict import Predict
import polars as pl
import os

PATH = os.path.dirname(__file__)
WORK_PATH = os.path.join(PATH, 'ml')

UUID = '59f5db7a-a41d-5166-90d8-207ca87fecc6'
#UUID = 'f22a4d55-0655-5bb4-923d-ea1dbec39d58'
if __name__ == "__main__":

    (x,y,y_hat) = next(Predict(
        root=WORK_PATH,
        uuid=UUID
    ).predict())

    plot(x, y, y_hat)


