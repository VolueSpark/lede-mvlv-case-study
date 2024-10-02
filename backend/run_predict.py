from lib.ml.predict import Predict
import os

PATH = os.path.dirname(__file__)
WORK_PATH = os.path.join(PATH, 'ml')

UUID = '59f5db7a-a41d-5166-90d8-207ca87fecc6'
if __name__ == "__main__":
    p = Predict(
        root=WORK_PATH,
        uuid=UUID
    )

    p.predict()