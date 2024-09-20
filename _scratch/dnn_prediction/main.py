from lib.ml.ml import Ml
import os

PATH = os.path.dirname(__file__)

UUID = '59f5db7a-a41d-5166-90d8-207ca87fecc6'
WORK_PATH = os.path.join(PATH, '../../backend/ml')

if __name__ == "__main__":
    ml = Ml(
        uuid=UUID,
        work_dir=WORK_PATH
    )

    ml.create()
    ml.train()
