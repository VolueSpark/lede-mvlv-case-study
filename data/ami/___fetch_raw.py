from datetime import datetime
import os, shutil

from lib.blob import fetch_lede_measurement_ts, fetch_lede_prediction_ts

PATH = os.path.dirname(__file__)
RAW_PATH = os.path.join(PATH, 'raw')
RAW_PRED_PATH = os.path.join(RAW_PATH, 'pred')
RAW_MEAS_PATH = os.path.join(RAW_PATH, 'meas')

FROM_DATE = datetime(year=2023,month=8,day=28)
TO_DATE = datetime(year=2024,month=8,day=28)


if __name__ == '__main__':

    if os.path.exists(RAW_PATH):
        shutil.rmtree(RAW_PATH)
    os.makedirs(RAW_MEAS_PATH, exist_ok=True)

    for date, data in fetch_lede_measurement_ts(from_date=FROM_DATE, to_date=TO_DATE):
        data.write_parquet(os.path.join(RAW_MEAS_PATH, f"{date}.parquet"))

    #for date, data in fetch_lede_prediction_ts(from_date=FROM_DATE, to_date=TO_DATE):
    #    data.write_parquet(os.path.join(RAW_MEAS_PATH, f"{date}.parquet"))