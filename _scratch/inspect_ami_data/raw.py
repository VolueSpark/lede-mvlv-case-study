import polars as pl
import os

PATH = os.path.dirname(__file__)

if __name__ == '__main__':
    folder = '20220101'
    path = os.path.join(PATH, folder)

    df = pl.scan_parquet(path).select('value_dt','meter_id','kWhout', 'kWhin', 'kVArhout', 'kVArhin').collect()

    print(f"{path} has {df.shape[0]} samples with {df.n_unique('meter_id')} unique meters")

