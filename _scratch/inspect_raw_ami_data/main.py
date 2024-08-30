import polars as pl
import os

PATH = os.path.dirname(__file__)

if __name__ == '__main__':
    df = pl.scan_parquet(os.path.join(PATH, '20240822')).select('value_dt','meter_id','kWhout', 'kWhin', 'kVArhout', 'kVArhin').collect()

