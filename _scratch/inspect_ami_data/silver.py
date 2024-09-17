import polars as pl
import os

PATH = os.path.dirname(__file__)

if __name__ == '__main__':
    meter_id = '707057500041323503'
    path = f'/home/phillip/repo/volue.spark/lede-mvlv-case-study/data/ami/silver/meas/{meter_id}'

    df = pl.scan_parquet(path).collect()

    print(f"Silver meter {meter_id} has {df.shape[0]} samples range from {df.select(pl.col('datetime')).min().item().isoformat()} to {df.select(pl.col('datetime')).max().item().isoformat()}")