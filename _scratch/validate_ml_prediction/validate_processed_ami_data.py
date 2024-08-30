import polars as pl
import os

from plotting import plot_processed_meter

PATH = os.path.dirname(os.path.abspath(__file__))

BRONZE_AMI_PATH = lambda meter_id: os.path.join(PATH, f'../../data/ami/bronze/{meter_id}')
SILVER_AMI_PATH = lambda meter_id: os.path.join(PATH, f'../../data/ami/silver/{meter_id}')

def plot_meter(meter_id: str):
    pass

if __name__ == "__main__":
    meter_id = '707057500041450476'

    df_bronze = (pl.scan_parquet(BRONZE_AMI_PATH(meter_id=meter_id)).collect()
                 .with_columns(pl.col('value_dt').dt.replace_time_zone(None).dt.datetime(),
                               pl.col('kWhout').diff(),
                               pl.col('kWhin').diff(),
                               pl.col('kVArhout').diff(),
                               pl.col('kVArhin').diff())
                 .rename({'value_dt':'datetime', 'kWhin':'p_kwh_in', 'kWhout':'p_kwh_out', 'kVArhout':'q_kvarh_out', 'kVArhin':'q_kvarh_in'}))[0:100]
    df_silver = pl.read_parquet(SILVER_AMI_PATH(meter_id=meter_id))[0:100]

    plot_processed_meter(original=df_bronze, processed=df_silver, feature='p_kwh_out')

