from datetime import datetime, timedelta
import polars as pl
import os

PATH = os.path.dirname(__file__)
DATA = os.path.join(PATH, '../../backend/lfa/data')

if __name__ == "__main__":
    WINDOW = 48
    MAX_USAGE_PERCENT_LIMIT = 0.7
    FLEX_ASSET_CAPACITY_MAX_LOW = 50
    FLEX_ASSET_CAPACITY_MAX_HIGH = 80

    agg = (pl.scan_parquet(DATA).sort('datetime', descending=False)
                       .group_by_dynamic('datetime', every='1h')
                       .agg((1000*pl.col('p_mw'))
                            .sum()
                            .alias('sum_p_kw'))).collect()

    arg_max = agg['sum_p_kw'].rolling_max(window_size=WINDOW).arg_max()-WINDOW
    from_date = agg[arg_max]['datetime'].item()
    to_date = from_date + timedelta(hours=WINDOW)

    print(f'Extremum from data {from_date} to {to_date}')


