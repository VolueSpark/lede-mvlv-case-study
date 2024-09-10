import polars as pl
import os

PATH = os.path.dirname(__file__)
METADATA = os.path.join(PATH, '../lfa/metadata.parquet')
DATA = os.path.join(PATH, '../lfa/data')

if __name__ == "__main__":
    EXTREMUM_POINT_COUNT = 100
    MAX_USAGE_PERCENT_LIMIT = 0.7
    FLEX_ASSET_CAPACITY_MAX_LOW = 50
    FLEX_ASSET_CAPACITY_MAX_HIGH = 80

    flex_assets = (
        pl.read_parquet(METADATA)
        .filter(pl.col('p_kw_max')
                .is_between(FLEX_ASSET_CAPACITY_MAX_LOW, FLEX_ASSET_CAPACITY_MAX_HIGH))
        .with_columns(
            max_usage_limit_kwh=pl.col('p_kw_max') * MAX_USAGE_PERCENT_LIMIT,
            max_usage_limit_kvarh=pl.col('q_kvar_max') * MAX_USAGE_PERCENT_LIMIT,
        )
    ).select(['uuid','meter_id','max_usage_limit_kwh', 'max_usage_limit_kvarh']).write_parquet(os.path.join(PATH, 'flex_assets.parquet'))

    extremum_points = (pl.scan_parquet(DATA).sort('datetime', descending=False)
                 .group_by_dynamic('datetime', every='1h')
                 .agg((1000*pl.col('p_mw'))
                      .sum()
                      .alias('sum_p_kw'))
                 .sort('sum_p_kw', descending=True)[0:EXTREMUM_POINT_COUNT]
    ).collect().sort('datetime', descending=False).write_parquet(os.path.join(PATH, 'extremum_points.parquet'))