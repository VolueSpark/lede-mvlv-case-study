from datetime import datetime, timedelta
from lib.blob import read
from typing import List
import polars as pl
import os

from lib import _log

PATH = os.path.dirname(os.path.abspath(__file__))
MEASUREMENTS_PATH = os.path.join(PATH, '../../data/ami/silver')


def fetch_predictions(from_date: datetime, days: int=1):
    df = read(from_date=from_date, to_date=from_date + timedelta(days=days-1))
    return (
        df.with_columns(
            pl.col('value_dt').dt.cast_time_unit('us').alias('datetime'))
        .rename({'kWhout': 'p_kwh_out', 'kWhin': 'p_kwh_in', 'kVArhout': 'q_kvarh_out', 'kVArhin': 'q_kvarh_in'})
        .select('datetime', 'meter_id',  'p_kwh_in', 'p_kwh_out', 'q_kvarh_in', 'q_kvarh_out')
    )


def fetch_measurements(from_date: datetime, to_date: datetime=None, meter_list: List[str]=None):
    df = pl.DataFrame()
    for meter in meter_list:
        if os.path.exists(os.path.join(MEASUREMENTS_PATH, meter)):
            df = df.vstack(pl.read_parquet(os.path.join(MEASUREMENTS_PATH, meter)).filter(pl.col('datetime').is_between(from_date, to_date)))
        else:
            _log.warning(f'{os.path.join(MEASUREMENTS_PATH, meter)} has not valid data')
    return df


def fetch(from_date: datetime, days: int=1, path=''):

    folder = f"{from_date}_{from_date+timedelta(days=days)}"
    folder_path = os.path.join(path, folder)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

        p = fetch_predictions(from_date=from_date,
                              days=days)

        non_unique = abs(p.unique(('meter_id', 'datetime')).shape[0] - p.shape[0])
        if non_unique:
            _log.warning(f'Predictions fetched for {folder} as {non_unique} samples being non-unique and will be removed')
            p = p.unique(('meter_id', 'datetime'))

        m = fetch_measurements(from_date=p['datetime'].min(),
                               to_date=p['datetime'].max(),
                               meter_list=p['meter_id'].unique().to_list())

        p.write_parquet(os.path.join(folder, 'raw_predictions'))
        m.write_parquet(os.path.join(folder, 'raw_measurements'))

        missing_meters = list(set(p['meter_id'].to_list())^(set(m['meter_id'].to_list())))
        if len(missing_meters) > 0:
            _log.warning(f'Missing meters: {missing_meters}')
            meters = list(set(p['meter_id'].to_list()).union(set(m['meter_id'].to_list())))
            p = p.filter(pl.col('meter_id').is_in(meters))
            m = m.filter(pl.col('meter_id').is_in(meters))

        (p.sort(by='datetime', descending=False).group_by_dynamic('datetime', every='1h')
        .agg(
            pl.col('p_kwh_in').sum().alias('sum_p_kwh_in'),
            pl.col('p_kwh_out').sum().alias('sum_p_kwh_out'),
            pl.col('q_kvarh_in').sum().alias('sum_q_kvarh_in'),
            pl.col('q_kvarh_out').sum().alias('sum_q_kvarh_out'))
        ).write_parquet(os.path.join(folder, 'predictions'))

        (m.sort(by='datetime', descending=False).group_by_dynamic('datetime', every='1h')
        .agg(
            pl.col('p_kwh_in').sum().alias('sum_p_kwh_in'),
            pl.col('p_kwh_out').sum().alias('sum_p_kwh_out'),
            pl.col('q_kvarh_in').sum().alias('sum_q_kvarh_in'),
            pl.col('q_kvarh_out').sum().alias('sum_q_kvarh_out'))
        ).write_parquet(os.path.join(folder, 'measurements'))
