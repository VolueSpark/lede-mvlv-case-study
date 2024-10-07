from datetime import datetime
import polars as pl
import os, json

from lib import logger

class DataLoader():
    def __init__(
            self,
            lv_path: str,
            data_path: str,
            work_path: str
    ):

        self.data_path = os.path.join(work_path, 'data')
        os.makedirs(self.data_path, exist_ok=True)

        topology_list = os.listdir(lv_path)
        metadata = pl.DataFrame()
        for j, topology_j in enumerate(topology_list):
            if not os.path.exists(os.path.join(work_path, 'data', f"{topology_j}.parquet")):
                with open(os.path.join(lv_path, topology_j), 'r') as fp:

                    meter_list = {load['meter_id']:load['cfl_mrid'] for load in json.load(fp)['load']}

                    df = pl.DataFrame()
                    for i, (meter_i, cfl_mrid_i) in enumerate(meter_list.items()):
                        if os.path.exists(os.path.join(data_path, meter_i)):
                            df = df.vstack(pl.read_parquet(os.path.join(data_path, meter_i)).with_columns(pl.lit(cfl_mrid_i).alias('cfl_mrid')))
                        else:
                            logger.warning(f"[meter {i+1} of {len(meter_list)}] {topology_j} has no data for meter {meter_i}")

                    if df.shape[0]:
                        df=(
                            df.with_columns(((pl.col('p_kwh_out') - pl.col('p_kwh_in')) / 1e3).alias('p_mw'),
                                            ((pl.col('q_kvarh_out') - pl.col('q_kvarh_in')) / 1e3).alias('q_mvar'))
                            .drop('p_kwh_in', 'p_kwh_out', 'q_kvarh_in', 'q_kvarh_out')
                        )

                        metadata = metadata.vstack(
                            df.select('meter_id', 'cfl_mrid').unique().join(
                                (
                                    df.group_by('meter_id')
                                    .agg(
                                        (pl.col('p_mw').min() * 1000).round(1).alias('p_kw_min'),
                                        (pl.col('p_mw').max() * 1000).round(1).alias('p_kw_max'),
                                        (pl.col('p_mw').mean() * 1000).round(1).alias('p_kw_mean'),
                                        (pl.col('q_mvar').min() * 1000).round(1).alias('q_kvar_min'),
                                        (pl.col('q_mvar').max() * 1000).round(1).alias('q_kvar_max'),
                                        (pl.col('q_mvar').mean() * 1000).round(1).alias('q_kvar_mean'),
                                        pl.lit(topology_j).alias('uuid')
                                    ).select('uuid', 'meter_id', 'p_kw_min', 'p_kw_mean', 'p_kw_max', 'q_kvar_min', 'q_kvar_mean', 'q_kvar_max')
                                ), on='meter_id', validate='1:1')
                        )

                        df.write_parquet(os.path.join(self.data_path, f"{topology_j}.parquet"))

                        logger.info(f"[topology {j+1} of {len(topology_list)}] {topology_j} has been processed with {df.n_unique('meter_id')} unique meters")
                    else:
                        logger.exception(f"[topology {j+1} of {len(topology_list)}] {topology_j} has no available data")

        if not metadata.is_empty():
            metadata.write_parquet(os.path.join(work_path, f"metadata.parquet"))

    def load_profile_iter(
            self,
            from_date: datetime,
            to_date: datetime = None,
            step_every=1
    ):
        if to_date is None:
            to_date = from_date

        data_list = os.listdir(self.data_path)
        data = pl.DataFrame()
        for data_file in data_list:
            data = data.vstack(
                pl.read_parquet(os.path.join(self.data_path, data_file))
                .filter(
                    pl.col('datetime').is_between(from_date, to_date)
                )
            )

        # pl.read_parquet(self.data_path).group_by('meter_id').agg(peak_s_mva=pl.col('s_mva').max()).sort(by='peak_s_mva', descending=True)
        for batch in data.sort('datetime', descending=False).group_by_dynamic('datetime', every=f'{step_every}h', period=f'1h'):
            yield batch[1]