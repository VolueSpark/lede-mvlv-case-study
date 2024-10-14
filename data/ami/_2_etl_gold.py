import concurrent.futures, multiprocessing, os, shutil, threading
from pydantic import BaseModel, Field, AliasChoices
from datetime import datetime
import polars as pl
from typing import List

from lib import logger
from lib.lfa.memgraph import UsagePoint

PATH = os.path.dirname(__file__)

SILVER_PATH = os.path.join(PATH, 'silver')
GOLD_PATH = os.path.join(PATH, 'gold')

SILVER_MEAS_PATH = os.path.join(SILVER_PATH, 'meas')

TOPOLOGY_PATH = os.path.join(PATH, '../topology/raw')


class Topology(BaseModel):
    uuid: str = Field(
        validation_alias=AliasChoices('neighbourhoodId', 'regionId', 'uuid')
    )
    load: List[UsagePoint] = Field(alias='usagePoints')


class Statistics(BaseModel):
    max_net_p_kwh: float = Field(default=0.0)
    max_net_q_kvarh: float = Field(default=0.0)
    min_net_p_kwh: float = Field(default=0.0)
    min_net_q_kvarh: float = Field(default=0.0)


def process_batch(**kwargs) -> dict:
    statistics = {}
    for i, meter_id in enumerate(kwargs['folder_sub_list']):
        if os.path.exists(os.path.join(SILVER_MEAS_PATH, meter_id)):
            # out-> grid to energy consumer (grid consumption / grid export)
            #  in-> energy consumer to grid (IPP production / grid import)
            df = pl.read_parquet(os.path.join(SILVER_MEAS_PATH, meter_id)).with_columns(
                (pl.col('p_kwh_out') - pl.col('p_kwh_in')).alias('net_p_kwh'),
                (pl.col('q_kvarh_out') - pl.col('q_kvarh_in')).alias('net_q_kvarh')
            )
            statistics[meter_id] = Statistics(
                max_net_p_kwh=df['p_kwh_out'].max(),
                max_net_q_kvarh=df['net_q_kvarh'].max(),
                min_net_p_kwh=df['net_p_kwh'].min(),
                min_net_q_kvarh=df['net_q_kvarh'].min()
            )
        else:
            statistics[meter_id] = Statistics()

        logger.info(f"[{datetime.utcnow().isoformat()}] (thread {kwargs['thread_idx']} id:{threading.get_native_id()}) Processed meter {meter_id} for gold ETL [{i} of {len(kwargs['folder_sub_list'])}].",
                    color=f"\u001b[38;5;{16 + kwargs['thread_idx']}m")

    return statistics


if __name__ == '__main__':

    if os.path.exists(GOLD_PATH):
        shutil.rmtree(GOLD_PATH)
    os.makedirs(GOLD_PATH)

    metadata = pl.DataFrame()
    for path, folders, files in os.walk(TOPOLOGY_PATH):
        for file in files:
            with open(os.path.join(path, file), 'r') as fp:
                topology = Topology.parse_raw(fp.read())
                for load in topology.load:
                    metadata = metadata.vstack(pl.DataFrame({'uuid': topology.uuid, 'cfl_mrid':load.cfl_mrid, 'mrid': load.mrid, 'meter_id': load.meter_id})).unique('meter_id')

    metadata = metadata.join(metadata.group_by('uuid').agg(
        pl.col('cfl_mrid').n_unique().alias('cnt_cfl_mrid'),
        pl.col('meter_id').n_unique().alias('cnt_meter_id')),
        on='uuid',
        validate='m:1'
    )

    meter_list = metadata['meter_id'].to_list()
    max_workers = multiprocessing.cpu_count()
    batch_size = round(len(meter_list) / max_workers)

    statistics = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_batch, folder_sub_list=batch, thread_idx=i) for i, batch in
                   enumerate([meter_list[x:x + batch_size] for x in range(0, len(meter_list), batch_size)])]
        for future in concurrent.futures.as_completed(futures):
            statistics |= future.result()
    metadata.with_columns(pl.col('meter_id').map_elements(lambda meter_id: statistics[meter_id].dict(), return_dtype=pl.Struct).alias('statistics')).unnest('statistics').write_parquet(os.path.join(GOLD_PATH, 'metadata.parquet'))
