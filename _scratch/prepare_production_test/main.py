from datetime import datetime
from typing import List
import polars as pl
import os, json

PATH = os.path.dirname(__file__)

from lib import logger

METER_SILVER_PATH = os.path.join(PATH, '../../data/ami/silver/meas')
TOPOLOGY_RAW_PATH = os.path.join(PATH, '../../data/topology/raw/topology')

TOPOLOGY_ID = ['fe7db2db-ec48-506c-b342-19c1dd2444ab', 'aaef8b14-7ba6-5a09-8ee2-4629857ae952', 'abcc64e4-bb50-58a3-b5b7-cff746d1856f']
SCENARIO_CNT = 10

time_format = '%Y-%m-%dT%H:%M:%S'

def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, (datetime)):
        return obj.strftime(format=time_format)
    raise TypeError ("Type %s not serializable" % type(obj))

def build_scenario(data: pl.DataFrame, topology_data: dict,  scenario_cnt:int) -> dict:

    #
    # load data is in format net active power [p_kw], reactive power [q_kvar]
    #
    data=(
        data.with_columns(((pl.col('p_kwh_out') - pl.col('p_kwh_in')) ).alias('p_kw'),
                        ((pl.col('q_kvarh_out') - pl.col('q_kvarh_in')) ).alias('q_kvar'))
        .drop('p_kwh_in', 'p_kwh_out', 'q_kvarh_in', 'q_kvarh_out')
    )

    #
    # aggregate meter loads over neighborhood on hourly bass
    #
    data_sum = (
        data
        .sort(by='datetime')
        .group_by_dynamic('datetime', every='1h')
        .agg(
            pl.col(f'p_kw').sum().alias(f'sum_p_kwh'),
            pl.col(f'q_kvar').sum().alias(f'sum_q_kvar')
        )
    )

    #
    # extract scenario endpoints for production and consumption extremums
    #

    meter_id_mrid_map = { up['meterPointId']:up['mrid'] for up in topology_data['usagePoints']}

    production_extremum_points = data_sum.sort(by='sum_p_kwh', descending=False)['datetime'][0:scenario_cnt].to_list()
    consumption_extremum_points = data_sum.sort(by='sum_p_kwh', descending=True)['datetime'][0:scenario_cnt].to_list()

    #
    # generate scenario's along the extreme data points
    #
    scenario = []
    for t in production_extremum_points:

        load = []

        for row in data.filter(pl.col('datetime') == t).iter_rows(named=True):
            load.append(
                {
                    'mrid': meter_id_mrid_map[row['meter_id']],
                    'pKw': row['p_kw'],
                    'qKvar': row['q_kvar'],
                }
            )

        switch = []
        for s in topology_data['switches']:
            switch.append(
                {
                    'mrid': s['mrid'],
                    'isOpen': s['isOpen']
                }
            )

        scenario.append(
            {
                'desc': 'grid import extremum (solar case)',
                'timestamp': t.strftime(time_format),
                'load': load,
                'switch':switch
            }
        )


    for t in consumption_extremum_points:

        load = []

        for row in data.filter(pl.col('datetime') == t).iter_rows(named=True):
            load.append(
                {
                    'mrid': meter_id_mrid_map[row['meter_id']],
                    'pKw': row['p_kw'],
                    'qKvar': row['q_kvar'],
                }
            )

        switch = []
        for s in topology_data['switches']:
            switch.append(
                {
                    'mrid': s['mrid'],
                    'isOpen': s['isOpen']
                }
            )

        scenario.append(
            {
                'desc': 'grid export extremum (peak load)',
                'timestamp': t.strftime(time_format),
                'load': load,
                'switch':switch
            }
        )

    #
    # return List[LfaScenario]
    #
    return scenario


def get_meter_data(meter_list: List[str]) -> pl.DataFrame:
    df = pl.DataFrame()
    for i, meter in enumerate(meter_list):
        if os.path.exists(os.path.join(METER_SILVER_PATH, meter)):
            df = df.vstack(pl.read_parquet(os.path.join(METER_SILVER_PATH, meter)))
            logger.info(f'[{i+1}] {meter} does valid data')
        else:
            logger.warning(f'[{i+1}] {meter} does not have valid data')

    return df

if __name__ == "__main__":

    for topology_id in TOPOLOGY_ID:

        topology_path = os.path.join(TOPOLOGY_RAW_PATH, topology_id)

        with open(topology_path, 'r') as fp:
            topology_data = json.load(fp)

        meter_list =[up['meterPointId'] for up in topology_data['usagePoints']]

        meter_data = get_meter_data(
            meter_list=meter_list
        )

        scenario = build_scenario(
            data=meter_data,
            topology_data=topology_data,
            scenario_cnt=SCENARIO_CNT
        )

        scenario_payload = {
            'topology':topology_data,
            'scenario':scenario,
        }

        with open(os.path.join(PATH, f'{topology_id}.json'), 'w') as fp:
            json.dump(scenario_payload, fp, default=json_serial)