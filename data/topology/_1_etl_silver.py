import re, json, shutil, os
import polars as pl

from lib import logger
from lib.lfa import Lfa
from lib.schemas.topology import Topology

PATH = os.path.dirname(__file__)

BRONZE_PATH = os.path.join(PATH, 'bronze')
SILVER_PATH = os.path.join(PATH, 'silver')

BRONZE_MV_PATH = os.path.join(BRONZE_PATH, 'mv')
BRONZE_LV_PATH = os.path.join(BRONZE_PATH, 'lv')

SILVER_MV_PATH = os.path.join(SILVER_PATH, 'mv')
SILVER_LV_PATH = os.path.join(SILVER_PATH, 'lv')
#
# Parse spark.grid event files as queried from memgraph and convert base units and aliases for SI units and spark.forecast aliases
#


if __name__ == "__main__":
    shutil.rmtree(SILVER_PATH, ignore_errors=True)

    os.makedirs(SILVER_MV_PATH, exist_ok=True)
    os.makedirs(SILVER_LV_PATH, exist_ok=True)

    mv_file_list = os.listdir(BRONZE_MV_PATH)
    assert len(mv_file_list) == 1, f'Medium voltage are exceeding the expected quantity of one'
    mv_file = os.path.join(BRONZE_MV_PATH, mv_file_list[0])

    exceptions_log =[]
    with open(mv_file, 'r') as fp:
        try:
            mv_topology = Topology(**json.load(fp))
        except Exception as e:
            logger.exception(f'[{mv_file}({1} of {len(mv_file_list)})] exception raised. [{e}]')
            exceptions_log.append(f'[{mv_file}({1} of {len(mv_file_list)})] exception raised. [{e}]')

    lv_file_list = os.listdir(BRONZE_LV_PATH)
    for index, lv_file in enumerate(lv_file_list):
        with open(os.path.join(BRONZE_LV_PATH, lv_file), 'r') as fp:
            try:
                lv_topology = Topology(**json.load(fp))

                assert len(lv_topology.load) > 0, f'topology has no load and will be discarded for lfa'

                Lfa.run_lfa(net=Lfa.create_net(topology=lv_topology))
                slack_bus = lv_topology.slack[0].bus
                mv_topology_trafo = [trafo for trafo in mv_topology.trafo if trafo.hv_bus == slack_bus]

                assert len(mv_topology_trafo) == 1, f'slack bus {slack_bus} not found in admissible mv slack busses'

                mv_topology_trafo[0].in_service = lv_topology.trafo[0].in_service = True
            except Exception as e:
                msg = f'{len(exceptions_log)+1}. Exception raised for {lv_file} ({index} of {len(lv_file_list)} of {lv_topology.__class__.__name__}) [{e}]'
                logger.exception(msg)
                exceptions_log.append(msg)
            else:
                with open(os.path.join(SILVER_LV_PATH, lv_topology.uuid), 'w') as fp:
                    json.dump(lv_topology.dict(), fp)
                    logger.info(f'[{lv_file}({index} of {len(lv_file_list)})] successfully validated.' )

    with open(os.path.join(SILVER_MV_PATH, mv_topology.uuid), 'w') as fp:
        json.dump(mv_topology.dict(), fp)
    with open(os.path.join(SILVER_PATH, 'exceptions.json'), 'w') as fp:
        json.dump(exceptions_log, fp)

