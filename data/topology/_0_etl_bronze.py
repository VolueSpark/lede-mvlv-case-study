import re, json, shutil, os
import polars as pl

from lib import logger
from lib.lfa import Lfa
from lib.schemas.topology import Topology

PATH = os.path.dirname(__file__)

RAW_PATH = os.path.join(PATH, 'raw')
RAW_REGION_PATH = os.path.join(RAW_PATH, 'region')
RAW_TOPOLOGY_PATH = os.path.join(RAW_PATH, 'topology')

BRONZE_PATH = os.path.join(PATH, 'bronze')
BRONZE_MV_PATH = os.path.join(BRONZE_PATH, 'mv')
BRONZE_LV_PATH = os.path.join(BRONZE_PATH, 'lv')


if __name__ == "__main__":

    shutil.rmtree(BRONZE_PATH, ignore_errors=True)
    os.makedirs(BRONZE_MV_PATH, exist_ok=True)
    os.makedirs(BRONZE_LV_PATH, exist_ok=True)

    region_file_list = os.listdir(RAW_REGION_PATH)
    topology_file_list = os.listdir(RAW_TOPOLOGY_PATH)

    assert len(region_file_list) == 1, f'Regions are exceeding the expected quantity of one'

    region_file = os.path.join(RAW_REGION_PATH, region_file_list[0])
    with open(region_file, 'r') as fp:
        region = Topology(**json.load(fp))
    exit(1)
    for index, topology_file in enumerate(topology_file_list):
        with open(os.path.join(RAW_TOPOLOGY_PATH, topology_file), 'r') as fp:
            try:
                topology = Topology(**json.load(fp))
                Lfa.run_lfa(net=Lfa.create_net(topology=topology))
                slack_bus = topology.slack[0].bus
                region_trafo = [trafo for trafo in region.trafo if trafo.hv_bus == slack_bus]
                assert len(region_trafo) == 1, f'slack bus {slack_bus} not found in admissible regional slack busses'
                region_trafo[0].in_service = topology.trafo[0].in_service = True
            except Exception as e:
                logger.exception(f'[{topology_file}({index} of {len(topology_file_list)})] exception raised. [{e}]')
            else:
                with open(os.path.join(BRONZE_LV_PATH, topology.uuid), 'w') as fp:
                    json.dump(topology.dict(), fp)
                    logger.info(f'[{topology_file}({index} of {len(topology_file_list)})] successfully validated.' )
    with open(os.path.join(BRONZE_MV_PATH, region.uuid), 'w') as fp:
        json.dump(region.dict(), fp)
