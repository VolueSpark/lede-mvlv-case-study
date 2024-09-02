import json, os, shutil
import polars as pl
import re, json, shutil

from lib.schemas.topology import Topology
from lib.lfa import Lfa


PATH = os.path.dirname(__file__)

RAW_PATH = os.path.join(PATH, 'raw')
RAW_REGION_PATH = os.path.join(RAW_PATH, 'region')
RAW_TOPOLOGY_PATH = os.path.join(RAW_PATH, 'topology')

BRONZE_PATH = os.path.join(PATH, 'bronze')
BRONZE_REGION_PATH = os.path.join(BRONZE_PATH, 'region')
BRONZE_TOPOLOGY_PATH = os.path.join(BRONZE_PATH, 'topology')


from lib import logger

if __name__ == "__main__":

    shutil.rmtree(BRONZE_PATH, ignore_errors=True)
    os.makedirs(BRONZE_REGION_PATH, exist_ok=True)
    os.makedirs(BRONZE_TOPOLOGY_PATH, exist_ok=True)

    region_file_list = os.listdir(RAW_REGION_PATH)
    topology_file_list = os.listdir(RAW_TOPOLOGY_PATH)

    assert len(region_file_list) == 1, f'Regions are exceeding the expected quantity of one'

    with open(os.path.join(RAW_REGION_PATH, region_file_list[0]), 'r') as fp:
        region = Topology(**json.load(fp))
    for index, topology in enumerate(topology_file_list):
        with open(os.path.join(RAW_TOPOLOGY_PATH, topology), 'r') as fp:
            try:
                topology = Topology(**json.load(fp))
                Lfa.run_lfa(net=Lfa.create_net(topology=topology))
                slack_bus = topology.slack[0].bus
                region_trafo = [trafo for trafo in region.trafo if trafo.hv_bus == slack_bus]
                assert len(region_trafo) == 1, f'slack bus {slack_bus} not found in admissible regional slack busses'
                region_trafo[0].in_service = topology.trafo[0].in_service = True
            except Exception as e:
                logger.exception(f'[{topology}({index} of {len(topology_file_list)})] exception raised. [{e}]')
            else:
                with open(os.path.join(BRONZE_TOPOLOGY_PATH, topology.uuid), 'w') as fp:
                    json.dump(topology.dict(by_alias=True), fp)
                    logger.info(f'[{topology}({index} of {len(topology_file_list)})] successfully validated.' )
    with open(os.path.join(BRONZE_REGION_PATH, region.uuid), 'w') as fp:
        json.dump(region.dict(by_alias=True), fp)

    '''
    if os.path.exists(REGION_PATH):
        os.makedirs(os.path.join(BRONZE_PATH, 'medium_voltage'))

        with open(os.path.join(self.mv_path, os.listdir(self.mv_path)[0]), 'r') as fp:
            mv_topology = Topology(**json.load(fp))

        file_list = os.listdir(REGION_PATH)
        for index, file_name in enumerate(file_list):
            with open(os.path.join(REGION_PATH, file_name), 'r') as fp:
                try:
                    region = Topology(**json.load(fp))
                    shutil.copyfile(src=os.path.join(REGION_PATH, file_name), dst=os.path.join(BRONZE_PATH, f'medium_voltage/{topology.uuid}'))
                    logger.info(f'[{index+1} of {len(file_list)}] Processing {os.path.join(REGION_PATH, file_name)} succeeded.')
                except Exception as e:
                    logger.exception(f'[{index+1} of {len(file_list)}] Processing {os.path.join(REGION_PATH, file_name)} failed. [{e}]')

    if os.path.exists(TOPOLOGY_PATH):
        os.makedirs(os.path.join(BRONZE_PATH, 'low_voltage'))

        file_list = os.listdir(TOPOLOGY_PATH)
        for index, file_name in enumerate(file_list):
            with open(os.path.join(TOPOLOGY_PATH, file_name), 'r') as fp:
                try:
                    topology = Topology(**json.load(fp))
                    shutil.copyfile(src=os.path.join(TOPOLOGY_PATH, file_name), dst=os.path.join(BRONZE_PATH, f'low_voltage/{topology.uuid}'))
                    logger.info(f'[{index+1} of {len(file_list)}] Processes {os.path.join(TOPOLOGY_PATH, file_name)}')
                except Exception as e:
                    logger.exception(f'[{index+1} of {len(file_list)}] Processing {os.path.join(TOPOLOGY_PATH, file_name)} failed. [{e}]')




    
    if os.path.exists(BRONZE_PATH):

    os.makedirs(os.path.join(BRONZE_PATH, 'low_voltage'), exist_ok=True)
    os.makedirs(os.path.join(BRONZE_PATH,'medium_voltage'), exist_ok=True)

    exceptions = []

    for index, topology_name in enumerate(os.listdir(os.path.join(RAW_PATH, 'Topology'))):
        try:
            with open(os.path.join(RAW_PATH, 'Topology', topology_name), 'r') as fp:
                data = json.load(fp)
                topology = Topology(**data)

                _log.info(f'[{index}] Processed {topology_name}')
        except Exception as e:
            exceptions.append({topology_name: e.__str__()})
            _log.exception(f'[{index}] Exception raised for {topology_name}')

    with open(os.path.join(BRONZE_PATH, 'low_voltage_exceptions.json'), 'w') as fp:
        json.dump(exceptions, fp)
    '''