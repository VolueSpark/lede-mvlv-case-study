import json, os, shutil
import polars as pl
import re

from lib.schemas.topology import Topology



PATH = os.path.dirname(__file__)
RAW_PATH = os.path.join(PATH, 'raw')
BRONZE_PATH = os.path.join(PATH, 'bronze')

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.FileHandler(filename=os.path.join(BRONZE_PATH,'bronze.log'), mode='a'))


if __name__ == "__main__":

    shutil.rmtree(BRONZE_PATH, ignore_errors=True)

    if os.path.exists(os.path.join(RAW_PATH, 'Region')):
        os.makedirs(os.path.join(BRONZE_PATH, 'medium_voltage'))

        for file_name in os.listdir(os.path.join(RAW_PATH,'Region')):
            with open(os.path.join(RAW_PATH, 'Region', file_name), 'r') as fp:
                try:
                    topology = Topology(**json.load(fp))
                except Exception as e:
                    logger.exception('NOT WOKRING')




    '''
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