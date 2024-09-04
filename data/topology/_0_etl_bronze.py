import re, json, shutil, os
import polars as pl

from lib import logger
from lib.schemas.memgraph import MemgraphEvent

PATH = os.path.dirname(__file__)

RAW_PATH = os.path.join(PATH, 'raw')
BRONZE_PATH = os.path.join(PATH, 'bronze')

RAW_REGION_PATH = os.path.join(RAW_PATH, 'region')
RAW_TOPOLOGY_PATH = os.path.join(RAW_PATH, 'topology')

BRONZE_MV_PATH = os.path.join(BRONZE_PATH, 'mv')
BRONZE_LV_PATH = os.path.join(BRONZE_PATH, 'lv')

#
# Parse spark.grid event files as queried from memgraph and convert base units and aliases for SI units and spark.forecast aliases
#


if __name__ == "__main__":
    shutil.rmtree(BRONZE_PATH, ignore_errors=True)

    os.makedirs(BRONZE_MV_PATH, exist_ok=True)
    os.makedirs(BRONZE_LV_PATH, exist_ok=True)

    region_file_list = os.listdir(RAW_REGION_PATH)
    topology_file_list = os.listdir(RAW_TOPOLOGY_PATH)

    for index, region_file in enumerate(region_file_list):
        region_src_path = os.path.join(RAW_REGION_PATH, region_file)
        with open(region_src_path, 'r') as fp:
            region_data = MemgraphEvent(**json.load(fp))

        mv_dst_path = os.path.join(BRONZE_MV_PATH, region_data.uuid)
        with open(mv_dst_path, 'w+') as fp:
            json.dump(region_data.dict(), fp)

        logger.info(f'[{index+1} of {len(region_file_list)}] Parsed region event {region_src_path} and saved to {mv_dst_path}')

    for index, topology_file in enumerate(topology_file_list):
        topology_src_path = os.path.join(RAW_TOPOLOGY_PATH, topology_file)
        with open(topology_src_path, 'r') as fp:
            topology_data = MemgraphEvent(**json.load(fp))

        lv_dst_path = os.path.join(BRONZE_LV_PATH, topology_data.uuid)
        with open(lv_dst_path, 'w+') as fp:
            json.dump(topology_data.dict(), fp)

        logger.info(f'[{index+1} of {len(topology_file_list)}] Parsed topology event {topology_src_path} and saved to {lv_dst_path}')
