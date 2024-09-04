import geojson, os, json, shutil
from typing import List
import polars as pl
import re, json

PATH = os.path.dirname(os.path.abspath(__file__))

TOPOLOGY_PATH = os.path.join(PATH, '../../data/topology/raw/topology')
REGION_PATH = os.path.join(PATH, '../../data/topology/raw/region')

if __name__ == "__main__":

    topology_list = os.listdir(TOPOLOGY_PATH)
    region_list = os.listdir(REGION_PATH)

    region_usagepoints = []
    with open(os.path.join(REGION_PATH, region_list[0]), 'r') as f:
        region_data = json.load(f)
    region_usagepoints.extend(region_data['usagePoints'])

    topology_usagepoints = []
    for topology in topology_list:
        with open(os.path.join(TOPOLOGY_PATH, topology), 'r') as f:
            topology_data = json.load(f)
            topology_usagepoints.extend(topology_data['usagePoints'])

    print(f"MV net has {len(region_usagepoints)} usage points")
    print(f"LV net has {len(topology_usagepoints)} usage points")