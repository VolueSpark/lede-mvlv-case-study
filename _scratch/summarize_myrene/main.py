import geojson, os, json, shutil
from typing import List
import polars as pl
import re, json

PATH = os.path.dirname(os.path.abspath(__file__))

TOPOLOGY_PATH = os.path.join(PATH, '../../data/topology/raw/topology')
REGION_PATH = os.path.join(PATH, '../../data/topology/raw/region')

if __name__ == "__main__":

    usagepoints = []

    topology_list = os.listdir(TOPOLOGY_PATH)
    region_list = os.listdir(REGION_PATH)

    region_usagepoints = []
    for region in region_list:
        with open(os.path.join(REGION_PATH, region), 'r') as f:
            region_data = json.load(f)
            region_usagepoints.extend(region_data['usagePoints'])

            for usagepoint in region_data['usagePoints']:
                usagepoints.append(usagepoint)

    topology_usagepoints = []
    for topology in topology_list:
        with open(os.path.join(TOPOLOGY_PATH, topology), 'r') as f:
            topology_data = json.load(f)
            topology_usagepoints.extend(topology_data['usagePoints'])

            for usagepoint in topology_data['usagePoints']:
                usagepoints.append(usagepoint)

    usagepoints = pl.from_dicts(usagepoints)
    usagepoints.write_parquet(os.path.join(PATH,'usagepoints.parquet'))

    print(f"Detected {usagepoints.n_unique('meterPointId')} meter point id's allocated to {usagepoints.n_unique('conformLoadId')} unique conform loads")



