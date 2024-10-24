import geojson, os, json, shutil
from typing import List
import polars as pl
import re

PATH = os.path.dirname(os.path.abspath(__file__))

CIM_PATH = os.path.join(PATH, '../../data/cim/raw/Lede_2024.09.29')
AMI_PATH = os.path.join(PATH, '../../data/ami/silver/meas')

if __name__ == "__main__":
    filepaths = []
    for (dirpath, dirnames, filenames) in os.walk(CIM_PATH):
        filepaths.extend([os.path.join(dirpath, file) for file in filenames])

    usagepoints_mrid = []
    for filepath in filepaths:
        with open(filepath, 'r') as fp:

            if os.path.basename(filepath).endswith('.jsonld'):
                data = json.load(fp)
                for item in data['@graph']:
                    if '@type' in item.keys() and item['@type'] == 'cim:UsagePoint':
                        usagepoints_mrid.append(item['cim:IdentifiedObject.mRID'])

    print(f"{CIM_PATH} has {len(set(usagepoints_mrid))} unique usagepoints mrids")

    ami_meter_point_id = os.listdir(AMI_PATH)

    print(f"{AMI_PATH} has {len(ami_meter_point_id)} unique meter points is")

    coop_meter_point_id = '707057500042745649'
    if coop_meter_point_id not in ami_meter_point_id:
        print(f"{coop_meter_point_id} not in availible AMI sensor list.")
    else:
        print(f"{coop_meter_point_id} is found in the availible AMI sensor list.")

