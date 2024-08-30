import geojson, os, json, shutil
from typing import List
import polars as pl

PATH = os.path.dirname(os.path.abspath(__file__))

TOPOLOGY_PATH = os.path.join(PATH, '../topology/bronze/topology.json')
GEOJSON_PATH = os.path.join(PATH, 'bronze/lede.geojson')

from lib import _log

if __name__ == "__main__":
    with open(TOPOLOGY_PATH) as f:
        topology = json.load(f)

    trafo_mrid = [trafo['mrid'] for trafo in topology['powerTransformers']]
    branch_mrid = [line['mrid'] for line in topology['acLineSegments']]
    cfl_mrid = [usagepoint['conformLoadId'] for usagepoint in topology['usagePoints'] ]



    with open(GEOJSON_PATH) as f:
        geojson = json.load(f)

    features = []

    for feature in geojson['features']:
        if feature['properties']['@type'] == 'ACLineSegment' and feature['properties']['@id'].split(':')[-1] in branch_mrid:
            features.append(feature)
        elif feature['properties']['@type'] == 'ConformLoad' and feature['properties']['@id'].split(':')[-1] in cfl_mrid:
            features.append(feature)
        elif feature['properties']['@type'] == 'PowerTransformer' and feature['properties']['@id'].split(':')[-1] in trafo_mrid:
            features.append(feature)

    branch_cnt = len([feature for feature in features if feature['properties']['@type']=='ACLineSegment'])
    cfl_cnt = len([feature for feature in features if feature['properties']['@type']=='ConformLoad'])
    trafo_cnt = len([feature for feature in features if feature['properties']['@type']=='PowerTransformer'])
    print(f'Geojson features processed with {branch_cnt} ac line segments; {cfl_cnt} conform loads; {trafo_cnt} power transformers')

    if not os.path.exists(os.path.join(PATH, 'silver')):
        os.makedirs(os.path.join(PATH, 'silver'), exist_ok=True)

    geojson['features'] = features
    with open(os.path.join(PATH, 'silver', 'lede.geojson'), 'w+') as fp:
        json.dump(geojson, fp)

    _log.info(f"Spark-Grid memgraph query resulted in a network model with: {len(topology['usagePoints'])} unique usagepoints")
