from matplotlib import colors
import shutil

import pandapower as pp
import geojson
import os

from lib import logger

PATH = os.path.dirname(os.path.abspath(__file__))

NET_PATH = os.path.join(PATH, '../../backend/lfa/net.sqlite')

GEOJSON_ELEMENTS = ['ConformLoad', 'AcLineSegment', 'PowerTransformer']

BRONZE_PATH = os.path.join(PATH, 'bronze')
SILVER_PATH = os.path.join(PATH, 'silver')

FRONTEND_PATH = os.path.join(PATH, '../../frontend/public/assets')

LOW_THRESHOLD_COLOR = "#BFBFBF"
MED_THRESHOLD_COLOR = "#F7889A"
HIGH_THRESHOLD_COLOR = "#6D3C44"

cmap = colors.LinearSegmentedColormap.from_list("custom_cmap", [LOW_THRESHOLD_COLOR, MED_THRESHOLD_COLOR, HIGH_THRESHOLD_COLOR])

def get_color_from_value(value):
    rgb_color = cmap(value)[:3]  # Get the RGB part (ignore the alpha channel)
    return colors.rgb2hex(rgb_color)


def read_ppnet()-> pp.pandapowerNet:
    if not os.path.exists(NET_PATH):
        raise Exception(f'{NET_PATH} does not exists. First create some pandapower net.sqlite and set <NET_PATH> appropriately.')
    return pp.from_sqlite(NET_PATH)


def element_mrid_for_net(net: pp.pandapowerNet) -> dict:
    element_mrid = {}
    for element in GEOJSON_ELEMENTS:
        if element == 'ConformLoad':
            element_mrid[element] = {
                cfl_mrid.replace('-', ''): net.load.iloc[i]['name']
                for i, cfl_mrid in enumerate( set(net.load['cfl_mrid'].to_list()) )
            }
        elif element == 'PowerTransformer':
            element_mrid[element] = { mrid.replace('-',''):{
                'name':net.trafo.iloc[i]['name'],
                'topology_id':net.trafo.iloc[i]['topology_id']
            } for i, mrid in enumerate(net.trafo['mrid'].to_list())}
        elif element == 'AcLineSegment':
            element_mrid[element] = [mrid.replace('-','') for mrid in net.line['mrid'].to_list()]
    return element_mrid


def element_mrid_feature(features: dict, element_mrid: dict) -> geojson.FeatureCollection:

        feature_collection = []
        for feature in features:
            if feature['properties']['objecttype'] == 'ConformLoad' and feature['properties']['id'] in element_mrid['ConformLoad']:
                feature['properties']['color'] = LOW_THRESHOLD_COLOR
                feature['properties']['value'] = '1.0 p.u'
                feature['properties']['cfl_id'] = feature['properties']['id']
                feature_collection.append(feature)
            elif feature['properties']['objecttype'] == 'AcLineSegment' and feature['properties']['id'] in element_mrid['AcLineSegment']:
                feature['properties']['value'] = '0.0 %'
                feature['properties']['color'] = LOW_THRESHOLD_COLOR
                feature_collection.append(feature)
            elif feature['properties']['objecttype'] == 'PowerTransformer' and feature['properties']['id'] in element_mrid['PowerTransformer'].keys():
                feature['properties']['color'] = LOW_THRESHOLD_COLOR
                feature['properties']['value'] = '0.0 %'
                feature['properties']['name'] = element_mrid['PowerTransformer'][feature['properties']['id']]['name']
                feature['properties']['topology_id'] = element_mrid['PowerTransformer'][feature['properties']['id']]['topology_id']
                feature_collection.append(feature)
            else:
                logger.exception(f"Feature id {feature['properties']['id']} could not be color mapped")
        return geojson.FeatureCollection(feature_collection)


if __name__ == "__main__":
    shutil.rmtree(SILVER_PATH, ignore_errors=True)
    os.makedirs(SILVER_PATH, exist_ok=True)

    net = read_ppnet()
    element_mrid = element_mrid_for_net(net)

    if not os.path.exists(os.path.join(BRONZE_PATH, 'lede.geojson')):
        raise Exception(f"{os.path.join(BRONZE_PATH, 'lede.geojson')} does not exists. First run the script <python data/geojson/_0_etl_bronze.py>.")

    with open(os.path.join(BRONZE_PATH, 'lede.geojson'), 'r') as fp:
        features = geojson.load(fp)['features']

    feature_collection = element_mrid_feature(
        features=features,
        element_mrid=element_mrid)

    with open(os.path.join(SILVER_PATH, 'lede.geojson'), 'w') as fp:
        geojson.dump(feature_collection, fp)

    with open(os.path.join(FRONTEND_PATH, 'lede.geojson'), 'w') as fp:
        geojson.dump(feature_collection, fp)
    with open(os.path.join(FRONTEND_PATH, 'lede.base.geojson'), 'w') as fp:
            geojson.dump(feature_collection, fp)
