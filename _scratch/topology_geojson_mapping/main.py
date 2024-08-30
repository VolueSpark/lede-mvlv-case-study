from lib.topology import Topology
from lib.lfa import Lfa
import polars as pl
import os, json

PATH = os.path.dirname(os.path.abspath(__file__))

topology_path = os.path.join(PATH, f'../../topology/silver/topology.json')
geojson_path = os.path.join(PATH, f'../../cim/bronze/geojson.json')

if __name__ == "__main__":

    with open(topology_path, 'r') as fp:
        topology_data = json.load(fp)

    with open(geojson_path, 'r') as fp:
        geojson_data = json.load(fp)

    geojson = {}
    geojson['branch_mrid'] = [feature['properties']['@id'].split(':')[-1] for feature in geojson_data['features'] if feature['properties']['@type'] == 'cim.ACLineSegment']
    geojson['conformload_mrid'] = conform_load_mrid = [ feature['properties']['@id'].split(':')[-1]  for feature in geojson_data['features'] if feature['properties']['@type'] == 'cim.ConformLoad']
    geojson['bus_mrid']

