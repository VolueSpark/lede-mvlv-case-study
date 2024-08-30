from lib.topology import Topology
from lib.lfa import Lfa
import os, json, re

PATH = os.path.dirname(os.path.abspath(__file__))


def validate_meter_id(meter_id: str) -> bool:
    return re.match(r'^\d{18}$', meter_id)


if __name__ == "__main__":
    lede_geojson_path = '../../data/geojson/bronze/lede.geojson'
    topology_path = '/home/phillip/repo/spark-roadmap/778-lede-mvlv-case-study/data/topology/bronze/topology.json'

    with open(topology_path) as f:
        topology = json.load(f)

    branch_mrid = [line['mrid'] for line in topology['acLineSegments']]
    bus_mrid = [bus['mrid'] for bus in topology['connectivityNodes']]
    cfl_mrid = [usagepoint['conformLoadId'] for usagepoint in topology['usagePoints'] ]

    topology_mrid = {'branch': {'count': len(branch_mrid),
                                'mrid': branch_mrid},
                     'bus': {'count': len(bus_mrid),
                             'mrid': bus_mrid},
                     'cfl': {'count': len(set(cfl_mrid)),
                             'mrid': list(set(cfl_mrid))}}

    with open(lede_geojson_path) as f:
        lede_geojson = json.load(f)

    #
    # extract topology mrid id
    #
    branch_mrid = [feature['properties']['@id'].split(':')[-1] for feature in lede_geojson['features'] if (feature['properties']['@type'] == 'cim.ACLineSegment' or feature['properties']['@type'] == 'cim:ACLineSegment')]
    cfl_mrid = [feature['properties']['@id'].split(':')[-1] for feature in lede_geojson['features'] if feature['properties']['@type'] == 'cim.ConformLoad']

    geojson_mrid = {'branch': {'count': len(branch_mrid),
                               'mrid': branch_mrid},
                    'cfl': {'count': len(set(cfl_mrid)),
                            'mrid': list(set(cfl_mrid))}}

    print(f"Topology has {topology_mrid['cfl']['count']} unique conform loads. Geojson has {geojson_mrid['cfl']['count']} has  unique conform loads")
    print(f"Toplogy and Geojson has {len(set(topology_mrid['cfl']['mrid']).intersection(set(geojson_mrid['cfl']['mrid'])))} common conform loads\n")

    print(f"Topology has {topology_mrid['branch']['count']} ac line segments. Geojson has {geojson_mrid['branch']['count']} has ac line segments.")
    print(f"Toplogy and Geojson has {len(set(topology_mrid['branch']['mrid']).intersection(set(geojson_mrid['branch']['mrid'])))} common ac line segments.\n")
    #
    # extract geojson mrid id's
    #
