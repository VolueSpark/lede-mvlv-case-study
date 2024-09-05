from datetime import datetime, timedelta
from lib.lfa import Lfa
import geojson
import os, random

PATH = os.path.dirname(os.path.abspath(__file__))

WORK_PATH = os.path.join(PATH, 'lfa')
MV_PATH = os.path.join(PATH, '../data/topology/silver/mv')
LV_PATH = os.path.join(PATH, '../data/topology/silver/lv')
DATA_PATH = os.path.join(PATH, '../data/ami/silver/meas')

GEOJSON_BACKEND_PATH = os.path.join(PATH, '../data/geojson/bronze/lede.geojson')
GEOJSON_FRONTEND_PATH = os.path.join(PATH, '../frontend/public/assets/lede.geojson')

def get_color_from_value(value):
    """
    Takes a value between 0 and 1 and returns the corresponding color
    between blue (0) and red (1) as an RGB tuple.

    0 -> Blue (0, 0, 255)
    1 -> Red  (255, 0, 0)
    """
    if not 0 <= value <= 1:
        raise ValueError("Input value must be between 0 and 1.")

    # Interpolating between blue (0, 0, 255) and red (255, 0, 0)
    red = int(value * 255)
    green = 0  # Green is constant in this interpolation
    blue = int((1 - value) * 255)

    return f'#{red:02x}{green:02x}{blue:02x}'

def prepare_net_geojson(net):
    if not os.path.exists(GEOJSON_FRONTEND_PATH):
        cfl_mrid, branch_mrid, trafo_mrid  = net.load['cfl_mrid'].to_list(), net.line['mrid'].to_list(), net.trafo['mrid'].to_list()

        cfl_mrid = [cfl.replace('-','') for cfl in cfl_mrid]
        branch_mrid = [branch.replace('-','') for branch in branch_mrid]
        trafo_mrid = [trafo_mrid.replace('-','') for trafo_mrid in trafo_mrid]

        features = []
        with open(GEOJSON_BACKEND_PATH, 'r') as fp:
            gj_backend = geojson.load(fp)
        for feature in gj_backend['features']:
            if feature['properties']['objecttype'] == 'ConformLoad' and feature['properties']['id'] in cfl_mrid:
                feature['properties']['color'] = get_color_from_value(random.random())
                features.append(feature)
            elif feature['properties']['objecttype'] == 'AcLineSegment' and feature['properties']['id'] in branch_mrid:
                feature['properties']['color'] = get_color_from_value(random.random())
                features.append(feature)
            elif feature['properties']['objecttype'] == 'PowerTransformer' and feature['properties']['id'] in trafo_mrid:
                feature['properties']['color'] = get_color_from_value(random.random())
                features.append(feature)
        gj_backend_frontend = geojson.FeatureCollection(features)

        with open(GEOJSON_FRONTEND_PATH, 'w') as fp:
            geojson.dump(gj_backend_frontend, fp)




if __name__ == "__main__":

    lfa = Lfa(
        work_path=WORK_PATH,
        mv_path=MV_PATH,
        lv_path=LV_PATH,
        data_path=DATA_PATH
    )

    # TODO Do better
    prepare_net_geojson(net=lfa.read_net())

    exit(1)

    horizon_hours = 24
    from_date = datetime(year=2023, month=8, day=28)
    to_date = from_date + timedelta(hours=horizon_hours)

    lfa.run_lfa(from_date=from_date, to_date=to_date)

        #lfa.plot(simple=True)
