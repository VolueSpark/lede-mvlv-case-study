from lib.ml.ml import Ml, Config
import os

PATH = os.path.dirname(__file__)

UUID = '59f5db7a-a41d-5166-90d8-207ca87fecc6'
WORK_PATH = os.path.join(PATH, '../../backend/ml')
METER_DATA_PATH = os.path.join(PATH, '../../data/ami/silver/meas')
TOPOLOGY_DATA_PATH = os.path.join(PATH, '../../data/topology/silver/lv')
GEOJSON_DATA_PATH = os.path.join(PATH, '../../data/geojson/silver')

if __name__ == "__main__":

    ml = Ml(
        Config(
            topology_uuid=UUID,
            work_path=WORK_PATH,
            meter_data_path=METER_DATA_PATH,
            topology_data_path=TOPOLOGY_DATA_PATH,
            geojson_path=GEOJSON_DATA_PATH,
        )
    )