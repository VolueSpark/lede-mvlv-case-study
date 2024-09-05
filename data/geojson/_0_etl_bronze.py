import geojson, os, json, shutil
import matplotlib.pyplot as plt
from typing import List
import polars as pl
import re

PATH = os.path.dirname(os.path.abspath(__file__))

RAW_PATH = os.path.join(PATH, 'raw')
BRONZE_PATH = os.path.join(PATH, 'bronze')

FEATURE_LIST = ['AcLineSegment', 'ConformLoad', 'PowerTransformer']


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


if __name__ == "__main__":
    shutil.rmtree(BRONZE_PATH, ignore_errors=True)
    os.makedirs(BRONZE_PATH, exist_ok=True)

    cmap = plt.get_cmap('coolwarm')

    features = []
    for file in os.listdir(RAW_PATH):
        file_path = os.path.join(RAW_PATH, file)
        with open(file_path, 'r') as f:
            gj = geojson.load(f)
            for feature in gj['features']:
                if feature['type'] == 'Feature' and 'properties' in feature.keys() and 'objecttype' in feature['properties']:
                    if feature['properties']['objecttype'] in FEATURE_LIST:
                        feature['properties']['color'] = get_color_from_value(0)
                        features.append (feature)

    with open(os.path.join(PATH, 'preamble.geojson'), 'r') as fp:
        gj_new = json.load(fp)
    gj_new['features'] = features
    gj_new = geojson.FeatureCollection(features)
    with open(os.path.join(BRONZE_PATH, 'lede.geojson'), 'w') as f:
        geojson.dump(gj_new, f)

