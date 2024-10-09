import geojson, os, json, shutil

PATH = os.path.dirname(os.path.abspath(__file__))

RAW_PATH = os.path.join(PATH, 'raw')
BRONZE_PATH = os.path.join(PATH, 'bronze')

FEATURE_LIST = ['AcLineSegment', 'ConformLoad', 'PowerTransformer']

if __name__ == "__main__":
    shutil.rmtree(BRONZE_PATH, ignore_errors=True)
    os.makedirs(BRONZE_PATH, exist_ok=True)

    features = []
    for file in os.listdir(RAW_PATH):
        file_path = os.path.join(RAW_PATH, file)
        with open(file_path, 'r') as f:
            gj = geojson.load(f)

            for feature in gj['features']:
                if feature['type'] == 'Feature' and 'properties' in feature.keys() and 'objecttype' in feature['properties']:
                    if feature['properties']['objecttype'] in FEATURE_LIST:
                        feature['properties']['color'] = '#000000'
                        features.append (feature)

    gj_new = geojson.FeatureCollection(features)
    with open(os.path.join(BRONZE_PATH, 'lede.geojson'), 'w') as f:
        geojson.dump(gj_new, f)


