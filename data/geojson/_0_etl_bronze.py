import geojson, os, json, shutil
from typing import List
import polars as pl
import re

PATH = os.path.dirname(os.path.abspath(__file__))

CIM_PATH = os.path.join(PATH, '../cim/raw/Lede_2024')

class GeoJson:
    def __init__(self, jsonld):
        self.graph = jsonld['@graph']
        coordinates = [item for item in self.graph if 'cim:PositionPoint.Location' in item.keys()]
        self.coordinates = pl.DataFrame(coordinates).unnest('cim:PositionPoint.Location') if len(coordinates) else pl.DataFrame()

    def properties(self, item) -> dict:
        return {'@id': item['@id'] if '@id' in item.keys() else '',
                '@type': re.split(':| .',item['@type'])[-1] if '@type' in item.keys() else '',
                'cim:IdentifiedObject.mRID': item['cim:IdentifiedObject.mRID'] if 'cim:IdentifiedObject.mRID' in item.keys() else '',
                'cim:IdentifiedObject.name': item['cim:IdentifiedObject.name'] if 'cim:IdentifiedObject.name' in item.keys() else ''}

    def geometry(self, item) -> dict:
        if 'cim:PowerSystemResource.Location' in item.keys():
            sequence = self.coordinates.filter(pl.col('@id')==item['cim:PowerSystemResource.Location']['@id']).sort('cim:PositionPoint.sequenceNumber', descending=False)
            points = [(i['cim:PositionPoint.xPosition'],i['cim:PositionPoint.yPosition'],i['cim:PositionPoint.zPosition']) for i in sequence.iter_rows(named=True)]
            return geojson.Point(points[0]) if len(points)==1 else geojson.LineString(points) if len(points) > 1 else geojson.Point()

    def features(self) -> List[geojson.Feature]:
        items = [item for item in self.graph if 'cim:PowerSystemResource.Location' in item.keys()]
        collection =[]
        for item in items:
            properties=self.properties(item)
            geometry=self.geometry(item)
            if len(geometry.coordinates):
                collection.append(geojson.Feature(properties=properties, geometry=geometry))
        return collection


class JsonLd(GeoJson):
    def __init__(self, data):
        super().__init__(data)
        self.data = data

    @property
    def geojson(self):
        return super()

    def types(self) -> List[str]:
        return list(set([i['@type'] for i in self.data['@graph'] if '@type' in i.keys()]))


if __name__ == "__main__":
    filepaths = []
    for (dirpath, dirnames, filenames) in os.walk(CIM_PATH):
        filepaths.extend([os.path.join(dirpath, file) for file in filenames])

    features = []
    for filepath in filepaths:
        with open(filepath, 'r') as fp:

            if os.path.basename(filepath).endswith('.jsonld'):
                data = json.load(fp)
                jsonld_features = JsonLd(data=data).geojson.features()
                if len(jsonld_features):
                    features.extend(jsonld_features)
            elif os.path.basename(filepath).endswith('.geojson'):
                geojson_features = []
                for feature in geojson.load(fp)['features']:
                    if bool(re.search('ACLineSegment', feature['properties']['@type'])):
                        feature['properties']['@type'] = 'ACLineSegment'
                    elif bool(re.search('ConformLoad', feature['properties']['@type'])):
                        feature['properties']['@type'] = 'ConformLoad'
                    elif bool(re.search('PowerTransformer', feature['properties']['@type'])):
                        feature['properties']['@type'] = 'PowerTransformer'
                    elif bool(re.search('Substation', feature['properties']['@type'])):
                        feature['properties']['@type'] = 'Substation'
                    geojson_features.append(feature)
                if len(geojson_features):
                    features.extend(geojson_features)

    if not os.path.exists(os.path.join(PATH, 'bronze')):
        os.makedirs(os.path.join(PATH, 'bronze'), exist_ok=True)

    with open(os.path.join(PATH, 'preamble.geojson'), 'r') as fp:
        preamble = json.load(fp)
    preamble['features'] = features
    with open(os.path.join(PATH, 'bronze', 'lede.geojson'), 'w+') as fp:
        json.dump(preamble, fp)


