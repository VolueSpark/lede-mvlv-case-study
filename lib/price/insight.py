from shapely.geometry import shape, Point
from datetime import datetime, timedelta
from dotenv import load_dotenv
import wapi, os, json

import polars as pl

PATH = os.path.dirname(__file__)

with open(os.path.join(PATH, f'priceArea.json')) as fp:
    geojson_area_polygon = json.load(fp)


def resolve_area(latitude: float, longitude: float) -> (float, float, str, str):
    area = 'empty'
    point = Point(longitude, latitude)
    for feature in geojson_area_polygon['features']:
        polygon = shape(feature['geometry'])
        if polygon.contains(point):
            area = feature['properties']['ElSpotOmr']
    if area == 'empty':
        return (latitude, longitude, area, False)
    return (latitude, longitude, int(area.split(' ')[-1]), True)


def insight_client(func):
    def inner(*args, **kwargs):
        load_dotenv()
        session = wapi.Session(client_id=os.getenv('INSIGHT_CLIENT_ID'), client_secret=os.getenv('INSIGHT_CLIENT_SECRET'))
        return func(*args, **(kwargs|{'session':session, 'func':func.__name__}))
    return inner


@insight_client
def fetch_hist_spot(
        from_time: datetime,
        to_time: datetime,
        latitude: float,
        longitude: float,
        **kwargs
) ->pl.DataFrame:

    latitude, longitude, area, success = resolve_area(latitude, longitude)

    assert success
    assert area in range(6), 'Valid area codes for spot required to be in the range 1-6 for Norway'

    data = {}

    area_string = f"NO{area}"
    curve_name = f"pri no{area} spot €/mwh cet h a"
    curve: wapi.curves.TimeSeriesCurve = kwargs['session'].get_curve(name=curve_name)
    ts = curve.get_data(time_zone='UTC', data_from=from_time, data_to=to_time+timedelta(hours=1))

    for timestamp, price in ts.points:
        if timestamp not in data:
            data[timestamp] = {"timestamp": datetime.utcfromtimestamp(timestamp // 1000)}
        data[timestamp]['euro_mwh'] = round(price, 2)

    return pl.from_dicts(list(data.values()))



