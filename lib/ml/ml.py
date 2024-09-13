import polars as pl
import json

from pydantic import BaseModel, Field
import os, yaml

from lib.price.insight import fetch_hist_spot
from lib.price.valuta import fetch_hist_valuta
from lib.weather.weather import fetch_hist_weather

PATH = os.path.dirname(__file__)

with open(os.path.join(PATH,'config.yaml')) as fp:
    config = yaml.safe_load(fp)


class Config(BaseModel):
    topology_uuid: str = Field(description='Low voltage topology uuid')
    work_path: str = Field(description='Root folder to was as workspace')
    meter_data_path: str = Field(description='Path to the conform load meter data')
    topology_data_path: str = Field(description='Path to the topology data')
    geojson_path: str = Field(description='Path to the geojson file')


class Ml:
    def __init__(self, config: Config):

        self.ml_path = os.path.join(config.work_path, config.topology_uuid)
        self.ml_data = os.path.join(self.ml_path, 'data')

        os.makedirs(self.ml_path, exist_ok=True)
        os.makedirs(self.ml_data, exist_ok=True)

        if not os.path.exists(os.path.join(self.ml_data, 'meter.parquet')):
            with open( os.path.join( config.topology_data_path, f'{config.topology_uuid}'), 'r') as fp:
                data = json.load(fp)
            df = pl.DataFrame()
            for meter in data['load']:
                df = df.vstack(pl.read_parquet(os.path.join(config.meter_data_path, meter['meter_id'])))
            df.write_parquet(os.path.join(self.ml_data, 'meter.parquet'))

        df = pl.read_parquet(os.path.join(self.ml_data, 'meter.parquet'))

        from_time =df['datetime'].min()
        to_time =df['datetime'].max()

        with open( os.path.join( config.geojson_path, 'lede.geojson'), 'r') as fp:
            data = json.load(fp)
            for feature in data['features']:
                if feature['properties']['objecttype'] == 'ConformLoad':
                    longitude, latitude = feature['geometry']['coordinates']
                    break

        if not os.path.exists(os.path.join(self.ml_data, 'weather.parquet')):
            (fetch_hist_weather(
                latitude=latitude,
                longitude=longitude,
                date_from=from_time,
                date_to=to_time)
             .write_parquet(os.path.join(self.ml_data, 'weather.parquet')))

        if not os.path.exists(os.path.join(self.ml_data, 'price.parquet')):
            spot = fetch_hist_spot(
                from_time=from_time,
                to_time=to_time,
                latitude=latitude,
                longitude=longitude
            )
            valuta = fetch_hist_valuta(
                from_time=from_time,
                to_time=to_time
            )
            (spot
             .join(valuta, on='timestamp', validate='1:1')
             .with_columns((pl.col('euro_mwh')/1000.0*pl.col('nok_euro'))
                           .alias(f'nok_kwh'))
             .write_parquet(os.path.join(self.ml_data, 'price.parquet')))






