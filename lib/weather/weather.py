from datetime import datetime, timedelta, time,  date
from requests.adapters import HTTPAdapter, Retry
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import List
import polars as pl
import os, requests
from lib import decorator_timer

PATH = os.path.dirname(__file__)


class Hours(BaseModel):
    time_: time = Field(alias='datetime', default=None)
    temp: float = Field(alias='temp', default=None)
    feelslike: float = Field(alias='feelslike', default=None)
    humidity: float = Field(alias='humidity', default=None)
    dew: float = Field(alias='dew', default=None)
    precipprob: float = Field(alias='precipprob', default=None)
    pressure: float = Field(alias='pressure', default=None)
    cloudcover: float = Field(alias='cloudcover', default=None)
    solarradiation: float = Field(alias='solarradiation', default=None)
    solarenergy: float = Field(alias='solarenergy', default=None)


class Weather(BaseModel):
    date_: date = Field(alias='datetime')
    tempmax: float = Field(alias='tempmax')
    tempmin: float = Field(alias='tempmin')
    tempmean: float = Field(alias='temp')
    feelslikemin: float = Field(alias='feelslikemin')
    feelslikemean: float = Field(alias='feelslike')
    dewmean: float = Field(alias='dew')
    humiditymean: float = Field(alias='humidity')
    precipprobmean: float = Field(alias='precipprob')
    pressuremean: float = Field(alias='pressure')
    cloudcovermean: float = Field(alias='cloudcover')
    solarradiationmean: float = Field(alias='solarradiation')
    solarenergymean: float = Field(alias='solarenergy')
    windspeedmax: float = Field(alias='windspeedmax')
    windspeedmin: float = Field(alias='windspeedmin')
    sunrise: time = Field(alias='sunrise')
    sunset: time = Field(alias='sunset')
    hours: List[Hours] = Field(alias='hours')

@decorator_timer
def fetch_hist_weather(
        latitude: float,
        longitude: float,
        date_from: datetime,
        date_to: datetime) -> pl.DataFrame:
    with requests.Session() as s:
        retries = Retry(total=5, backoff_factor=2, allowed_methods=frozenset(['GET']))
        s.mount('https://', HTTPAdapter(max_retries=retries))

        try:
            load_dotenv()
            # prepare request
            url = os.getenv('VISUAL_CROSSING_URL')+f"{latitude}%2C{longitude}/{date_from.date()-timedelta(days=1)}/{date_to.date()+timedelta(days=1)}"
            headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
            params={'unitGroup':'metric',
                    'elements':'datetime,tempmax,tempmin,temp,feelslikemin,feelslike,dew,humidity,precipprob,windspeedmax,windspeedmean,windspeedmin,pressure,cloudcover,solarradiation,solarenergy,sunrise,sunset',
                    'include':'hours',
                    'key':os.getenv('VISUAL_CROSSING_API_KEY')
                    }

            # execute query
            req = requests.Request('GET', url=url, headers=headers, params=params)
            prepped = req.prepare()
            response = s.send(prepped, timeout=1000)
        except Exception as e:
            raise Exception(f"[{datetime.utcnow()}] Failed in the API request with error code {e}. ")

        # get json payload
        if response.status_code == 200:
            weather = response.json()

            df = pl.DataFrame()
            for day in weather['days']:
                try:
                    df = df.vstack(pl.from_dicts(Weather(**day).dict()).unnest('hours'))
                except Exception as e:
                    print(e)

            df = df.with_columns(
                pl.col('date_').dt.combine(pl.col('time_')).alias('timestamp'),
                pl.col('date_').dt.combine(pl.col('sunrise')).alias('sunrise'),
                pl.col('date_').dt.combine(pl.col('sunset')).alias('sunset')
            ).drop('date_','time_')

            columns = df.columns
            columns.sort()

            df = (df.select(columns)
                  .sort(by='timestamp')
                  .upsample(time_column='timestamp',every='1h')
                  .fill_null(strategy="forward")
                  .filter(pl.col('timestamp')
                          .is_between(date_from, date_to))
                  .unique('timestamp'))

            return df