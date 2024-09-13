from requests.adapters import HTTPAdapter, Retry
from dotenv import load_dotenv
from datetime import datetime, timedelta
import polars as pl
import os, requests

PATH = os.path.dirname(__file__)

time_format = '%Y-%m-%dT%H:%M:%S'

def fetch_hist_valuta(from_time: datetime, to_time: datetime) -> pl.DataFrame:
    with requests.Session() as s:
        retries = Retry(total=5, backoff_factor=2, allowed_methods=frozenset(['GET']))
        s.mount('https://', HTTPAdapter(max_retries=retries))

        try:
            load_dotenv()
            # prepare request
            url = os.getenv('NORGES_BANK_URL')
            headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
            params={'format': 'sdmx-json',
                    'startPeriod': (from_time.date()-timedelta(days=3)).strftime(time_format), # need a time buffer of 3 days for weekends not reporting
                    'endPeriod': (to_time.date()+timedelta(days=3)).strftime(time_format),
                    'locale': 'no'
                    }

            # execute query
            req = requests.Request('GET', url=url, headers=headers, params=params)
            prepped = req.prepare()
            response = s.send(prepped, timeout=1000)
        except Exception as e:
            raise Exception(f"[{datetime.utcnow()}] Failed in the API request with error code {e}. ")

        # get json payload
        data = response.json()

        # parse interesting fields
        values = [float(value[0]) for value in data['data']['dataSets'][0]['series']['0:0:0:0']['observations'].values()]
        dates = data['data']['structure']['dimensions']['observation'][0]['values']

        df=pl.DataFrame(dates).with_columns(nok_euro=pl.lit(pl.Series(values))).with_columns( pl.col('start').str.to_datetime('%Y-%m-%dT%H:%M:%S').alias('timestamp') ).drop('start','end','id','name')
        df = df.sort(by='timestamp').upsample(time_column='timestamp',every='1h').fill_null(strategy="forward").filter(pl.col('timestamp').is_between(from_time, to_time))
        # transform to polars dataframe
        return df
