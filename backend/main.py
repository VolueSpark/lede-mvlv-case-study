import random
import time
from datetime import datetime, timedelta
from matplotlib import colors
from lib.lfa import Lfa
import polars as pl
import os, json, requests

PATH = os.path.dirname(os.path.abspath(__file__))

WORK_PATH = os.path.join(PATH, 'lfa')
MV_PATH = os.path.join(PATH, '../data/topology/silver/mv')
LV_PATH = os.path.join(PATH, '../data/topology/silver/lv')
DATA_PATH = os.path.join(PATH, '../data/ami/silver/meas')

from lib import logger


cmap = colors.LinearSegmentedColormap.from_list("custom_cmap", ["#0000FF", "#00FF00", "#FF0000"])
def get_color_from_value(value):
    #value = random.random()
    rgb_color = cmap(value)[:3]  # Get the RGB part (ignore the alpha channel)
    return colors.rgb2hex(rgb_color)


def front_end_update(date: datetime, lfa_result: dict):

    def map_voltage_range(
            v_pu: float,
            v_pu_min: float= 0.9,
            v_pu_max: float= 1.1
    ):
        return get_color_from_value(value=(v_pu - v_pu_min) / (v_pu_max - v_pu_min))

    def map_loading_percent_range(
            loading_percent: float,
            loading_percent_min: float=0.0,
            loading_percent_max: float=100
    ):
       return get_color_from_value(value=(loading_percent-loading_percent_min)/(loading_percent_max-loading_percent_min))

    payload = pl.DataFrame()
    payload = payload.vstack(
        lfa_result['conform_load'].with_columns(pl.col('v_pu').map_elements(lambda v_pu: map_voltage_range(v_pu=v_pu),
                                                                            return_dtype=pl.Utf8)
                                                .alias('color'))
        .rename({'cfl_mrid':'id'})
        .select('id', 'color')
    )
    payload = payload.vstack(
        lfa_result['branch'].with_columns(pl.col('loading_percent').map_elements(lambda loading_percent: map_loading_percent_range(loading_percent=loading_percent),
                                                                                 return_dtype=pl.Utf8).alias('color'))
        .rename({'branch_mrid':'id'})
        .select('id', 'color')
    )
    payload = payload.vstack(
        lfa_result['trafo'].with_columns(pl.col('loading_percent').map_elements(lambda loading_percent: map_loading_percent_range(loading_percent=loading_percent),
                                                                                return_dtype=pl.Utf8).alias('color'))
        .rename({'trafo_mrid':'id'})
        .select('id', 'color')
    )

    # Send the JSON via a POST request
    url = 'http://localhost:5000/api/v1/lede/update'   # Replace with your API endpoint
    headers = {'Content-Type': 'application/json'}
    params = {'timestamp': date.isoformat()}
    data = json.dumps(payload.to_dict(as_series=False))

    with open('data.json', 'w') as f:
        json.dump(payload.to_dict(as_series=False), f)

    response = requests.post(
        url,
        data=data,
        params=params,
        headers=headers
    )

    if response.status_code == 200:
        logger.info(f'[{datetime.utcnow().isoformat()}] Update of Lede API successful for LFA simulation at {date.isoformat()}')
    else:
        logger.exception(f'[{datetime.utcnow().isoformat()}] Update of Lede API failed for LFA simulation at {date.isoformat()}')


if __name__ == "__main__":

    lfa = Lfa(
        work_path=WORK_PATH,
        mv_path=MV_PATH,
        lv_path=LV_PATH,
        data_path=DATA_PATH
    )

    horizon_hours = 24
    from_date = datetime(year=2023, month=8, day=28)
    to_date = from_date + timedelta(hours=horizon_hours)

    for (date, lfa_result) in  lfa.run_lfa(from_date=from_date, to_date=to_date):
        front_end_update(
            date=date,
            lfa_result=lfa_result
        )
        time.sleep(1)

