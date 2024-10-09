import os, json, requests, math
from datetime import datetime
from matplotlib import colors
from lib.lfa.lfa import Lfa
from typing import Tuple
import pandapower as pp
import polars as pl

from lib import logger

PATH = os.path.dirname(os.path.abspath(__file__))

WORK_PATH = os.path.join(PATH, 'lfa')
MV_PATH = os.path.join(PATH, '../data/topology/silver/mv')
LV_PATH = os.path.join(PATH, '../data/topology/silver/lv')
DATA_PATH = os.path.join(PATH, '../data/ami/silver/meas')

FLEX_ASSETS_PATH = os.path.join(PATH, 'flex')

LOW_THRESHOLD_COLOR = "#BFBFBF"
MED_THRESHOLD_COLOR = "#F7889A"
HIGH_THRESHOLD_COLOR = "#6D3C44"

cmap = colors.LinearSegmentedColormap.from_list("custom_cmap", [LOW_THRESHOLD_COLOR, MED_THRESHOLD_COLOR, HIGH_THRESHOLD_COLOR])
def get_color_from_value(value):
    if math.isnan(value):
        return LOW_THRESHOLD_COLOR
    rgb_color = cmap(value)[:3]  # Get the RGB part (ignore the alpha channel)
    return colors.rgb2hex(rgb_color)


def front_end_update(date: datetime, lfa_result: dict):

    def map_voltage_range(
            v_pu: float,
            v_pu_min: float= 0.9,
            v_pu_max: float= 1.1
    ):
        v_pu = max(min(v_pu, v_pu_max),v_pu_min)
        return get_color_from_value(value=abs(1 - v_pu) / ((v_pu_max - v_pu_min)/2))

    def map_loading_percent_range(
            loading_percent: float,
            loading_percent_min: float=30,
            loading_percent_max: float=80
    ):
       loading_percent = max(min(loading_percent, loading_percent_max),loading_percent_min)
       return get_color_from_value(value=(loading_percent-loading_percent_min)/(loading_percent_max-loading_percent_min))

    def map_loss_percent_range(
            loss_percent: float,
            loss_percent_range_min: float=0.0,
            loss_percent_range_max: float=5
    ):
        loss_percent = max(min(loss_percent, loss_percent_range_max),loss_percent_range_min)
        return get_color_from_value(value=(loss_percent-loss_percent_range_min)/(loss_percent_range_max-loss_percent_range_min))

    payload = pl.DataFrame()
    payload = payload.vstack(
        lfa_result['conform_load'].with_columns(
            pl.col('v_pu').map_elements(lambda v_pu: map_voltage_range(v_pu=v_pu),return_dtype=pl.Utf8).alias('color'),
            pl.col('v_pu').map_elements(lambda v_pu: f'{round(v_pu,3)} p.u',return_dtype=pl.Utf8).alias('value'),
        )
        .rename({'cfl_mrid':'id'})
        .select('id', 'value','color')
    )
    payload = payload.vstack(
        lfa_result['branch'].with_columns(
            pl.col('loss_percent').map_elements(lambda loss_percent: map_loss_percent_range(loss_percent=loss_percent),return_dtype=pl.Utf8).alias('color'),
            pl.col('loss_percent').map_elements(lambda loss_percent: f'{round(loss_percent,1)} %',return_dtype=pl.Utf8).alias('value'),
        )
        .rename({'branch_mrid':'id'})
        .select('id', 'value', 'color')
    )
    payload = payload.vstack(
        lfa_result['trafo'].with_columns(
            pl.col('loading_percent').map_elements(lambda loading_percent: map_loading_percent_range(loading_percent=loading_percent),return_dtype=pl.Utf8).alias('color'),
            pl.col('loading_percent').map_elements(lambda loading_percent: f'{round(loading_percent,1)} %',return_dtype=pl.Utf8).alias('value'),
        )
        .rename({'trafo_mrid':'id'})
        .select('id', 'value', 'color')
    )

    # Send the JSON via a POST request
    url = 'http://localhost:5000/api/v1/lede/update'   # Replace with your API endpoint
    headers = {'Content-Type': 'application/json'}
    params = {'timestamp': date.isoformat()}
    data = json.dumps(payload.to_dict(as_series=False))

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


class FlexAssets:

    def __init__(
            self,
            lfa_path: str,
            flex_assets_path: str,
            max_usage_pct_limit: int = 50,
    ):
        self.flex_assets_path = flex_assets_path
        self.max_usage_pct_limit = max_usage_pct_limit
        if not os.path.exists(os.path.join(flex_assets_path, 'flex_assets.json')):
            raise Exception(f"{os.path.join(flex_assets_path, 'flex_assets.json')} does noet exists")

        if not os.path.exists(lfa_path):
            raise Exception(f"{lfa_path} does noet exists")

        with open(os.path.join(flex_assets_path, 'flex_assets.json'), 'r') as fp:
            flex_assets = json.load(fp)
        metadata = pl.read_parquet(os.path.join(lfa_path, 'metadata.parquet'))
        self.net = pp.from_sqlite(os.path.join(lfa_path, 'net.sqlite'))

        if not os.path.exists(os.path.join(flex_assets_path, 'flex_assets.parquet')):

            for i, asset in enumerate(flex_assets):
                flex_asset = metadata.filter(pl.col('meter_id')==asset['meter_id'])
                flex_assets[i]['max_usage_p_kw'] = flex_asset['p_kw_max'].item()*max_usage_pct_limit/100
                flex_assets[i]['max_usage_q_kvar'] = flex_asset['q_kvar_max'].item()*max_usage_pct_limit/100

            df = pl.from_dicts(flex_assets)
            (df.join(df.group_by('uuid').agg(pl.col('meter_id').n_unique()
                                             .alias('#cnt')), on='uuid', validate='m:1')
             .write_parquet(os.path.join(flex_assets_path, 'flex_assets.parquet')))
        self.flex_assets = pl.read_parquet(os.path.join(flex_assets_path, 'flex_assets.parquet'))

        uuid_list = list(set(self.flex_assets['uuid'].to_list())) # all topologies of interest
        uuid_list.append(self.net.name) # add grid tie-in

        self.trafo_meta = (
                {uuid_i: {
                    'name': self.net.trafo3w.iloc[i]['name'],
                    'mrid': self.net.trafo3w.iloc[i]['mrid'].replace('-',''),
                } for i, uuid_i in enumerate(self.net.trafo3w['uuid']) if uuid_i in uuid_list} |
                {uuid_i: {
                    'name': self.net.trafo.iloc[i]['name'],
                    'mrid': self.net.trafo.iloc[i]['mrid'].replace('-',''),
                } for i, uuid_i in enumerate(self.net.trafo['uuid']) if uuid_i in uuid_list}
        )

        self.log_uuid = { uuid:[] for uuid in uuid_list}

    def log(self,date: datetime, lfa_result: Tuple[datetime, dict], flex_active: bool):

        for uuid in self.log_uuid.keys():

            loading_percent = lfa_result['trafo'].filter(pl.col('trafo_mrid') == self.trafo_meta[uuid]['mrid'] )['loading_percent'].item()
            self.log_uuid[uuid].append(
                {
                    'date': date,
                    'trafo_mrid': self.trafo_meta[uuid]['mrid'],
                    'loading_percent': loading_percent,
                    'flex_active':flex_active
                }
            )

    def save(self, name):
        pl.from_dicts(self.log_uuid).write_parquet(os.path.join(self.flex_assets_path, name))

    def plot(self, name):
        if not os.path.exists(os.path.join(self.flex_assets_path, name)):
            logger.info(f'No data to plot for supplied path {os.path.join(self.flex_assets_path, name)}')
            return

        import matplotlib.pyplot as plt

        df = pl.read_parquet(os.path.join(self.flex_assets_path, name))

        data = {}
        for uuid in df.columns:
            df_ = df.select(pl.col(uuid)).unnest(uuid).sort(by='date')

            data[uuid] = {
                't': df_.filter(pl.col('flex_active') == True).sort(by='date', descending=False)['date'].to_list(),
                'y_flex_active': df_.filter(pl.col('flex_active') == True).sort(by='date', descending=False)['loading_percent'].to_list(),
                'y_flex_inactive': df_.filter(pl.col('flex_active') == False).sort(by='date', descending=False)['loading_percent'].to_list(),
                'title': f"{self.trafo_meta[uuid]['name']} ({uuid})"
            }

        fig, axs = plt.subplots(len(df.columns),1, figsize=(15,10), sharex=True)

        fig.text(0.5, 0.0, 'time', ha='center', fontsize=16)
        fig.text(0.0, 0.5, f'Trafo. Util. [%] with {self.max_usage_pct_limit}% max usage flexibility activation', va='center', rotation='vertical', fontsize=14)

        for i, (uuid_i, data_i) in enumerate(data.items()):

            fill_where = [True if ((date.hour > 6) and (date.hour <= 18)) else False for date in  data_i['t']]
            flex_asset_cnt = self.flex_assets.filter(pl.col('uuid')==uuid_i)['#cnt'][0] if len(self.flex_assets.filter(pl.col('uuid')==uuid_i)['#cnt']) else 0

            y_flex_active = [ y_flex_active if fill_where[i] else data_i['y_flex_inactive'][i] for i, y_flex_active in enumerate(data_i['y_flex_active'])   ]

            axs[i].plot(data_i['t'], y_flex_active, label=f'Flex (#{flex_asset_cnt})', color="#DB7889")
            axs[i].plot(data_i['t'], data_i['y_flex_inactive'], label='Normal', color="#000000")

            axs[i].fill_between(data_i['t'], data_i['y_flex_active'], data_i['y_flex_inactive'], alpha=0.2, color="#DB7889", where=fill_where)

            axs[i].set_title(data_i['title'])
            axs[i].legend(loc="upper right")

        fig.suptitle(f'Transformer  Utilization with {self.max_usage_pct_limit}% max usage flexibility activation: 6am-18pm', fontsize=22)
        plt.tight_layout()

        if not os.path.exists(os.path.join(self.flex_assets_path, f"{name.split('.')[0]}.png")):
            fig.savefig(os.path.join(self.flex_assets_path, f"{name.split('.')[0]}.png"), dpi=300)


def run_flexibility(lfa: Lfa, flex: FlexAssets):

    from_date= datetime(2024, 1, 3, 0, 0)
    to_date=datetime(2024, 1, 7, 0, 0)

    name = f'flex_{from_date.isoformat()}-{to_date.isoformat()}.parquet'

    if not os.path.exists(os.path.join(flex.flex_assets_path, name)):

        for (date, lfa_result) in lfa.run_lfa(
                from_date=from_date,
                to_date=to_date,
                step_every=1
        ):
            flex.log(date, lfa_result, flex_active=False)
            logger.info(f'[{date.isoformat()}] Lfa processed for inactive flexible assets')

        for (date, lfa_result) in lfa.run_lfa(
                from_date=from_date,
                to_date=to_date,
                step_every=1,
                flex_assets=flex.flex_assets
        ):

            flex.log(date, lfa_result, flex_active=True)
            logger.info(f'[{date.isoformat()}] Lfa processed for active flexible assets')

        flex.save(f'flex_{from_date.isoformat()}-{to_date.isoformat()}.parquet')
    flex.plot(f'flex_{from_date.isoformat()}-{to_date.isoformat()}.parquet')


#
# run backend LFA engine with date resolution defined by step_every
#
def run_demonstrator(lfa: Lfa, step_every:int=1 ):

    from_date=datetime(2024, 1, 1, 0, 0)
    to_date=datetime(2024, 3, 1, 0, 0)

    for (date, lfa_result) in lfa.run_lfa(
            from_date=from_date,
            to_date=to_date,
            step_every=step_every,
    ):
        front_end_update(
            date=date,
            lfa_result=lfa_result
        )
        logger.info(f'[{date.isoformat()}] Lfa processed for inactive flexible assets')


if __name__ == "__main__":

    lfa = Lfa(
        work_path=WORK_PATH,
        mv_path=MV_PATH,
        lv_path=LV_PATH,
        data_path=DATA_PATH
    )

    flex = FlexAssets(
        lfa_path=WORK_PATH,
        flex_assets_path=FLEX_ASSETS_PATH
    )

    run_flexibility(lfa=lfa, flex=flex)
    #run_demonstrator(lfa=lfa, step_every=8)







