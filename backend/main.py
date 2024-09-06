from datetime import datetime, timedelta
from lib.lfa import Lfa
import geojson
import os, random

PATH = os.path.dirname(os.path.abspath(__file__))

WORK_PATH = os.path.join(PATH, 'lfa')
MV_PATH = os.path.join(PATH, '../data/topology/silver/mv')
LV_PATH = os.path.join(PATH, '../data/topology/silver/lv')
DATA_PATH = os.path.join(PATH, '../data/ami/silver/meas')



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

    lfa.run_lfa(from_date=from_date, to_date=to_date)

        #lfa.plot(simple=True)
