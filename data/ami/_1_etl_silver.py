import concurrent.futures, multiprocessing, threading, os, shutil
from datetime import datetime
import polars as pl
import pandas as pd

PATH = os.path.dirname(__file__)

BRONZE_PATH = os.path.join(PATH, 'bronze')
SILVER_PATH = os.path.join(PATH, 'silver')

BRONZE_MEAS_PATH = os.path.join(BRONZE_PATH, 'meas')
SILVER_MEAS_PATH = os.path.join(SILVER_PATH, 'meas')

from lib import logger

def process_batch(**kwargs) -> str:
    for i, meter_id in enumerate(kwargs['folder_sub_list']):

        df = (
            pl.scan_parquet(os.path.join(BRONZE_MEAS_PATH, meter_id))
            .with_columns(pl.col('value_dt').dt.replace_time_zone(None))
            .drop_nulls()
            .unique(subset=['value_dt','meter_id'])
            .sort(by='value_dt', descending=False)
        ).collect()

        # out-> grid to energy consumer (grid consumption / grid export)
        #  in-> energy consumer to grid (IPP production / grid import)
        df = df.filter(
            pl.col('kWhout') >= 0,
            pl.col('kWhin') >= 0,
            pl.col('kVArhout') >= 0,
            pl.col('kVArhin') >= 0
        )

        df_orig = df.to_pandas()

        # upsample dataseries
        df_ = df.select('value_dt','meter_id')
        for column in ['kWhout', 'kWhin', 'kVArhout', 'kVArhin']:
            series = pd.Series(data=df_orig[column].values, index=df_orig['value_dt'].values)
            interp = series.resample(rule='1h').interpolate(method='linear')
            df_interp = pd.DataFrame(data=interp, columns=[column])
            df_interp.reset_index(inplace=True)
            df_interp = df_interp.rename(columns={'index': 'value_dt'})
            df_ = df_.join(pl.from_pandas(df_interp), on='value_dt', validate='1:1')

            if False:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()

                a = df_.sort(by='value_dt', descending=False)[0:100]
                b = pl.from_pandas(df_orig).sort(by='value_dt', descending=False)[0:100]

                plt.plot(a['value_dt'], b[column], label='interpolated', color='blue')
                plt.plot(b['value_dt'], a[column], label='original', linestyle='--', color='red')

                plt.legend(loc='best')
                plt.title(column)
                plt.show()

        if df.shape[0]:
            assert df.select('kWhout', 'kWhin', 'kVArhout', 'kVArhin').min().min_horizontal().item() >= 0

            df = (df_.rename({'value_dt':'datetime', 'kWhout':'p_kwh_out', 'kWhin':'p_kwh_in', 'kVArhout':'q_kvarh_out', 'kVArhin':'q_kvarh_in'})
                  .select('datetime', 'meter_id', 'p_kwh_in', 'p_kwh_out', 'q_kvarh_in', 'q_kvarh_out'))

            df.write_parquet(os.path.join(SILVER_MEAS_PATH, meter_id))

            logger.info(f"[{datetime.utcnow().isoformat()}] (thread {kwargs['thread_idx']} id:{threading.get_native_id()}) Processed meter {meter_id} for silver ETL [{i} of {len(kwargs['folder_sub_list'])}].",
                      color=f"\u001b[38;5;{16 + kwargs['thread_idx']}m")
        else:
            logger.warning(f"[{datetime.utcnow().isoformat()}] (thread {kwargs['thread_idx']} id:{threading.get_native_id()}) Processed meter {meter_id} for silver ETL [{i} of {len(kwargs['folder_sub_list'])} with no data].")

    return f"[{kwargs['thread_idx']}:{threading.get_native_id()}] Thread execution finished"


if __name__ == '__main__':

    if os.path.exists(SILVER_PATH):
        shutil.rmtree(SILVER_PATH)
    os.makedirs(SILVER_MEAS_PATH)

    # use each core of CPU to process batches of neighborhood data
    folder_list = os.listdir(BRONZE_MEAS_PATH)
    max_workers = multiprocessing.cpu_count()
    batch_size = round(len(folder_list) / max_workers)

    #process_batch(folder_sub_list=['707057500041450476'], thread_idx=0)
    #exit(1)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_batch, folder_sub_list=batch, thread_idx=i) for i, batch in
                   enumerate([folder_list[x:x + batch_size] for x in range(0, len(folder_list), batch_size)])]
        for future in concurrent.futures.as_completed(futures):
            print(future.result())
