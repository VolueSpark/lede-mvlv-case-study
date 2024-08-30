import concurrent.futures, multiprocessing, threading, os, shutil
from datetime import datetime
import polars as pl

PATH = os.path.dirname(__file__)
RAW_PATH = os.path.join(PATH, 'raw')
BRONZE_PATH = os.path.join(PATH, 'bronze')

RAW_MEAS_PATH = os.path.join(RAW_PATH, 'meas')
BRONZE_MEAS_PATH = os.path.join(BRONZE_PATH, 'meas')

from lib import logger


def process_batch(**kwargs) -> str:
    for i, file in enumerate(kwargs['folder_sub_list']):

        df = pl.scan_parquet(os.path.join(RAW_MEAS_PATH, file)).select('value_dt','meter_id','kWhout', 'kWhin', 'kVArhout', 'kVArhin').collect()
        for (meter_id, data) in df.group_by(by='meter_id'):

            meter_path = os.path.join(BRONZE_MEAS_PATH, meter_id[0])
            if not os.path.exists(meter_path):
                os.makedirs(meter_path, exist_ok=True)

            meter_date_path = os.path.join(meter_path, file)
            if not os.path.exists(meter_date_path):
                data.write_parquet(meter_date_path)
            else:
                data.vstack(pl.read_parquet(meter_date_path)).write_parquet(meter_date_path)

        logger.info(
            f"[{datetime.utcnow().isoformat()}] (thread {kwargs['thread_idx']} id:{threading.get_native_id()}) Processed date {file} with "
            f"{df.unique('meter_id').shape[0]} meters for bronze ETL [{i} of {len(kwargs['folder_sub_list'])}].",
            color=f"\u001b[38;5;{16 + kwargs['thread_idx']}m")

    return f"[{kwargs['thread_idx']}:{threading.get_native_id()}] Thread execution finished"


if __name__ == '__main__':

    if os.path.exists(BRONZE_PATH):
        shutil.rmtree(BRONZE_PATH)
    os.makedirs(BRONZE_PATH)

    # use each core of CPU to process batches of neighborhood data
    folder_list = os.listdir(RAW_MEAS_PATH)
    max_workers = multiprocessing.cpu_count()
    batch_size = round(len(folder_list) / max_workers)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_batch, folder_sub_list=batch, thread_idx=i) for i, batch in
                   enumerate([folder_list[x:x + batch_size] for x in range(0, len(folder_list), batch_size)])]
        for future in concurrent.futures.as_completed(futures):
            print(future.result())
