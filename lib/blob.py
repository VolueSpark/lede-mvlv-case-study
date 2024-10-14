from azure.storage.blob import BlobServiceClient, BlobPrefix
from typing import List, Tuple
from dotenv import load_dotenv
from datetime import datetime
import polars as pl
import pandas as pd
import os, io

from lib import logger

load_dotenv()

TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
LEDE_PREDICTION_PREFIX_PATH = 'delta/lede/prediction_prod/data'
LEDE_MEASUREMENT_PREFIX_PATH = 'data/lede/'

def azure_client(func):
    def inner(*args, **kwargs):
        connection_string = f"DefaultEndpointsProtocol=https;AccountName={os.getenv('AZURE_ACCOUNT')};AccountKey={os.getenv('AZURE_KEY')};EndpointSuffix=core.windows.net"
        with BlobServiceClient.from_connection_string(conn_str=connection_string) as client:
            return func(*args, **(kwargs|{'client':client}))
    return inner


@azure_client
def list_containers(**kwargs) -> List[str]:
    with kwargs['client'] as client:
        return [container.name for container in client.list_containers()]

@azure_client
def list_blobs(**kwargs) -> List[str]:
    with kwargs['client'].get_container_client(os.getenv('AZURE_CONTAINER')) as client:
        return [blob.name for blob in client.list_blobs()]


@azure_client
def list_blobs_hierarchical(**kwargs)->List[str]:
    def build_blobs_hierarchy(client, prefix, collected_blobs):
        for blob in client.walk_blobs(name_starts_with=prefix, delimiter='/'):
            if isinstance(blob, BlobPrefix):
                build_blobs_hierarchy(client, blob.name, collected_blobs)
            else:
                collected_blobs.append(blob.name)

    with kwargs['client'].get_container_client(os.getenv('AZURE_CONTAINER')) as client:
        return build_blobs_hierarchy(client=client, prefix=kwargs['prefix'], collected_blobs=kwargs['collected_blobs'])


@azure_client
def upload_blob_file(blob_path: str, **kwargs):
    blob_name = blob_path.split('/')[-1]
    container_client = kwargs['client'].get_container_client(container=os.getenv('AZURE_CONTAINER'))
    with open(file=blob_path, mode="rb") as fp:
        container_client.upload_blob(name=blob_name, data=fp.read(), overwrite=True)

@azure_client
def download_blob_file(blob_name: str, **kwargs):
    container_client = kwargs['client'].get_container_client(container=os.getenv('AZURE_CONTAINER'))
    return container_client.download_blob(blob_name).readall()


def fetch_lede_prediction_ts(from_date: datetime, to_date: datetime=None) -> pl.DataFrame:
    if to_date is None:
        to_date =from_date

    prefixes = [ f"{LEDE_PREDICTION_PREFIX_PATH}/date={date.strftime('%Y-%m-%d')}" for date in pd.date_range(from_date, to_date, freq='d')]

    df = pl.DataFrame()
    for prefix in prefixes:

        collected_blobs = []
        list_blobs_hierarchical(prefix=prefix, collected_blobs=collected_blobs)

        logger.info(f"Fetched {len(collected_blobs)} blobs for {prefix}")
        for blob_name in collected_blobs:
            df = df.vstack(
                (
                    pl.read_parquet(io.BytesIO(download_blob_file(blob_name=blob_name)))
                    .select('meter_id', 'value_dt', 'kWhout', 'kWhin', 'kVArhin', 'kVArhout')
                )
            )

    return df


def fetch_lede_measurement_ts(from_date: datetime, to_date: datetime=None) -> Tuple[str, pl.DataFrame]:
    if to_date is None:
        to_date = from_date

    date_range = [ date.strftime('%Y%m%d') for date in pd.date_range(from_date, to_date, freq='d')]

    for date in date_range:

        collected_blobs = []
        prefix = os.path.join(LEDE_MEASUREMENT_PREFIX_PATH, date)
        list_blobs_hierarchical(prefix=prefix, collected_blobs=collected_blobs)

        data = pl.DataFrame()
        for blob_name in collected_blobs:
            data = data.vstack(
                (
                    pl.read_parquet(io.BytesIO(download_blob_file(blob_name=blob_name)))
                    .select('meter_id', 'value_dt', 'kWhout', 'kWhin', 'kVArhin', 'kVArhout')
                )
            )

        if not data.is_empty():
            logger.info(f"Fetched {len(collected_blobs)} blobs for {prefix} which yielded {data.n_unique('meter_id')} unique meter readings")
            yield date, data
        else:
            logger.info(f"Fetched {len(collected_blobs)} blobs for {prefix}")


