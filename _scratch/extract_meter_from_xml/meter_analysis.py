import polars as pl
import os

PATH = os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    # read upkip CIMAMI sensor data id's
    path = os.path.join(PATH, '../../data/ami/bronze')
    upkip = pl.DataFrame().with_columns(pl.lit(os.listdir(path)).explode().alias('meter_id'))

    # read memgraph spark mrid
    path  = os.path.join(PATH, 'output/clmrid' )
    clmrid = pl.read_parquet(path)

    # read CGMES data
    path  = os.path.join(PATH, 'output/cgmes' )
    cgmes = pl.read_parquet(path)

    # analysis of data
    print(f"Upkip as AMI data for {upkip.n_unique('meter_id')} unique meter id's.")

    common_cgmes_upkip = list(set(cgmes.unique('meter_id')['meter_id'].to_list()).intersection(set(upkip.unique('meter_id')['meter_id'].to_list())))
    print(f"DIGIN10-24-LV1-P420_EQ has {cgmes.n_unique('meter_id')} unique meter id's where {len(common_cgmes_upkip)} is queried by Upkip.")

    cgmes_mrid = cgmes.unique('mrid')['mrid'].to_list()
    memgraph_mrid = clmrid.unique('meter_mrid')['meter_mrid'].to_list()
    common_memgraph_cgmes = list(set(cgmes_mrid).intersection(set(memgraph_mrid)))

    print(f"Spark memgraph has {len(memgraph_mrid)} meter mrid's. DIGIN10-24-LV1-P420_EQ {len(cgmes_mrid)} meter mrid's. {len(common_memgraph_cgmes)} are common")

    # read CGMES 2023 data
    path  = os.path.join(PATH, 'output/cgmes_lede2023' )
    cgmes_lede2023 = pl.read_parquet(path)

    cgmes_mrid = cgmes_lede2023.unique('mrid')['mrid'].to_list()
    common_memgraph_cgmes = list(set(cgmes_mrid).intersection(set(memgraph_mrid)))
    print(f"Spark memgraph has {len(memgraph_mrid)} meter mrid's. DIGIN10-24-LV1-P420_EQ {len(cgmes_mrid)} meter mrid's. {len(common_memgraph_cgmes)} are common")


