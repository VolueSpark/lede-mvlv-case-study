import glob, os, json
import polars as pl
import pandas as pd

PATH = os.path.dirname(os.path.abspath(__file__))
CIM_FOLDER = 'Lede_2024.09.27'
TOPOLOGY_PATH = os.path.join(PATH, f'../../data/cim/raw/{CIM_FOLDER}')
MYRENE_PATH = os.path.join(PATH, 'Myrene_NIS_25082024.xls')

if __name__ == "__main__":
    #
    # Parse meter id data from recent CIM
    #
    id_map = pl.DataFrame()
    for index, file_path in enumerate(glob.glob(os.path.join(TOPOLOGY_PATH,'LowVoltage', '*_CU.jsonld'))):
        print(f'[{index+1}] Loading {file_path} for processing')
        with open(file_path, 'r') as fp:
            usagepoint_mrid = []
            for item in json.load(fp)['@graph']:
                if item['@type'] == 'cim:UsagePoint':
                    usagepoint_mrid.append({'usagepoint_mrid':item['cim:IdentifiedObject.mRID']})
        if len(usagepoint_mrid):
            usagepoint_mrid = pl.from_dicts(usagepoint_mrid)

            meter_map = []
            with open(os.path.join(TOPOLOGY_PATH,'LowVoltage', os.path.basename(file_path).replace('_CU','_OR')), 'r') as fp:
                for item in json.load(fp)['@graph']:
                    if 'nc:Name.mRID' in item.keys() and 'cim:Name.name' in item.keys():
                        if item['cim:Name.name'] == '707057500042745649':
                            print('')
                        meter_map.append({'usagepoint_mrid':item['cim:Name.IdentifiedObject']['@id'].split(':')[-1],'mrid':item['nc:Name.mRID'], 'name':item['cim:Name.name']})
            meter_map = pl.from_dicts(meter_map)

            try:
                id_map = id_map.vstack(meter_map.join(usagepoint_mrid, on='usagepoint_mrid', how='inner'))
            except Exception as e:
                print(e)

    for index, file_path in enumerate(glob.glob(os.path.join(TOPOLOGY_PATH,'MediumVoltage', '*_CU.jsonld'))):
        print(f'[{index+1}] Loading {file_path} for processing')
        with open(file_path, 'r') as fp:
            usagepoint_mrid = []
            for item in json.load(fp)['@graph']:
                if item['@type'] == 'cim:UsagePoint':
                    usagepoint_mrid.append({'usagepoint_mrid':item['cim:IdentifiedObject.mRID']})
        if len(usagepoint_mrid):
            usagepoint_mrid = pl.from_dicts(usagepoint_mrid)

            meter_map = []
            with open(os.path.join(TOPOLOGY_PATH,'LowVoltage', os.path.basename(file_path).replace('_CU','_OR')), 'r') as fp:
                for item in json.load(fp)['@graph']:
                    if 'nc:Name.mRID' in item.keys() and 'cim:Name.name' in item.keys():
                        meter_map.append({'usagepoint_mrid':item['cim:Name.IdentifiedObject']['@id'].split(':')[-1],'mrid':item['nc:Name.mRID'], 'name':item['cim:Name.name']})
            meter_map = pl.from_dicts(meter_map)

            try:
                id_map = id_map.vstack(meter_map.join(usagepoint_mrid, on='usagepoint_mrid', how='inner'))
            except Exception as e:
                print(e)

    print(f'{id_map.shape[0]} unique meterpoints have been parsed from {TOPOLOGY_PATH}')
    id_map = id_map.with_columns(pl.col('name').cast(pl.Int64))
    id_map.write_parquet(os.path.join(PATH, f'meter_mrid_name_{CIM_FOLDER.lower()}.parquet'))

    coop_meter_point_id = 707057500042745649
    if coop_meter_point_id not in id_map['name'].to_list():
        print(f"{coop_meter_point_id} not in availible AMI sensor list.")
    else:
        print(f"COOP mrid {coop_meter_point_id} is availible AMI sensor list.")

    #
    # Load meter data from Grid
    #
    usagepoint = pl.read_csv(os.path.join(PATH, 'usagepoints.csv')).rename({'n.Mrid':'mrid', 'n.MeterPointId':'name'})

    print(f"The following meters are not found in both strategies: {list(set(usagepoint['name'])^(set(id_map['name'])))}")

    #
    # Load meter name data extract from Lede
    #
    myrene = (pl.from_pandas(pd.read_excel(open(MYRENE_PATH, 'rb'), sheet_name='Anleggspunkt'))
              .select('Status','EAN', 'Målernummer')
              .rename({'EAN':'name'}))
