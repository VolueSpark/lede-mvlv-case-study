import polars as pl
import json, os

PATH = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(PATH, 'data/cl-to-up.json')) as fp:

    rows = []
    while True:

        # Get next line from file
        line = fp.readline()

        # if line is empty
        # end of file is reached
        if not line:
            break
        data = json.loads(line)

        for up in data['UPs']:
            rows.append({'cl_mrid':data['CL']['Mrid'], 'cl_name':data['CL']['Name'], 'meter_mrid':up['Mrid']})

    df = pl.from_dicts(rows)
    df.write_parquet(os.path.join(PATH, 'output/clmrid'))

    print(f"We have {pl.from_dicts(rows).n_unique('cl_mrid')} unique conform load mrid's")
    print(f"We have {pl.from_dicts(rows).n_unique('meter_mrid')} unique meter mrid's")

