import os, json

TOPOLOGY_PATH = '/home/phillip/repo/volue.spark/lede-mvlv-case-study/data/topology/silver/lv'


topology_list = os.listdir(TOPOLOGY_PATH)

coop_meter_id = '707057500042745649'

for topology in topology_list:
    with open(os.path.join(TOPOLOGY_PATH, topology), 'r') as f:
        data = json.load(f)
        if len([load for load in data['load'] if load['meter_id'] == coop_meter_id]):
            print(f'Meter {coop_meter_id} belongs to topology: {topology}')
            break