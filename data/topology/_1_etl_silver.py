from lib.topology import Topology
import os, json

PATH = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    with open(os.path.join(os.path.join(PATH, 'bronze/topology.json')), 'r') as fp:
        data = json.load(fp)
        topology = Topology(**data)

    if not os.path.exists(os.path.join(PATH, 'silver')):
        os.makedirs(os.path.join(PATH, 'silver'))

    with open(os.path.join(PATH, 'silver/topology.json'), 'w+') as fp:
        json.dump(topology.dict(), fp)

