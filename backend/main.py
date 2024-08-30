from lib.topology import Topology
from lib.lfa import Lfa
import os, json

PATH = os.path.dirname(os.path.abspath(__file__))
TOPOLOGY_PATH = os.path.join(PATH, '../data/topology/bronze/topology.json')
AMI_DATA_PATH = os.path.join(PATH, '../data/ami/silver')

if __name__ == "__main__":

        lfa = Lfa(topology_path=TOPOLOGY_PATH,
                  ami_data_path=AMI_DATA_PATH,
                  workspace_path=os.path.join(PATH, 'lfa'))

        lfa.run()

        lfa.plot(simple=True)
