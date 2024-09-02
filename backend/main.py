from lib.lfa import Lfa
import os

PATH = os.path.dirname(os.path.abspath(__file__))

MEDIUM_VOLTAGE_PATH = os.path.join(PATH, '../data/topology/bronze/medium_voltage')
LOW_VOLTAGE_PATH = os.path.join(PATH, '../data/topology/bronze/low_voltage')
WORKSPACE_PATH = os.path.join(PATH, 'lfa')
AMI_DATA_PATH = os.path.join(PATH, '../data/ami/silver/meas')

if __name__ == "__main__":

        lfa = Lfa(
            medium_voltage_path=MEDIUM_VOLTAGE_PATH,
            low_voltage_path=LOW_VOLTAGE_PATH,
            workspace_path=WORKSPACE_PATH,
            ami_data_path=AMI_DATA_PATH
        )

        #lfa.run()

        #lfa.plot(simple=True)
