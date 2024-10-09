import re, json, shutil, os


from lib import logger
from lib.lfa.lfa import LfaValidation
from lib.lfa.topology import Topology

PATH = os.path.dirname(__file__)

BRONZE_PATH = os.path.join(PATH, 'bronze')
SILVER_PATH = os.path.join(PATH, 'silver')

BRONZE_MV_PATH = os.path.join(BRONZE_PATH, 'mv')
BRONZE_LV_PATH = os.path.join(BRONZE_PATH, 'lv')

SILVER_MV_PATH = os.path.join(SILVER_PATH, 'mv')
SILVER_LV_PATH = os.path.join(SILVER_PATH, 'lv')



#
# Parse spark.grid event files as queried from memgraph and convert base units and aliases for SI units and spark.forecast aliases
#
if __name__ == "__main__":
    shutil.rmtree(SILVER_PATH, ignore_errors=True)

    os.makedirs(SILVER_MV_PATH, exist_ok=True)
    os.makedirs(SILVER_LV_PATH, exist_ok=True)

    mv_file_list = os.listdir(BRONZE_MV_PATH)

    exceptions_log =[]

    mv_topology_dict={}
    for index, mv_file in enumerate(mv_file_list):
        with open(os.path.join(BRONZE_MV_PATH, mv_file), 'r') as fp:
            try:
                mv_topology = Topology(**json.load(fp))
                for mv_trafo in mv_topology.trafo:
                    if mv_trafo.hv_bus.mrid == mv_topology.slack[0].mrid:
                        mv_trafo.in_service = True
                mv_topology_dict[mv_topology.uuid] = mv_topology
                #{trafo.mrid:trafo.in_service for trafo in mv_topology.trafo if trafo.hv_bus.mrid == mv_topology.slack[0].mrid}
            except Exception as e:
                logger.exception(f'[{mv_file}({index} of {len(mv_file_list)})] exception raised. [{e}]')
                exceptions_log.append(f'[{mv_file}({index} of {len(mv_file_list)})] exception raised. [{e}]')

    lv_file_list = os.listdir(BRONZE_LV_PATH)
    successful_validated = 0
    failed_validated = 0
    for index, lv_file in enumerate(lv_file_list):

        with open(os.path.join(BRONZE_LV_PATH, lv_file), 'r') as fp:
            try:
                lv_topology = Topology(**json.load(fp))

                if not len(lv_topology.load):
                    raise Exception(f'{lv_topology.uuid} has no loads  and will be discarded')

                for uuid in mv_topology_dict.keys():
                    for mv_trafo in mv_topology_dict[uuid].trafo:
                        if lv_topology.trafo[0].mrid == mv_trafo.mrid:
                            logger.info(f'[{lv_file} ({index+1} of {len(lv_file_list)})] lv_trafo.mrid={lv_topology.trafo[0].mrid} associated with mv.uuid={uuid}')
                            lv_topology.trafo[0].in_service = True
                            mv_trafo.in_service = True

                if not lv_topology.trafo[0].in_service:
                    raise Exception(f'topology.uuid={lv_topology.uuid} trafo.mrid={lv_topology.trafo[0].mrid} is not in service')

                LfaValidation(topology=lv_topology).validate

                with open(os.path.join(SILVER_LV_PATH, lv_topology.uuid), 'w') as fp:
                    successful_validated+=1
                    json.dump(lv_topology.dict(), fp)
                    logger.info(f'[{lv_file} ({index+1} of {len(lv_file_list)})] {successful_validated} successfully validated.' )

            except Exception as e:
                failed_validated+=1
                logger.exception(f'[{lv_file} ({index+1} of {len(lv_file_list)})] {failed_validated} failed validation.' )
                exceptions_log.append(f'{len(exceptions_log)+1}. Exception raised for {lv_file} ({index} of {len(lv_file_list)} of {lv_topology.__class__.__name__}) [{e}]')

    for uuid, mv_topology in mv_topology_dict.items():
        with open(os.path.join(SILVER_MV_PATH, mv_topology.uuid), 'w') as fp:
            json.dump(mv_topology.dict(), fp)

    with open(os.path.join(SILVER_PATH, 'exceptions.json'), 'w') as fp:
        json.dump(exceptions_log, fp)


