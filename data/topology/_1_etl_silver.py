import re, json, shutil, os


from lib import logger
from lib.lfa import LfaValidation
from lib.schemas.topology import Topology

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
    assert len(mv_file_list) == 1, f'Medium voltage are exceeding the expected quantity of one'
    mv_file = os.path.join(BRONZE_MV_PATH, mv_file_list[0])

    exceptions_log =[]
    with open(mv_file, 'r') as fp:
        try:
            mv_topology = Topology(**json.load(fp))
            slack_bus = mv_topology.slack[0].bus
            mv_topology_trafo = [trafo for trafo in mv_topology.trafo if trafo.hv_bus in slack_bus]
            mv_topology_trafo[0].in_service = True
        except Exception as e:
            logger.exception(f'[{mv_file}({1} of {len(mv_file_list)})] exception raised. [{e}]')
            exceptions_log.append(f'[{mv_file}({1} of {len(mv_file_list)})] exception raised. [{e}]')
    mv_trafo_list = [trafo.mrid for trafo in mv_topology.trafo]

    lv_file_list = os.listdir(BRONZE_LV_PATH)
    lv_trafo_list =[]
    for index, lv_file in enumerate(lv_file_list):
        with open(os.path.join(BRONZE_LV_PATH, lv_file), 'r') as fp:
            try:
                lv_topology = Topology(**json.load(fp))

                if not bool(len(lv_topology.load)):
                    raise Exception(f'[{index+1}] topology {lv_topology.uuid} has no loads and will be discarded for lfa')
                if lv_topology.trafo[0].mrid not in mv_trafo_list:
                    raise Exception(f'[{index+1}] topology {lv_topology.uuid} trafo mrid is not found in the mv layer')

                LfaValidation(topology=lv_topology)

                lv_trafo = lv_topology.trafo[0]
                mv_trafo = [mv_trafo for mv_trafo in mv_topology.trafo if mv_trafo.mrid == lv_trafo.mrid ][0]

                lv_trafo.in_service = mv_trafo.in_service = True
                lv_trafo_list.append({'nbhd_id':lv_topology.uuid, 'trafo_mrid':lv_trafo.mrid})

                with open(os.path.join(SILVER_LV_PATH, lv_topology.uuid), 'w') as fp:
                    json.dump(lv_topology.dict(), fp)
                    logger.info(f'[{lv_file}({index} of {len(lv_file_list)})] successfully validated.' )

            except Exception as e:
                msg = f'{len(exceptions_log)+1}. Exception raised for {lv_file} ({index} of {len(lv_file_list)} of {lv_topology.__class__.__name__}) [{e}]'
                logger.exception(msg)
                exceptions_log.append(msg)

    with open(os.path.join(SILVER_MV_PATH, mv_topology.uuid), 'w') as fp:
        json.dump(mv_topology.dict(), fp)

    with open(os.path.join(SILVER_PATH, 'exceptions.json'), 'w') as fp:
        json.dump(exceptions_log, fp)

    logger.info(f'{len(lv_trafo_list)} of {len(lv_file_list)} LV topologies could be validated via LFA')

