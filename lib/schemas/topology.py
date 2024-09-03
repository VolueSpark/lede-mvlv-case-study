from pydantic import BaseModel, Field, AliasChoices, model_validator
from typing_extensions import Self
from typing import List, Optional, Tuple, Any

from lib.schemas.schema import (
    PowerTransformer,
    ConnectivityNode,
    AcLineSegment,
    Switch,
    UsagePoint,
    GhostNodes
)

from lib import logger

class Topology(BaseModel):
    uuid: str = Field(
        validation_alias=AliasChoices('neighbourhoodId', 'regionId', 'uuid')
    )

    slack: List[ConnectivityNode] = Field(alias='slackBus')
    trafo: List[PowerTransformer] = Field(alias='powerTransformers')
    bus: List[ConnectivityNode] = Field(alias='connectivityNodes')
    branch: List[AcLineSegment] = Field(alias='acLineSegments')
    switch: List[Switch] = Field(alias='switches')
    conform_load: List[ConnectivityNode] = Field(alias='conformLoads')
    load: Optional[List[UsagePoint]] = Field(alias='usagePoints')
    ghost: Optional[List[GhostNodes]] = Field(alias='ghostNodes')

    def recover_voltage(self, mrid: str, forward_search: dict=None) -> dict:
        if forward_search == None:
            forward_search = {}

        # check if been visited before
        if mrid in forward_search.keys():
            # terminate if bus was already visited
            logger.info(f'Forward: {mrid} ({forward_search[mrid]}) -> Voltage unknown', color=logger.RED)
            print(f'Forward: {mrid} ({forward_search[mrid]}) -> Voltage unknown')
            return {mrid:forward_search[mrid]}
        else:
            # locate bus of interest
            bus = [bus for bus in self.bus if bus.bus == mrid][0]

            # bus needs termination if deep search reveals nothing
            current_bus = {bus.bus:bus.rated_kv}
            forward_search |= current_bus

            # terminate if visited bus does have a valid voltage
            if bus.rated_kv:
                logger.info(f'Forward: {bus.bus} ({bus.rated_kv}) -> Voltage Recovered', color=logger.GREEN)
                return current_bus

            # query all branch and switch segments to get next bus(s) destination
            next_bus_via_switch = [switch.to_bus if switch.from_bus == mrid else switch.from_bus for switch in self.switch if switch.to_bus == mrid or switch.from_bus == mrid]
            next_bus_via_branch = [branch.to_bus if branch.from_bus == mrid else branch.from_bus for branch in self.branch if branch.to_bus == mrid or branch.from_bus == mrid]

            # attempt to resolve voltages from next bus
            forward_result = {}
            for next_mrid in list(set(next_bus_via_switch).union(set(next_bus_via_branch))):
                if next_mrid not in forward_search.keys():
                    logger.info(f'Forward search: {bus.bus} ({bus.rated_kv} kV) -> {next_mrid} (? kV)', color=logger.BLACK)

                    result = self.recover_voltage(mrid=next_mrid, forward_search=forward_search)

                    forward_result = {mrid: result[list(result)[0]]} | result

                    logger.info(f'Backward result: {list(forward_result)[0]} ({forward_result[list(forward_result)[0]]} kV) <- {list(forward_result)[1]} ({forward_result[list(forward_result)[1]]} kV)', color=logger.BLACK)
            return forward_result




    @model_validator(mode='after')
    def validate_topology(self) -> Self:
        if len(self.trafo) == 0:
            raise Exception(f'topology={self.uuid} has zero trafo entries')
        if len(self.bus) == 0:
            raise Exception(f'topology={self.uuid} has zero bus entries')
        if len(self.branch) == 0:
            raise Exception(f'topology={self.uuid} has zero branch entries')






        #bus_test = [b for b in self.bus if b.bus =='f4a9224e-79e6-5dd0-9d5f-f7cd40060f1d']
        #forward_search = {}
        #

        for bus in self.bus:
            if bus.bus is None:
                raise Exception(f'topology={self.uuid}, mrid={bus.bus}, exception due to unresolved bus mrid')
            if bus.rated_kv == 0:
                # attempt to recover bus voltages
                forward_result = self.recover_voltage(mrid=bus.bus)
                # apply recovered bus voltages
                if forward_result[bus.bus]:
                    for recovered_bus, recovered_bus_rated_kv in forward_result.items():
                        [bus_ for bus_ in self.bus if bus_.bus == recovered_bus][0].rated_kv = recovered_bus_rated_kv
                else:
                    # purge un-recovered buses and associated elements (branches and switches)
                    purge_bus_associations = forward_result.keys()
                    self.bus = [bus_ for bus_ in self.bus if bus_.bus not in purge_bus_associations]
                    self.switch = [switch_ for switch_ in self.switch if switch_.to_bus not in purge_bus_associations and switch_.from_bus not in purge_bus_associations]
                    self.branch = [branch_ for branch_ in self.switch if branch_.to_bus not in purge_bus_associations and branch_.from_bus not in purge_bus_associations]
        '''        
                
                recovered_bus_voltage = self.recover_bus_voltage(bus=bus)
                if recovered_bus_voltage == 0:
                    bus.in_service = False
                    logger.warning(f'topology={self.uuid} has an unrecoverable bus voltage for mrid={bus.bus}. Bus will be deactivated')
                else:
                    bus.rated_kv = recovered_bus_voltage
                    logger.info(f'topology={self.uuid}, mrid={bus.bus} recover voltage recovery and set to rated_kv={bus.rated_kv}')
        self.purge_duplicate_busses()
        '''


    def recover_bus_voltage(self, bus: ConnectivityNode):
        for branch in self.branch:
            if bus.bus == branch.to_bus:
                for node in self.bus:
                    if branch.from_bus == node.bus:
                        return node.rated_kv
        for switch in self.switch:
            if bus.bus == switch.to_bus:
                for node in self.bus:
                    if switch.from_bus == node.bus:
                        return node.rated_kv
        return 0

    def purge_duplicate_busses(self):
        minimal_set = []
        for bus in self.bus:
            if bus.bus not in [i.bus for i in minimal_set]:
                minimal_set.append(bus)
            else:
                logger.warning(f'topology={self.uuid} has a duplicate bus entry {bus.bus} which will be purged')
        self.bus = minimal_set
