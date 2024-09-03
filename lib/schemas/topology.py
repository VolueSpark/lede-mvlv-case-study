from pydantic import BaseModel, Field, AliasChoices, model_validator
from typing_extensions import Self
from typing import List, Optional, Tuple, Any
import polars as pl

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

    @model_validator(mode='after')
    def validate_topology(self) -> Self:
        if len(self.trafo) == 0:
            raise Exception(f'topology={self.uuid} has zero trafo entries')
        if len(self.bus) == 0:
            raise Exception(f'topology={self.uuid} has zero bus entries')
        if len(self.branch) == 0:
            raise Exception(f'topology={self.uuid} has zero branch entries')

        #forward_result = self.recover_voltage(mrid='7253644b-e40b-53b9-b9ec-d1e8b433dbda', root=True)
        #exit(1)

        #
        # trafo boiler plate
        #
        for trafo in self.trafo:
            end = pl.from_dicts([end.dict() for end in trafo.end])

            if end.filter(pl.col('bus') == self.uuid).is_empty():
                arg_min = end['rated_kv'].arg_min()
                trafo.lv_bus = end[arg_min]['bus'].item()
                trafo.vn_lv_kv =  end[arg_min]['rated_kv'].item()
            else:
                trafo.lv_bus = end.filter(pl.col('bus') == self.uuid)['bus'].item()
                trafo.vn_lv_kv =  end.filter(pl.col('bus') == self.uuid)['rated_kv'].item()

            arg_max = end['rated_kv'].arg_max()
            trafo.hv_bus = end[arg_max]['bus'].item()
            trafo.vn_hv_kv =  end[arg_max]['rated_kv'].item()

            trafo.sn_mva = end['rated_kva'].max() / 1000.0

            assert trafo.lv_bus != trafo.hv_bus, f'{trafo.__class__} mrid={trafo.mrid} raised exception due to unresolved trafo ending bus'
            assert trafo.vn_hv_kv > trafo.vn_lv_kv, f'{trafo.__class__} mrid={trafo.mrid} raised exception due to unresolved trafo ending voltage'
            assert trafo.sn_mva > 0, f'{trafo.__class__} mrid={trafo.mrid} raised exception due to unresolved trafo capacity'

        #
        # bus boiler plate
        #


        # determine unique bus entires and resolve to optimal voltage
        _busses = {}
        for bus in self.bus:
            if bus.bus not in _busses.keys():
                _busses[bus.bus] = bus
            else:
                if _busses[bus.bus].rated_kv < bus.rated_kv:
                    _busses[bus.bus].rated_kv = bus.rated_kv
                    logger.warning(f'topology={self.uuid} has duplicate bus {bus.bus} which will be removed. Defaults to {bus.bus} ({bus.rated_kv})')
        self.bus = list(_busses.values())

        # resolve bus voltage and purge if not possible
        for bus in self.bus:
            if bus.bus is None:
                raise Exception(f'topology={self.uuid}, mrid={bus.bus}, exception due to unresolved bus mrid')
            if bus.rated_kv == 0:
                forward_result = self.recover_voltage(mrid=bus.bus, root=True)
                if forward_result[bus.bus]:
                    self.apply_forward_result(forward_result = forward_result)
                else:
                    self.purge_forward_result(forward_result = forward_result)



    def apply_forward_result(self, forward_result: dict):
        for bus in self.bus:
            if bus.bus in forward_result.keys():
                bus.rated_kv = forward_result[bus.bus]
                logger.debug(msg = f'forward result applied for bus={bus.bus} with recovered rated_kv={bus.rated_kv}')

    def purge_forward_result(self, forward_result: dict):
        updated_bus = []
        updated_branch = []
        updated_switch = []
        for bus in self.bus:
            if bus.bus not in forward_result.keys():
                updated_bus.append(bus)
            else:
                logger.debug(msg = f'purging bus={bus.bus} with un-recovered voltage rated_kv={bus.rated_kv}')
        for switch in self.switch:
            if switch.to_bus not in forward_result.keys() and switch.from_bus not in forward_result.keys():
                updated_switch.append(switch)
            else:
                logger.debug(msg = f'purging switch={switch.mrid} due to un-recovered bus voltage')
        for branch in self.branch:
            if branch.to_bus not in forward_result.keys() and branch.from_bus not in forward_result.keys():
                updated_branch.append(branch)
            else:
                logger.debug(msg = f'purging branch={branch.mrid} due to un-recovered bus voltage')
        self.bus = updated_bus
        self.switch = updated_switch
        self.branch = updated_branch

    def recover_voltage(self, mrid: str, forward_search: dict=None, root: bool=False) -> dict:

        if forward_search is None:
            forward_search = {}

        # locate bus of interest
        bus = [bus for bus in self.bus if bus.bus == mrid][0]

        # bus needs termination if deep search reveals nothing
        current_bus = {bus.bus: bus.rated_kv}
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
        next_mrid_list = list(set(next_bus_via_switch).union(set(next_bus_via_branch)))
        next_mrid_list.sort()

        forward_result = {}
        for next_mrid in next_mrid_list:
            if next_mrid not in forward_search:
                logger.info(f'Forward search: {bus.bus} ({bus.rated_kv} kV) -> {next_mrid} (? kV)', color=logger.BLACK)
                forward_result |= self.recover_voltage(mrid=next_mrid, forward_search=forward_search) | current_bus
                if forward_result == current_bus:
                    logger.info(f'Forward: {next_mrid} (?) -> Voltage is unrecoverable', color=logger.RED)
                    return current_bus|{next_mrid:0}

        if not root:
            return forward_result

        if bool(len(forward_result)):
            rated_kv = max(forward_result.values())
            return {mrid: rated_kv for mrid in forward_result.keys()}

        logger.info(f'Forward: {mrid} (?) -> Voltage is unrecoverable', color=logger.RED)
        return current_bus

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
    '''
