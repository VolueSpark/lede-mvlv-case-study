from pydantic import BaseModel, Field, model_validator, field_validator
from typing_extensions import Self
from typing import List, Optional
import polars as pl

from lib import logger


class UsagePoint(BaseModel):
    bus: str
    mrid: str
    meter_id: str
    cfl_mrid: str


class ConformLoad(BaseModel):
    bus: str
    mrid: str


class ConnectivityNode(BaseModel):
    # pandapower bus entity
    mrid: str=''

    # bus voltage spesification in base unit (volts)
    rated_kv: float = Field(
        default=0.0
    )


class GhostNodes(BaseModel):
    mrid: str
    from_bus: str
    to_bus: str


class AcLineSegment(BaseModel):
    mrid: str
    from_bus: str
    to_bus: str
    r: float
    x: float
    name: str

    # validate for line impedance
    @property
    def has_impedance(self):
        return (self.r > 0) or (self.x > 0)


class Switch(BaseModel):
    mrid: str
    from_bus: str
    to_bus: str
    is_open: bool
    name: str


class PowerTransformerEnd(BaseModel):
    bus: str
    rated_kv: float
    rated_kva: float
    number: int


class PowerTransformer(BaseModel):
    mrid: str
    end: List[PowerTransformerEnd]
    name: str
    # 2W/3W high voltage bus
    hv_bus: Optional[ConnectivityNode] = Field(
        default=ConnectivityNode()
    )

    # 3W medium voltage bus
    mv_bus: Optional[ConnectivityNode] = Field(
        default=ConnectivityNode()
    )

    # 2W/3W medium voltage bus
    lv_bus: Optional[ConnectivityNode] = Field(
        default=ConnectivityNode()
    )

    # operational capacity
    sn_mva: Optional[float] = Field(
        default=0.0)
    in_service: Optional[bool] = Field(
        default=False
    )

    @property
    def is_3w_trafo(self):
        return bool(len(self.end) == 3)



class Topology(BaseModel):
    uuid: str
    slack: List[ConnectivityNode]
    trafo: List[PowerTransformer]
    bus: List[ConnectivityNode]
    branch: List[AcLineSegment]
    switch: List[Switch]
    conform_load: Optional[List[ConformLoad]] = Field(default=[])
    load: Optional[List[UsagePoint]] = Field(default=[])
    ghost: Optional[List[GhostNodes]] = Field(default=[])

    @model_validator(mode='after')
    def validate_topology(self) -> Self:
        #
        # validation for vanilla topology
        #
        if not len(self.trafo):
            raise Exception(f'{self.uuid} has zero trafo entries and will be discarded')
        if not len(self.switch):
            raise Exception(f'{self.uuid} has zero switch entries and will be discarded')
        if not len(self.branch):
            raise Exception(f'{self.uuid} has zero branch entries and will be discarded')


        #
        # asset that topology has uuid tied in with trafo and grid
        #
        trafo_end_busses = []
        for trafo in self.trafo:
            for end in trafo.end:
                trafo_end_busses.append(end.bus)
        network_busses = [bus.mrid for bus in self.bus]

        assert self.uuid in trafo_end_busses and self.uuid in network_busses, f'{self.uuid} has no direct association between trafo and bus'

        #
        # validation on trafo resolving bus tie-ins
        #
        for trafo in self.trafo:
            end = pl.from_dicts([end.dict() for end in trafo.end])

            arg_min = end['rated_kv'].arg_min()
            trafo.lv_bus = ConnectivityNode(
                mrid=end[arg_min]['bus'].item(),
                rated_kv=end[arg_min]['rated_kv'].item()
            )
            arg_max = end['rated_kv'].arg_max()
            trafo.hv_bus = ConnectivityNode(
                mrid=end[arg_max]['bus'].item(),
                rated_kv=end[arg_max]['rated_kv'].item()
            )
            if trafo.is_3w_trafo:
                mv_element = end.filter(pl.col('bus') != trafo.lv_bus.mrid).filter(pl.col('bus') != trafo.hv_bus.mrid)
                trafo.mv_bus = ConnectivityNode(
                    mrid=mv_element['bus'].item(),
                    rated_kv=mv_element['rated_kv'].item()
                )
            trafo.sn_mva = end['rated_kva'].max() / 1000.0

            if not (trafo.hv_bus.rated_kv > trafo.lv_bus.rated_kv > 0):
                raise Exception(f'topology={self.uuid}, trafo.mrid={trafo.mrid} violate rated voltage levels')
            if not trafo.sn_mva > 0:
                raise Exception(f'topology={self.uuid}, trafo.mrid={trafo.mrid} violate rated capacity level levels')
            if trafo.is_3w_trafo and not (trafo.hv_bus.rated_kv > trafo.mv_bus.rated_kv > trafo.lv_bus.rated_kv > 0):
                raise Exception(f'topology={self.uuid}, trafo.mrid={trafo.mrid} violate rated voltage levels')

        #
        # validation of bus voltage values, purging
        #
        _busses = {}
        for bus in self.bus:
            if bus.mrid not in _busses.keys():
                _busses[bus.mrid] = bus
            else:
                if _busses[bus.mrid].rated_kv < bus.rated_kv:
                    _busses[bus.mrid].rated_kv = bus.rated_kv
        self.bus = list(_busses.values())

        # resolve bus voltage and purge if not possible
        for bus in self.bus:
            if bus.mrid is None:
                raise Exception(f'topology={self.uuid}, mrid={bus.mrid}, exception due to unresolved bus mrid')
            if bus.rated_kv == 0:

                logger.info(f'topology={self.uuid} with bus mrid={bus.mrid} has unspesified rated_kv={bus.rated_kv} and will be resolved automatically, if possible')
                recovered = self.recover_voltage(mrid=bus.mrid, forward_search={})
                self.restore_voltage(recovered=recovered)

        #
        # resolve conform load id's for usagepoints
        #
        for load in self.load:
            cfl_mrid = [cfl.mrid for cfl in self.conform_load if cfl.bus == load.bus]
            if cfl_mrid is not None:
                load.cfl_mrid = cfl_mrid[0]

        return self

    #
    # validation helper functions
    #
    def restore_voltage(self, recovered: dict):
        voltage_level = max(recovered.values())
        if bool(voltage_level):
            for bus in self.bus:

                if bus.mrid in recovered.keys() and not bool(bus.rated_kv):
                    logger.info(f'restoring voltage level {bus.mrid} {bus.rated_kv}->{voltage_level}')
                    bus.rated_kv = voltage_level
                #elif bool(bus.rated_kv) and bus.rated_kv != voltage_level:
                #    raise Exception(f'topology.uuid={self.uuid} has resolved a conflicting bus.mrid={bus.mrid} voltage ralation {voltage_level}<?>{bus.rated_kv}')
        else:
            logger.warning(f'purge unrsolve bus.mrid={recovered.keys()} relations')
            self.bus = [bus for bus in self.bus if bus.mrid not in recovered.keys()]
            self.switch = [switch for switch in self.switch if switch.to_bus not in recovered.keys() and switch.from_bus not in recovered.keys()]
            self.branch = [branch for branch in self.branch if branch.to_bus not in recovered.keys() and branch.from_bus not in recovered.keys() ]

    def recover_voltage(self, mrid: str, forward_search: dict) -> dict:

        # bus needs termination if deep search reveals nothing
        forward_search |= {bus.mrid:bus.rated_kv for bus in self.bus if bus.mrid == mrid}

        # terminate if visited bus does have a valid voltage
        if forward_search[mrid]:
            logger.info(f'Forward: {mrid} ({forward_search[mrid]}) -> Voltage Recovered', color=logger.GREEN)
            return forward_search

        # query all branch and switch segments to get next bus(s) destination
        next_mrid_via_switch = [switch.to_bus if switch.from_bus == mrid else switch.from_bus for switch in self.switch if switch.to_bus == mrid or switch.from_bus == mrid]
        next_mrid_via_branch = [branch.to_bus if branch.from_bus == mrid else branch.from_bus for branch in self.branch if branch.to_bus == mrid or branch.from_bus == mrid]

        # attempt to resolve voltages from next bus
        next_mrid_list = list(set(next_mrid_via_switch).union(set(next_mrid_via_branch)))
        next_mrid_list.sort()

        forward_return = forward_search
        for next_mrid in next_mrid_list:
            logger.info(f'Forward search: {mrid} ({forward_search[mrid]} kV) -> {next_mrid} (? kV)', color=logger.BLACK)
            if next_mrid not in forward_return.keys():
                forward_return |= self.recover_voltage(mrid=next_mrid, forward_search=forward_search)
        return forward_return
