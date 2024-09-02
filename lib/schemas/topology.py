from pydantic import BaseModel, Field, AliasChoices, model_validator
from typing_extensions import Self
from typing import List, Optional

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

        for bus in self.bus:
            if bus.bus is None:
                raise Exception(f'topology={self.uuid}, mrid={bus.bus}, exception due to unresolved bus mrid')
            if bus.rated_kv == 0:
                recovered_bus_voltage = self.recover_bus_voltage(bus=bus)
                if recovered_bus_voltage == 0:
                    bus.in_service = False
                    logger.warning(f'topology={self.uuid} has an unrecoverable bus voltage for mrid={bus.bus}. Bus will be deactivated')
                else:
                    bus.rated_kv = recovered_bus_voltage
        self.purge_duplicate_busses()


    def recover_bus_voltage(self, bus: ConnectivityNode):
        logger.warning(f'topology={self.uuid}, mrid={bus.bus} attempts bus voltage recovery due to poor conditioning rated_kv={bus.rated_kv}')
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

    def purge_duplicate_busses(self):
        minimal_set = []
        for bus in self.bus:
            if bus.bus not in [i.bus for i in minimal_set]:
                minimal_set.append(bus)
            else:
                logger.warning(f'topology={self.uuid} has a duplicate bus entry {bus.bus} which will be purged')
        self.bus = minimal_set

