from pydantic import BaseModel, Field, AliasChoices, field_validator
from typing import List, Optional
import uuid


#
# associated with an AMI
#
class UsagePoint(BaseModel):
    # pandapower bus
    bus: str = Field(
        alias='fromMrid'
    )
    # cim mrid
    mrid: str = Field(
        alias='mrid'
    )
    # AEN of meter
    meter_id: Optional[str] = Field(
        alias='meterPointId',
        default=None
    )

    # conform load that cluster usage points
    cfl_mrid: Optional[str] = Field(
        alias='conformLoadId',
        default=None
    )


#
# a grouping of one or more Usagepoints
#
class ConformLoad(BaseModel):
    # pandapower bus
    bus: str = Field(
        alias='fromMrid'
    )
    # cim mrid
    mrid: str = Field(
        alias='mrid'
    )




#
# unresolved elements connecting drifting usagepoints
#
class GhostNodes(BaseModel):
    # cim mrid
    mrid: str = Field(
        alias='mrid'
    )
    # pandapower from bus
    from_bus: str = Field(
        alias='fromMrid'
    )
    # pandpower to bus
    to_bus: str = Field(
        alias='toMrid'
    )


#
# a general bus entity between branches
#
class ConnectivityNode(BaseModel):
    # pandapower bus entity
    mrid: str

    # bus voltage spesification in base unit (volts)
    rated_kv: float = Field(
        alias='voltageLevel',
        default=0.0
    )
    # pandapower option
    in_service: bool = Field(
        default=True
    )

    # validator to convert to kilovolt
    @field_validator('rated_kv')
    @classmethod
    def kilo_validator(cls, value):
        return value / 1000


#
# a general branch segment
#
class AcLineSegment(BaseModel):
    # cim mrid
    mrid: str = Field(
        alias='mrid'
    )
    # from pandapower bus node
    from_bus: str = Field(
        alias='fromMrid'
    )
    # to pandapower bus node
    to_bus: str = Field(
        alias='toMrid'
    )
    # line resistance, ohm
    r: float = Field(
        alias='resistance',
        default=0
    )
    # line reactance,  ohm
    x: float = Field(
        alias='reactance',
        default=0)
    # optional name is supplied
    name: str = Field(
        alias='name',
        default=f'branch_{uuid.uuid4().__str__()}'
    )

    @field_validator('x')
    @classmethod
    def validate_x(cls, value):
        if (value < 0) or(value > 0.35):
            return 0
        return value

    @field_validator('r')
    @classmethod
    def validate_r(cls, value):
        if (value < 0) or(value > 1.8769):
            return 0
        return value


#
# switch segment connecting two buses
#
class Switch(BaseModel):
    # cim mrid
    mrid: str = Field(
        alias='mrid'
    )
    # from pandapower bus
    from_bus: str = Field(
        alias='fromMrid'
    )
    # to pandapower bus
    to_bus: str = Field(
        alias='toMrid'
    )
    # closed switch in default
    is_open: bool = Field(
        alias='isOpen',
        default=False
    )
    # optional name, else generated
    name: str = Field(
        alias='name',
        default=f'switch__{uuid.uuid4().__str__()}'
    )


#
# power transformed winding end
#
class PowerTransformerEnd(BaseModel):
    # pandapower bus connection
    bus: str = Field(
        alias='mrid'
    )
    # winding rated voltage in volts, converted to kilo volt
    rated_kv: float = Field(
        alias='ratedU'
    )
    # winding rated capacity in volts-ampere, converted to kilo volt-ampere
    rated_kva: float = Field(
        alias='ratedS'
    )
    # winding numnber being either 1,2,3
    number: int = Field(alias='endNumber')

    @field_validator('rated_kv', 'rated_kva')
    @classmethod
    def kilo_validator(cls, value):
        return value / 1000

#
# power transformer for pandapower
#
class PowerTransformer(BaseModel):
    # cim mrid
    mrid: str = Field(
        alias='mrid'
    )
    # metadata for windings of transformer
    end: List[PowerTransformerEnd] = Field(
        alias='powerTransformerEnds'
    )

    # optional name
    name: str = Field(
        alias='name',
        default=f'trafo__{uuid.uuid4().__str__()}'
    )



class MemgraphEvent(BaseModel):
    uuid: str = Field(
        validation_alias=AliasChoices('neighbourhoodId', 'regionId', 'uuid')
    )

    slack: List[ConnectivityNode] = Field(alias='slackBus')
    trafo: List[PowerTransformer] = Field(alias='powerTransformers')
    bus: List[ConnectivityNode] = Field(alias='connectivityNodes')
    branch: List[AcLineSegment] = Field(alias='acLineSegments')
    switch: List[Switch] = Field(alias='switches')
    conform_load: List[ConformLoad] = Field(alias='conformLoads')
    load: List[UsagePoint] = Field(alias='usagePoints')
    ghost: List[GhostNodes] = Field(alias='ghostNodes')