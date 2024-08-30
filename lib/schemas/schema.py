from pydantic import BaseModel, Field, field_validator, model_validator
from typing_extensions import Self
from typing import List, Optional
import polars as pl
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
# a general bus entity between branches
#
class ConnectivityNode(BaseModel):
    # pandapower bus entity
    bus: str = Field(
        alias='mrid'
    )
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
    name: Optional[str] = Field(
        alias='name',
        default=f'branch_{uuid.uuid4().__str__()}'
    )

    # validate for line impedance
    @property
    def has_impedance(self):
        return (self.r > 0) or (self.x > 0)

    @field_validator('x')
    @classmethod
    def validate_x(cls, value):
        if (value < 0) or(value > 0.35):
            return 0

    @field_validator('r')
    @classmethod
    def validate_x(cls, value):
        if (value < 0) or(value > 1.8769):
            return 0


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
    name: Optional[str] = Field(
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
    # pandapower high-voltage bus connection. resolved dynamically from end
    hv_bus: Optional[str] = Field(
        default=''
    )
    # pandapower low-voltage bus connection. resolved dynamically from end
    lv_bus: Optional[str] = Field(
        default=''
    )
    # pandapower capacity. resolved dynamically from end
    sn_mva: Optional[float] = Field(
        default=0.0)
    # pandapower high voltage bus rated voltage
    vn_hv_kv: Optional[float] = Field(
        default=0.0
    )
    # pandapower low voltage bus rated voltage
    vn_lv_kv: Optional[float] = Field(
        default=0.0
    )
    # assume in service
    in_service: Optional[bool] = Field(
        default=True
    )
    # optional name
    name: Optional[str] = Field(alias='name', default='TF_')

    @model_validator(mode='after')
    def validate(self) -> Self:
        end = pl.from_dicts([end.dict() for end in self.end])

        arg_min = end['rated_kv'].arg_min()
        self.lv_bus = end[arg_min]['bus'].item()
        self.vn_lv_kv =  end[arg_min]['rated_kv'].item()

        arg_max = end['rated_kv'].arg_max()
        self.hv_bus = end[arg_max]['bus'].item()
        self.vn_hv_kv =  end[arg_max]['rated_kv'].item()

        self.sn_mva = end['rated_kva'].max() / 1000.0

        assert self.lv_bus != self.hv_bus, f'{self.__class__} mrid={self.mrid} raised exception due to unresolved trafo ending bus'
        assert self.vn_hv_kv > self.vn_lv_kv, f'{self.__class__} mrid={self.mrid} raised exception due to unresolved trafo ending voltage'
        assert self.sn_mva > 0, f'{self.__class__} mrid={self.mrid} raised exception due to unresolved trafo capacity'

        return self



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
    to_bus: str = Field(alias='toMrid')