from pydantic import BaseModel, Field, AliasChoices
from typing import List, Optional

from lib.schemas.schema import (
    PowerTransformer,
    ConnectivityNode,
    AcLineSegment,
    Switch,
    UsagePoint,
    GhostNodes
)

class Topology(BaseModel):
    uuid: str = Field(
        validation_alias=AliasChoices('neighbourhoodId', 'regionId')
    )

    slack: List[ConnectivityNode] = Field(alias='slackBus')
    trafo: List[PowerTransformer] = Field(alias='powerTransformers')
    bus: List[ConnectivityNode] = Field(alias='connectivityNodes')
    branch: List[AcLineSegment] = Field(alias='acLineSegments')
    switch: List[Switch] = Field(alias='switches')
    conform_load: List[ConnectivityNode] = Field(alias='conformLoads')
    load: Optional[List[UsagePoint]] = Field(alias='usagePoints')
    ghost: Optional[List[GhostNodes]] = Field(alias='ghostNodes')


