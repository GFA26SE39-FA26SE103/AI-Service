from __future__ import annotations

from datetime import timedelta
from typing import Annotated, Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator
from app.schemas import SessionState, StartSessionRequest

class Point(BaseModel):
    model_config=ConfigDict(extra="forbid",allow_inf_nan=False)
    x: Annotated[float,Field(ge=0,le=1)]
    y: Annotated[float,Field(ge=0,le=1)]

def _cross(a: Point,b: Point,c: Point) -> float:
    return (b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x)

def _on(a: Point,b: Point,p: Point) -> bool:
    return abs(_cross(a,b,p))<1e-9 and min(a.x,b.x)-1e-9<=p.x<=max(a.x,b.x)+1e-9 and min(a.y,b.y)-1e-9<=p.y<=max(a.y,b.y)+1e-9

def _intersect(a: Point,b: Point,c: Point,d: Point) -> bool:
    return (_cross(a,b,c)*_cross(a,b,d)<0 and _cross(c,d,a)*_cross(c,d,b)<0) or _on(a,b,c) or _on(a,b,d) or _on(c,d,a) or _on(c,d,b)

class MonitoringZoneInput(BaseModel):
    model_config=ConfigDict(extra="forbid",allow_inf_nan=False)
    zone_id: UUID
    config_id: UUID
    config_version: AwareDatetime
    roi_polygon: list[Point]=Field(min_length=3,max_length=1000)
    confidence: Annotated[float,Field(ge=0,le=1)]
    queue_enabled: bool=False

    @model_validator(mode="after")
    def valid(self):
        if self.config_version.utcoffset()!=timedelta(0): raise ValueError("UTC configuration version required")
        _valid_roi(self.roi_polygon)
        return self

def _valid_roi(points: list[Point]) -> list[Point]:
    if len({(p.x,p.y) for p in points})!=len(points): raise ValueError("Duplicate ROI points")
    area=sum(p.x*points[(i+1)%len(points)].y-p.y*points[(i+1)%len(points)].x for i,p in enumerate(points))
    if abs(area)<1e-9: raise ValueError("Degenerate ROI")
    n=len(points)
    for i in range(n):
        for j in range(i+1,n):
            if (j-i)%n in (1,n-1): continue
            if _intersect(points[i],points[(i+1)%n],points[j],points[(j+1)%n]): raise ValueError("Self-intersecting ROI")
    return points

class CheckoutMappingInput(BaseModel):
    model_config=ConfigDict(extra="forbid",allow_inf_nan=False)
    mapping_id: UUID
    counter_id: UUID
    zone_id: UUID
    config_id: UUID
    config_version: AwareDatetime
    mapping_version: AwareDatetime
    lane_roi: list[Point]=Field(min_length=3,max_length=1000)
    staff_roi: list[Point]=Field(min_length=3,max_length=1000)
    service_roi: list[Point]|None=Field(default=None,min_length=3,max_length=1000)
    is_queue_source: bool
    confidence: Annotated[float,Field(ge=0,le=1)]

    @field_validator("mapping_id","counter_id","zone_id","config_id")
    @classmethod
    def nonempty_id(cls,value: UUID) -> UUID:
        if value.int==0: raise ValueError("Checkout identity required")
        return value

    @field_validator("lane_roi","staff_roi","service_roi")
    @classmethod
    def valid_roi(cls,points: list[Point]|None) -> list[Point]|None: return _valid_roi(points) if points is not None else None

    @model_validator(mode="after")
    def valid_versions(self):
        if self.is_queue_source and self.service_roi is None: raise ValueError("Service ROI required for queue source")
        if self.config_version.utcoffset()!=timedelta(0) or self.mapping_version.utcoffset()!=timedelta(0):
            raise ValueError("UTC checkout versions required")
        return self

class StartMonitoringRequest(StartSessionRequest):
    protocol_version: Literal[1,2]=1
    owner_id: UUID=Field(repr=False)
    configuration_fingerprint: str=Field(min_length=1,max_length=128)
    zones: list[MonitoringZoneInput]=Field(default_factory=list,max_length=128)
    checkout_mappings: list[CheckoutMappingInput]=Field(default_factory=list,max_length=128)

    @model_validator(mode="after")
    def unique_zones(self):
        if len({z.zone_id for z in self.zones})!=len(self.zones): raise ValueError("Duplicate zones")
        if self.protocol_version==1 and (not self.zones or self.checkout_mappings): raise ValueError("Invalid v1 monitoring configuration")
        if self.protocol_version==2 and not self.checkout_mappings: raise ValueError("Checkout mappings required")
        if len({m.mapping_id for m in self.checkout_mappings})!=len(self.checkout_mappings): raise ValueError("Duplicate checkout mappings")
        if self.classes!=[0]: raise ValueError("Monitoring supports person class only")
        return self

class ZoneMeasurement(BaseModel):
    model_config=ConfigDict(extra="forbid")
    zone_id: UUID
    config_id: UUID
    config_version: AwareDatetime
    people_count: int=Field(ge=0,strict=True)
    queue_count: int=Field(ge=0,strict=True)

    @model_validator(mode="after")
    def subset(self):
        if self.queue_count>self.people_count: raise ValueError("Queue count exceeds people count")
        return self

class CheckoutMeasurement(BaseModel):
    model_config=ConfigDict(extra="forbid")
    mapping_id: UUID
    counter_id: UUID
    config_id: UUID
    config_version: AwareDatetime
    mapping_version: AwareDatetime
    person_present: bool|None
    queue_count: Annotated[int,Field(ge=0,strict=True)]|None
    max_wait_seconds: Annotated[float,Field(ge=0)]|None=None
    usable: bool

class MeasurementBatch(BaseModel):
    model_config=ConfigDict(extra="forbid")
    protocol_version: Literal[1,2]=1
    session_id: UUID
    continuity_id: UUID
    sequence: int=Field(ge=1,strict=True)
    captured_at: AwareDatetime
    source_elapsed_ms: int=Field(ge=0,strict=True)
    configuration_fingerprint: str
    zones: list[ZoneMeasurement]
    checkouts: list[CheckoutMeasurement]=Field(default_factory=list,max_length=128)

class MeasurementPage(BaseModel):
    protocol_version: Literal[1,2]=1
    session_id: UUID
    oldest_sequence: int
    newest_sequence: int
    gap: bool
    batches: list[MeasurementBatch]
    state: SessionState
    error_code: str|None=None
