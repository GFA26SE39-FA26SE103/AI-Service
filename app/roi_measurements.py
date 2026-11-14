from dataclasses import dataclass
import math
from uuid import UUID
from app.monitoring_schemas import CheckoutMappingInput,CheckoutMeasurement,MonitoringZoneInput,Point,ZoneMeasurement,_on

@dataclass(frozen=True)
class TrackedPerson:
    track_id: int
    score: float
    bbox: tuple[float,float,float,float]
    observed: bool

_BOTTOM_EDGE_Y = 0.99

def contains_footpoint(roi: list[Point],bbox: tuple[float,float,float,float]) -> bool:
    return _contains_person_anchor(roi, bbox)

def _contains_person_anchor(
    roi: list[Point],
    bbox: tuple[float, float, float, float],
    *,
    use_visible_center_at_bottom: bool = False,
) -> bool:
    left, top, right, bottom = bbox
    if not all(math.isfinite(value) for value in bbox) or right <= left or bottom <= top:
        return False
    x = (left + right) / 2
    y = bottom
    if use_visible_center_at_bottom and bottom >= _BOTTOM_EDGE_Y and top < 1:
        # A box at the bottom edge may end where the frame cuts off the person.
        y = (max(0, top) + min(1, bottom)) / 2
    if not (0<=x<=1 and 0<=y<=1): return False
    point=Point(x=x,y=y)
    inside=False
    for i,a in enumerate(roi):
        b=roi[(i+1)%len(roi)]
        if _on(a,b,point): return True
        if (a.y>y)!=(b.y>y) and x<(b.x-a.x)*(y-a.y)/(b.y-a.y)+a.x: inside=not inside
    return inside

class RoiMeasurements:
    def __init__(self):
        self._continuity: UUID|None=None
        self._starts: dict[UUID,dict[int,int]]={}
        self._last_ms: int|None=None

    def reset(self):
        self._starts.clear(); self._last_ms=None; self._continuity=None

    def observe(self,zone: MonitoringZoneInput,tracks: list[TrackedPerson],elapsed_ms: int,continuity_id: UUID) -> ZoneMeasurement:
        if self._continuity!=continuity_id or (self._last_ms is not None and elapsed_ms<self._last_ms): self.reset()
        self._continuity=continuity_id; self._last_ms=elapsed_ms
        current = {
            t.track_id for t in tracks
            if t.observed and isinstance(t.track_id, int) and not isinstance(t.track_id, bool)
            and t.track_id > 0 and math.isfinite(t.score) and zone.confidence <= t.score <= 1
            and _contains_person_anchor(zone.roi_polygon, t.bbox, use_visible_center_at_bottom=True)
        }
        previous=self._starts.get(zone.zone_id,{})
        starts={track_id:previous.get(track_id,elapsed_ms) for track_id in current}
        self._starts[zone.zone_id]=starts
        queued=sum(elapsed_ms-start>=5000 for start in starts.values()) if zone.queue_enabled else 0
        return ZoneMeasurement(zone_id=zone.zone_id,config_id=zone.config_id,config_version=zone.config_version,people_count=len(current),queue_count=queued)

class CheckoutRoiMeasurements:
    def __init__(self):
        self._continuity: UUID|None=None
        self._last_ms: int|None=None
        self._starts: dict[UUID,dict[int,int]]={}

    def observe(self,mapping: CheckoutMappingInput,tracks: list[TrackedPerson],elapsed_ms: int,continuity_id: UUID) -> CheckoutMeasurement:
        if self._continuity!=continuity_id or (self._last_ms is not None and elapsed_ms<self._last_ms): self._starts.clear()
        self._continuity=continuity_id; self._last_ms=elapsed_ms
        valid={t.track_id:t for t in tracks if t.observed and isinstance(t.track_id,int) and not isinstance(t.track_id,bool)
               and t.track_id>0 and math.isfinite(t.score) and mapping.confidence<=t.score<=1}
        present=any(contains_footpoint(mapping.staff_roi,t.bbox) for t in valid.values())
        count=None
        max_wait=None
        if mapping.is_queue_source:
            current={key for key,t in valid.items() if contains_footpoint(mapping.lane_roi,t.bbox)
                     and not contains_footpoint(mapping.service_roi,t.bbox)}
            old=self._starts.get(mapping.mapping_id,{})
            starts={key:old.get(key,elapsed_ms) for key in current}
            self._starts[mapping.mapping_id]=starts
            count=sum(elapsed_ms-start>=5000 for start in starts.values())
            max_wait=max((elapsed_ms-start for start in starts.values()),default=0)/1000
        return CheckoutMeasurement(mapping_id=mapping.mapping_id,counter_id=mapping.counter_id,config_id=mapping.config_id,
            config_version=mapping.config_version,mapping_version=mapping.mapping_version,person_present=present,
            queue_count=count,max_wait_seconds=max_wait,usable=True)
