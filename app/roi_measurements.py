from dataclasses import dataclass
import math
from uuid import UUID
from app.monitoring_schemas import MonitoringZoneInput,Point,ZoneMeasurement,_on

@dataclass(frozen=True)
class TrackedPerson:
    track_id: int
    score: float
    bbox: tuple[float,float,float,float]
    observed: bool

def contains_footpoint(roi: list[Point],bbox: tuple[float,float,float,float]) -> bool:
    if not all(math.isfinite(v) for v in bbox) or bbox[2]<=bbox[0] or bbox[3]<=bbox[1]: return False
    x=(bbox[0]+bbox[2])/2; y=bbox[3]
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
        current={t.track_id for t in tracks if t.observed and isinstance(t.track_id,int) and not isinstance(t.track_id,bool)
                 and t.track_id>0 and math.isfinite(t.score) and zone.confidence<=t.score<=1 and contains_footpoint(zone.roi_polygon,t.bbox)}
        previous=self._starts.get(zone.zone_id,{})
        starts={track_id:previous.get(track_id,elapsed_ms) for track_id in current}
        self._starts[zone.zone_id]=starts
        queued=sum(elapsed_ms-start>=5000 for start in starts.values()) if zone.queue_enabled else 0
        return ZoneMeasurement(zone_id=zone.zone_id,config_id=zone.config_id,config_version=zone.config_version,people_count=len(current),queue_count=queued)
