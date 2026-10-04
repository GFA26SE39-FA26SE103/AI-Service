from dataclasses import dataclass
import math
from typing import Literal

@dataclass(frozen=True)
class SourceTime:
    elapsed_ms: int
    continuity_broken: bool

class SourceClock:
    def __init__(self,source_type: Literal["LIVE","RECORDED"],max_observation_gap_ms: int=2000):
        self.source_type=source_type
        self.max_gap=max_observation_gap_ms
        self._index=0
        self._last: int|None=None
        self._last_pts: float|None=None
        self._origin: float|None=None

    def observe(self,position_ms: float|None,fps: float|None,monotonic_seconds: float) -> SourceTime:
        if self.source_type=="LIVE":
            if not math.isfinite(monotonic_seconds): raise ValueError("AI_SOURCE_TIME_INVALID")
            if self._origin is None: self._origin=monotonic_seconds
            elapsed=round((monotonic_seconds-self._origin)*1000)
        else:
            valid_pts=position_ms is not None and math.isfinite(position_ms) and position_ms>=0 and position_ms!=self._last_pts
            valid_fps=fps is not None and math.isfinite(fps) and 1<=fps<=240
            if valid_pts: elapsed=round(position_ms)
            elif valid_fps: elapsed=round(self._index*1000/fps)
            else: raise ValueError("AI_SOURCE_TIME_INVALID")
            self._last_pts=position_ms
        if elapsed<0: raise ValueError("AI_SOURCE_TIME_INVALID")
        broken=self._last is not None and (elapsed<=self._last or (self.source_type=="LIVE" and elapsed-self._last>self.max_gap))
        self._last=elapsed
        self._index+=1
        return SourceTime(elapsed,broken)
