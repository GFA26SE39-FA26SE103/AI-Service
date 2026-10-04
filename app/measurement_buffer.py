from collections import deque
from uuid import UUID
from app.monitoring_schemas import MeasurementBatch,MeasurementPage
from app.schemas import SessionState

class MeasurementBuffer:
    def __init__(self,session_id: UUID):
        self.session_id=session_id
        self._batches: deque[MeasurementBatch]=deque(maxlen=256)
        self._newest=0

    def clear(self): self._batches.clear()

    def append(self,batch: MeasurementBatch):
        if batch.session_id!=self.session_id or batch.sequence<=self._newest: raise ValueError("AI_MEASUREMENT_INVALID")
        self._newest=batch.sequence
        self._batches.append(batch.model_copy(deep=True))

    def read(self,after_sequence: int,limit: int,state: SessionState,*,wrong_session: bool=False,error_code: str|None=None) -> MeasurementPage:
        if not 1<=limit<=64 or after_sequence<0: raise ValueError("AI_REQUEST_INVALID")
        oldest=self._batches[0].sequence if self._batches else self._newest+1
        gap=wrong_session or after_sequence<oldest-1 or after_sequence>self._newest
        after=0 if wrong_session or after_sequence>self._newest else after_sequence
        batches=[b.model_copy(deep=True) for b in self._batches if b.sequence>after][:limit]
        return MeasurementPage(session_id=self.session_id,oldest_sequence=oldest,newest_sequence=self._newest,gap=gap,batches=batches,state=state,error_code=error_code)
