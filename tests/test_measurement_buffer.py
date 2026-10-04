from datetime import datetime,timezone
from uuid import uuid4
import pytest
from app.measurement_buffer import MeasurementBuffer
from app.monitoring_schemas import MeasurementBatch
from app.schemas import SessionState

def test_bounded_ordered_cursor_reports_gaps_and_is_repeatable():
    session=uuid4(); buffer=MeasurementBuffer(session)
    for seq in range(1,301): buffer.append(MeasurementBatch(session_id=session,continuity_id=session,sequence=seq,captured_at=datetime.now(timezone.utc),source_elapsed_ms=seq*40,configuration_fingerprint="f",zones=[]))
    page=buffer.read(0,64,SessionState.LIVE)
    assert page.oldest_sequence==45 and page.newest_sequence==300 and page.gap
    assert [b.sequence for b in page.batches]==list(range(45,109))
    assert buffer.read(0,64,SessionState.LIVE)==page
    assert not buffer.read(108,64,SessionState.LIVE).gap
    assert buffer.read(300,64,SessionState.COMPLETED).batches==[]
    with pytest.raises(ValueError): buffer.read(0,65,SessionState.LIVE)
