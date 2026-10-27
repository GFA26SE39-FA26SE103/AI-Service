import json
import io
import logging
from datetime import datetime,timezone
from uuid import uuid4

import pytest

from app.monitoring_schemas import MeasurementBatch,ZoneMeasurement
from app.schemas import SessionState
from tools.mf02_recorded_smoke import cli_async,summarize_batches


def batch(sequence,people,queue,*,session_id,continuity_id,fingerprint="mf02-demo"):
    return MeasurementBatch(
        session_id=session_id,
        continuity_id=continuity_id,
        sequence=sequence,
        captured_at=datetime.now(timezone.utc),
        source_elapsed_ms=sequence*1000,
        configuration_fingerprint=fingerprint,
        zones=[ZoneMeasurement(
            zone_id=uuid4(),
            config_id=uuid4(),
            config_version=datetime.now(timezone.utc),
            people_count=people,
            queue_count=queue,
        )],
    )


def test_summary_reports_contiguous_threshold_measurements_without_identifiers():
    session_id=uuid4(); continuity_id=uuid4()
    batches=[
        batch(1,4,3,session_id=session_id,continuity_id=continuity_id),
        batch(2,6,5,session_id=session_id,continuity_id=continuity_id),
    ]

    summary=summarize_batches(batches,SessionState.COMPLETED,"mf02-demo",False)
    encoded=json.dumps(summary)

    assert summary["healthy"] is True
    assert summary["sequence_contiguous"] is True
    assert summary["continuity_count"]==1
    assert summary["warning_reached"] is True
    assert summary["critical_reached"] is True
    assert summary["people_max"]==6
    assert summary["queue_max"]==5
    assert str(session_id) not in encoded
    assert str(continuity_id) not in encoded
    assert "owner_id" not in encoded
    assert "video" not in encoded


@pytest.mark.parametrize(
    ("sequences","page_gap"),
    [([1,2],True),([1,3],False)],
)
def test_summary_marks_gap_or_noncontiguous_sequence_unhealthy(sequences,page_gap):
    session_id=uuid4(); continuity_id=uuid4()
    batches=[batch(sequence,4,3,session_id=session_id,continuity_id=continuity_id) for sequence in sequences]

    summary=summarize_batches(batches,SessionState.LIVE,"mf02-demo",page_gap)

    assert summary["healthy"] is False


@pytest.mark.parametrize("state",[SessionState.ERROR,SessionState.STOPPED,SessionState.RECONNECTING])
def test_summary_rejects_failed_terminal_state_after_thresholds(state):
    session_id=uuid4(); continuity_id=uuid4()
    batches=[
        batch(1,4,3,session_id=session_id,continuity_id=continuity_id),
        batch(2,6,5,session_id=session_id,continuity_id=continuity_id),
    ]

    summary=summarize_batches(batches,state,"mf02-demo",False)

    assert summary["warning_reached"] is True
    assert summary["critical_reached"] is True
    assert summary["healthy"] is False
    assert summary["ok"] is False


@pytest.mark.asyncio
async def test_cli_reports_unmet_threshold_without_echoing_video_path(capsys):
    private_path="C:/private/camera-credentials/demo-secret.mp4"
    raw_identifier=str(uuid4())

    async def below_threshold(_options):
        return {
            "ok":False,
            "healthy":True,
            "state":"COMPLETED",
            "fingerprint":"mf02-demo",
            "batch_count":2,
            "warning_reached":False,
            "critical_reached":False,
            "runtime_note":raw_identifier,
        }

    exit_code=await cli_async(["--video",private_path],runner=below_threshold)
    output=capsys.readouterr().out

    assert exit_code!=0
    assert private_path not in output
    assert raw_identifier not in output
    assert json.loads(output)["error_code"]=="AI_SMOKE_THRESHOLDS_UNMET"


@pytest.mark.asyncio
async def test_cli_emits_one_json_line_when_runtime_logs_a_warning(capsys):
    from ultralytics.utils import LOGGER
    warning_output=io.StringIO()
    handler=logging.StreamHandler(warning_output)
    LOGGER.addHandler(handler)

    async def warning_runtime(_options):
        LOGGER.warning("runtime warning must not corrupt machine-readable output")
        return {
            "ok":True,
            "healthy":True,
            "state":"COMPLETED",
            "fingerprint":"mf02-demo",
            "batch_count":2,
            "warning_reached":True,
            "critical_reached":True,
        }

    try:
        exit_code=await cli_async(["--video","safe.mp4"],runner=warning_runtime)
    finally:
        LOGGER.removeHandler(handler)
    lines=capsys.readouterr().out.splitlines()

    assert exit_code==0
    assert warning_output.getvalue()==""
    assert len(lines)==1
    assert json.loads(lines[0])["ok"] is True
