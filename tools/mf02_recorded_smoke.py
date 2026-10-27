from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import datetime,timezone
from pathlib import Path
from typing import Awaitable,Callable,Sequence
from uuid import uuid4

from app.camera_reader import CameraReader
from app.monitoring_schemas import MeasurementBatch,MonitoringZoneInput,StartMonitoringRequest
from app.monitoring_tracker import MonitoringFrameTracker
from app.schemas import SessionState
from app.session_manager import SessionManager


_PUBLIC_SUMMARY_KEYS=(
    "ok","healthy","state","fingerprint","batch_count","sequence_first",
    "sequence_last","sequence_contiguous","continuity_count",
    "source_elapsed_ms_last","people_min","people_max","queue_min",
    "queue_max","warning_batches","critical_batches","warning_reached",
    "critical_reached","page_gap","counts_valid","fingerprint_consistent",
)


def summarize_batches(
    batches: Sequence[MeasurementBatch],
    state: SessionState | str,
    fingerprint: str,
    page_gap: bool,
    warning_threshold: int=3,
    critical_threshold: int=5,
) -> dict[str,object]:
    state_value=state.value if isinstance(state,SessionState) else str(state)
    sequences=[batch.sequence for batch in batches]
    expected=list(range(sequences[0],sequences[-1]+1)) if sequences else []
    sequence_contiguous=bool(sequences) and sequences==expected
    measurements=[zone for batch in batches for zone in batch.zones]
    people=[zone.people_count for zone in measurements]
    queues=[zone.queue_count for zone in measurements]
    counts_valid=bool(measurements) and all(
        isinstance(zone.people_count,int)
        and not isinstance(zone.people_count,bool)
        and isinstance(zone.queue_count,int)
        and not isinstance(zone.queue_count,bool)
        and 0<=zone.queue_count<=zone.people_count
        for zone in measurements
    )
    fingerprint_consistent=bool(batches) and all(
        batch.configuration_fingerprint==fingerprint for batch in batches
    )
    warning_batches=sum(queue>=warning_threshold for queue in queues)
    critical_batches=sum(queue>=critical_threshold for queue in queues)
    healthy=(
        bool(batches)
        and not page_gap
        and sequence_contiguous
        and counts_valid
        and fingerprint_consistent
        and state_value in (SessionState.LIVE.value,SessionState.COMPLETED.value)
    )
    return {
        "ok":healthy and warning_batches>0 and critical_batches>0,
        "healthy":healthy,
        "state":state_value,
        "fingerprint":fingerprint,
        "batch_count":len(batches),
        "sequence_first":sequences[0] if sequences else None,
        "sequence_last":sequences[-1] if sequences else None,
        "sequence_contiguous":sequence_contiguous,
        "continuity_count":len({batch.continuity_id for batch in batches}),
        "source_elapsed_ms_last":batches[-1].source_elapsed_ms if batches else None,
        "people_min":min(people) if people else None,
        "people_max":max(people) if people else None,
        "queue_min":min(queues) if queues else None,
        "queue_max":max(queues) if queues else None,
        "warning_batches":warning_batches,
        "critical_batches":critical_batches,
        "warning_reached":warning_batches>0,
        "critical_reached":critical_batches>0,
        "page_gap":page_gap,
        "counts_valid":counts_valid,
        "fingerprint_consistent":fingerprint_consistent,
    }


def _parse_roi(value: str) -> list[dict[str,float]]:
    points=[]
    for pair in value.split(";"):
        x_text,y_text=pair.split(",",maxsplit=1)
        points.append({"x":float(x_text),"y":float(y_text)})
    return points


def _parser() -> argparse.ArgumentParser:
    parser=argparse.ArgumentParser(description="Run a sanitized MF-02 recorded-video smoke check")
    parser.add_argument("--video",required=True)
    parser.add_argument("--duration-sec",type=float,default=0)
    parser.add_argument("--confidence",type=float,default=.50)
    parser.add_argument("--fingerprint",default="mf02-recorded-smoke-v1")
    parser.add_argument("--model",default="../yolo26s.pt")
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--roi",default="0,0;1,0;1,1;0,1")
    parser.add_argument("--warning-threshold",type=int,default=3)
    parser.add_argument("--critical-threshold",type=int,default=5)
    parser.add_argument("--wall-timeout-sec",type=float,default=0)
    return parser


async def run_recorded_smoke(options: argparse.Namespace) -> dict[str,object]:
    video=Path(options.video).resolve(strict=True)
    camera_id,owner_id,zone_id,config_id=uuid4(),uuid4(),uuid4(),uuid4()
    request=StartMonitoringRequest(
        stream_url=video.as_uri(),
        source_type="RECORDED",
        owner_id=owner_id,
        configuration_fingerprint=options.fingerprint,
        model=options.model,
        device=options.device,
        half=str(options.device).lower().startswith("cuda"),
        confidence=options.confidence,
        zones=[MonitoringZoneInput(
            zone_id=zone_id,
            config_id=config_id,
            config_version=datetime.now(timezone.utc),
            roi_polygon=_parse_roi(options.roi),
            confidence=options.confidence,
            queue_enabled=True,
        )],
    )
    timeout=options.wall_timeout_sec or max(30.0,options.duration_sec+30.0)
    manager=SessionManager(
        lambda body: CameraReader(body,recorded_root=video.parent),
        lambda body: MonitoringFrameTracker(body),
        monitoring_tracker_factory=lambda body: MonitoringFrameTracker(body),
        session_idle_timeout_seconds=timeout+10,
    )
    batches=[]; page_gap=False; after_sequence=0; after_session_id=None
    state=SessionState.STARTING
    try:
        await manager.start_monitoring(camera_id,request)
        async with asyncio.timeout(timeout):
            while True:
                page=await manager.measurements(camera_id,owner_id,after_session_id,after_sequence,64)
                page_gap=page_gap or page.gap
                if page.batches:
                    batches.extend(page.batches)
                    after_sequence=page.batches[-1].sequence
                after_session_id=page.session_id
                state=page.state
                if options.duration_sec>0 and batches and batches[-1].source_elapsed_ms>=round(options.duration_sec*1000):
                    break
                if state in (SessionState.COMPLETED,SessionState.ERROR,SessionState.STOPPED):
                    break
                await asyncio.sleep(.05)
    finally:
        await manager.shutdown()
    return summarize_batches(
        batches,state,options.fingerprint,page_gap,
        options.warning_threshold,options.critical_threshold,
    )


async def cli_async(
    argv: Sequence[str] | None=None,
    *,
    runner: Callable[[argparse.Namespace],Awaitable[dict[str,object]]]=run_recorded_smoke,
) -> int:
    try:
        options=_parser().parse_args(argv)
        ultralytics_logger=logging.getLogger("ultralytics")
        previous_level=ultralytics_logger.level
        ultralytics_logger.setLevel(logging.ERROR)
        try:
            result=await runner(options)
        finally:
            ultralytics_logger.setLevel(previous_level)
        public={key:result[key] for key in _PUBLIC_SUMMARY_KEYS if key in result}
        if not public.get("healthy"):
            public["ok"]=False
            public["error_code"]="AI_SMOKE_UNHEALTHY"
        elif not public.get("warning_reached") or not public.get("critical_reached"):
            public["ok"]=False
            public["error_code"]="AI_SMOKE_THRESHOLDS_UNMET"
        print(json.dumps(public,sort_keys=True,separators=(",",":")))
        return 0 if public.get("ok") else 2
    except Exception:
        print('{"error_code":"AI_SMOKE_FAILED","healthy":false,"ok":false}')
        return 2


def main(argv: Sequence[str] | None=None) -> int:
    return asyncio.run(cli_async(argv))


if __name__=="__main__":
    raise SystemExit(main())
