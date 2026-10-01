from __future__ import annotations

import asyncio
import os
from pathlib import Path
from uuid import uuid4

import pytest

from app.camera_reader import CameraReader
from app.schemas import SessionState, StartSessionRequest
from app.session_manager import SessionManager
from app.tracker import FrameTracker


@pytest.mark.asyncio
async def test_real_ip_webcam_reaches_live_and_returns_jpeg():
    stream_url = os.getenv("AI_TEST_STREAM_URL")
    if not stream_url:
        pytest.skip("Set AI_TEST_STREAM_URL to run the real IP Webcam contract test.")

    request = StartSessionRequest(
        stream_url=stream_url,
        username=os.getenv("AI_TEST_STREAM_USERNAME"),
        password=os.getenv("AI_TEST_STREAM_PASSWORD"),
        model=os.getenv("AI_TEST_MODEL", str(Path(__file__).parents[2] / "yolo26n.pt")),
    )
    manager = SessionManager(
        CameraReader,
        FrameTracker,
        reconnect_attempts=2,
        reconnect_delay_seconds=.5,
    )
    camera_id = uuid4()

    try:
        await manager.start(camera_id, request)
        async with asyncio.timeout(45):
            while manager.status_now(camera_id).state not in (SessionState.LIVE, SessionState.ERROR):
                await asyncio.sleep(.1)
        status = await manager.status(camera_id)
        assert status.state is SessionState.LIVE, status.error_code
        jpeg = await manager.frame(camera_id)
        assert jpeg[:2] == b"\xff\xd8"
        assert len(jpeg) > 1_000
    finally:
        await manager.stop(camera_id)

