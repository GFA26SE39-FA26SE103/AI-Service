import asyncio
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import pytest

from app.camera_reader import CameraReader
from app.schemas import SessionState, StartSessionRequest
from app.session_manager import SessionManager
from tests.fakes import FakeTracker


def make_video(path: Path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    assert writer.isOpened()
    for value in (40, 80, 120):
        writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
    writer.release()


@pytest.mark.asyncio
async def test_recorded_video_finishes_without_reconnect_and_keeps_last_frame(tmp_path):
    path = tmp_path / "sample.mp4"
    make_video(path)
    readers = []

    def reader_factory(request):
        reader = CameraReader(request, recorded_root=tmp_path)
        readers.append(reader)
        return reader

    manager = SessionManager(reader_factory, lambda _: FakeTracker())
    camera_id = uuid4()
    started = asyncio.get_running_loop().time()
    await manager.start(camera_id, StartSessionRequest(stream_url=path.as_uri(), source_type="RECORDED"))
    async with asyncio.timeout(3):
        while manager.status_now(camera_id).state.value not in ("COMPLETED", "ERROR"):
            await asyncio.sleep(.01)

    status = await manager.status(camera_id)
    assert status.state.value == "COMPLETED"
    assert status.frame_sequence == 3
    assert status.error_code is None
    assert len(readers) == 1
    assert asyncio.get_running_loop().time() - started >= .25
    assert manager.active_worker_count == 0
    assert await manager.frame(camera_id)
    await manager.stop(camera_id)
    assert manager.status_now(camera_id).state is SessionState.STOPPED

    # Replay starts from frame one with a new reader, not a reconnect at the EOF position.
    await manager.start(camera_id, StartSessionRequest(stream_url=path.as_uri(), source_type="RECORDED"))
    async with asyncio.timeout(3):
        while manager.status_now(camera_id).state.value not in ("COMPLETED", "ERROR"):
            await asyncio.sleep(.01)
    assert manager.status_now(camera_id).frame_sequence == 3
    assert len(readers) == 2
    await manager.stop(camera_id)


@pytest.mark.asyncio
async def test_next_frame_waits_for_new_sequence_and_returns_no_duplicate_at_eof(tmp_path):
    path = tmp_path / "sample.mp4"
    make_video(path)
    manager = SessionManager(lambda request: CameraReader(request, recorded_root=tmp_path), lambda _: FakeTracker())
    camera_id = uuid4()
    await manager.start(camera_id, StartSessionRequest(stream_url=path.as_uri(), source_type="RECORDED"))

    first = await manager.next_frame(camera_id, after_sequence=0, after_session_id=None)
    assert first is not None
    first_jpeg, first_sequence, session_id = first
    assert first_jpeg and first_sequence >= 1

    next_frame = await manager.next_frame(camera_id, after_sequence=first_sequence, after_session_id=session_id)
    assert next_frame is not None
    assert next_frame[1] > first_sequence
    assert next_frame[2] == session_id

    async with asyncio.timeout(3):
        while manager.status_now(camera_id).state is not SessionState.COMPLETED:
            await asyncio.sleep(.01)
    last_sequence = manager.status_now(camera_id).frame_sequence
    assert await manager.next_frame(camera_id, after_sequence=last_sequence, after_session_id=session_id) is None
    assert (await manager.next_frame(camera_id, after_sequence=last_sequence, after_session_id=uuid4()))[1] == last_sequence
    manager._sessions[camera_id].status.state = SessionState.ERROR
    assert await manager.next_frame(camera_id, after_sequence=0, after_session_id=None) is None
    await manager.stop(camera_id)


def test_recorded_reader_rejects_path_outside_storage(tmp_path):
    outside = tmp_path.parent / "outside.mp4"
    with pytest.raises(ValueError, match="AI_RECORDED_VIDEO_INVALID"):
        CameraReader(StartSessionRequest(stream_url=outside.as_uri(), source_type="RECORDED"), recorded_root=tmp_path)


def test_live_request_cannot_open_local_file(tmp_path):
    with pytest.raises(ValueError, match="AI_SOURCE_INVALID"):
        CameraReader(StartSessionRequest(stream_url=(tmp_path / "sample.mp4").as_uri()))


@pytest.mark.parametrize("uri", ["file://other-host/share/sample.mp4", "http://example.test/video.mp4", "file:///C:/outside.mp4?key=secret", "file:///C:/outside.mp4#fragment"])
def test_recorded_reader_rejects_nonlocal_or_ambiguous_uri(tmp_path, uri):
    with pytest.raises(ValueError, match="AI_RECORDED_VIDEO_INVALID"):
        CameraReader(StartSessionRequest(stream_url=uri, source_type="RECORDED"), recorded_root=tmp_path)


@pytest.mark.asyncio
async def test_invalid_recorded_file_errors_without_reconnect(tmp_path):
    path = tmp_path / "broken.mp4"
    path.write_bytes(b"not a video")
    readers = []

    def reader_factory(request):
        reader = CameraReader(request, recorded_root=tmp_path)
        readers.append(reader)
        return reader

    manager = SessionManager(reader_factory, lambda _: FakeTracker())
    camera_id = uuid4()
    await manager.start(camera_id, StartSessionRequest(stream_url=path.as_uri(), source_type="RECORDED"))
    async with asyncio.timeout(3):
        while manager.status_now(camera_id).state is not SessionState.ERROR:
            await asyncio.sleep(.01)
    assert manager.status_now(camera_id).error_code == "AI_RECORDED_VIDEO_INVALID"
    assert manager.status_now(camera_id).frame_sequence == 0
    assert len(readers) == 1
    assert manager.active_worker_count == 0
    await manager.stop(camera_id)
