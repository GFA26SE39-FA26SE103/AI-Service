from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from app.schemas import SessionState, StartSessionRequest
from app.session_manager import FrameNotReadyError, SessionCapacityError, SessionManager
from tests.fakes import RecordingFactories, frame


async def wait_until(predicate, timeout: float = 1.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.005)


def request(password: str = "secret") -> StartSessionRequest:
    return StartSessionRequest(
        stream_url="http://camera.local:8080/video",
        username="viewer",
        password=password,
    )


@pytest.mark.asyncio
async def test_start_processes_sequential_frames_and_publishes_latest_jpeg():
    factories = RecordingFactories.with_sequences([(True, frame(1)), (True, frame(2))])
    manager = SessionManager(factories.reader_factory, factories.tracker_factory)
    camera_id = uuid4()

    await manager.start(camera_id, request())
    await wait_until(lambda: manager.status_now(camera_id).frame_sequence >= 2)

    assert await manager.frame(camera_id) == b"jpeg:2"
    assert factories.trackers[0].annotated_values[:2] == [1, 2]
    assert (await manager.status(camera_id)).state is SessionState.LIVE
    await manager.stop(camera_id)


@pytest.mark.asyncio
async def test_second_start_for_same_camera_is_idempotent():
    factories = RecordingFactories.with_sequences([(True, frame(1))])
    manager = SessionManager(factories.reader_factory, factories.tracker_factory)
    camera_id = uuid4()

    first, second = await asyncio.gather(
        manager.start(camera_id, request()),
        manager.start(camera_id, request("different-secret")),
    )
    await wait_until(lambda: manager.status_now(camera_id).state is SessionState.LIVE)

    assert first.camera_id == second.camera_id == camera_id
    assert len(factories.readers) == 1
    assert len(factories.trackers) == 1
    assert factories.supplied_passwords == ["secret"]
    await manager.stop(camera_id)


@pytest.mark.asyncio
async def test_stop_is_idempotent_and_releases_reader():
    factories = RecordingFactories.with_sequences([(True, frame(3))])
    manager = SessionManager(factories.reader_factory, factories.tracker_factory)
    camera_id = uuid4()
    await manager.start(camera_id, request())
    await wait_until(lambda: bool(factories.readers))

    first = await manager.stop(camera_id)
    second = await manager.stop(camera_id)

    assert first.state is second.state is SessionState.STOPPED
    assert factories.readers[0].closed
    assert factories.readers[0].close_calls == 1
    assert manager.active_worker_count == 0


@pytest.mark.asyncio
async def test_disconnect_transitions_to_reconnecting_then_error():
    factories = RecordingFactories.with_sequences([(False, None)], [(False, None)])
    manager = SessionManager(
        factories.reader_factory,
        factories.tracker_factory,
        reconnect_attempts=1,
        reconnect_delay_seconds=0.03,
    )
    camera_id = uuid4()

    await manager.start(camera_id, request())
    await wait_until(lambda: manager.status_now(camera_id).state is SessionState.RECONNECTING)
    await wait_until(lambda: manager.status_now(camera_id).state is SessionState.ERROR)

    status = await manager.status(camera_id)
    assert status.error_code == "CAMERA_STREAM_UNAVAILABLE"
    assert all(reader.closed for reader in factories.readers)
    assert manager.active_worker_count == 0


@pytest.mark.asyncio
async def test_restart_does_not_reuse_previous_tracker_state_or_credentials():
    factories = RecordingFactories.with_sequences(
        [(True, frame(4))],
        [(True, frame(5))],
    )
    manager = SessionManager(factories.reader_factory, factories.tracker_factory)
    camera_id = uuid4()

    await manager.start(camera_id, request("old-password"))
    await wait_until(lambda: manager.status_now(camera_id).frame_sequence >= 1)
    await manager.stop(camera_id)
    await manager.start(camera_id, request("new-password"))
    await wait_until(lambda: manager.status_now(camera_id).frame_sequence >= 1)

    assert factories.supplied_passwords == ["old-password", "new-password"]
    assert len(factories.trackers) == 2
    assert factories.trackers[0] is not factories.trackers[1]
    assert await manager.frame(camera_id) == b"jpeg:5"
    await manager.stop(camera_id)


@pytest.mark.asyncio
async def test_concurrent_start_and_stop_leave_no_orphan_worker():
    factories = RecordingFactories.with_sequences([(True, frame(6))])
    manager = SessionManager(factories.reader_factory, factories.tracker_factory)
    camera_id = uuid4()

    start_task = asyncio.create_task(manager.start(camera_id, request()))
    await asyncio.sleep(0)
    stop_task = asyncio.create_task(manager.stop(camera_id))
    await asyncio.gather(start_task, stop_task)

    assert (await manager.status(camera_id)).state is SessionState.STOPPED
    assert manager.active_worker_count == 0
    assert all(reader.closed for reader in factories.readers)


@pytest.mark.asyncio
async def test_start_waits_for_inflight_stop_then_creates_new_worker():
    factories = RecordingFactories.with_sequences([(True, frame(7))], [(True, frame(8))])
    manager = SessionManager(factories.reader_factory, factories.tracker_factory)
    camera_id = uuid4()
    await manager.start(camera_id, request("old-password"))
    await wait_until(lambda: manager.status_now(camera_id).state is SessionState.LIVE)

    stop_task = asyncio.create_task(manager.stop(camera_id))
    restart_task = asyncio.create_task(manager.start(camera_id, request("new-password")))
    await asyncio.gather(stop_task, restart_task)
    await wait_until(lambda: manager.status_now(camera_id).state is SessionState.LIVE)

    assert factories.supplied_passwords == ["old-password", "new-password"]
    assert manager.active_worker_count == 1
    await manager.stop(camera_id)


@pytest.mark.asyncio
async def test_reconnecting_and_error_never_serve_stale_frame():
    factories = RecordingFactories.with_sequences([(True, frame(9)), (False, None)], [(False, None)])
    manager = SessionManager(
        factories.reader_factory,
        factories.tracker_factory,
        reconnect_attempts=1,
        reconnect_delay_seconds=.05,
    )
    camera_id = uuid4()
    await manager.start(camera_id, request())
    await wait_until(lambda: manager.status_now(camera_id).state is SessionState.RECONNECTING)

    with pytest.raises(FrameNotReadyError):
        await manager.frame(camera_id)
    await wait_until(lambda: manager.status_now(camera_id).state is SessionState.ERROR)
    with pytest.raises(FrameNotReadyError):
        await manager.frame(camera_id)


@pytest.mark.asyncio
async def test_successful_frame_resets_reconnect_budget():
    factories = RecordingFactories.with_sequences(
        [(True, frame(10)), (False, None)],
        [(True, frame(11)), (False, None)],
        [(True, frame(12))],
    )
    manager = SessionManager(
        factories.reader_factory,
        factories.tracker_factory,
        reconnect_attempts=1,
        reconnect_delay_seconds=0,
    )
    camera_id = uuid4()
    await manager.start(camera_id, request())

    await wait_until(lambda: manager.status_now(camera_id).frame_sequence >= 3)

    assert manager.status_now(camera_id).state is SessionState.LIVE
    await manager.stop(camera_id)


@pytest.mark.asyncio
async def test_only_one_gpu_session_can_run_at_a_time():
    factories = RecordingFactories.with_sequences([(True, frame(13))], [(True, frame(14))])
    manager = SessionManager(factories.reader_factory, factories.tracker_factory)
    first, second = uuid4(), uuid4()
    await manager.start(first, request())
    await wait_until(lambda: manager.status_now(first).state is SessionState.LIVE)

    with pytest.raises(SessionCapacityError):
        await manager.start(second, request())

    await manager.stop(first)
    await manager.start(second, request())
    await manager.stop(second)


@pytest.mark.asyncio
async def test_idle_session_self_stops_when_client_polling_disappears():
    factories = RecordingFactories.with_sequences([(True, frame(15))])
    manager = SessionManager(
        factories.reader_factory,
        factories.tracker_factory,
        session_idle_timeout_seconds=.03,
    )
    camera_id = uuid4()
    await manager.start(camera_id, request())

    await wait_until(lambda: manager.status_now(camera_id).state is SessionState.STOPPED)

    assert manager.active_worker_count == 0
    assert factories.readers[0].closed
