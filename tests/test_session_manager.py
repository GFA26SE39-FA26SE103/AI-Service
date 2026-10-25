from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.schemas import SessionState, StartSessionRequest
from app.session_manager import FrameNotReadyError, SessionCapacityError, SessionManager
from app.settings import Settings
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


def test_settings_bound_active_camera_sessions_to_current_host_ceiling():
    assert Settings().max_active_camera_sessions == 2
    with pytest.raises(ValidationError):
        Settings(max_active_camera_sessions=0)
    with pytest.raises(ValidationError):
        Settings(max_active_camera_sessions=3)


@pytest.mark.asyncio
async def test_start_processes_sequential_frames_and_publishes_latest_jpeg():
    factories = RecordingFactories.with_sequences([(True, frame(1)), (True, frame(2))])
    manager = SessionManager(factories.reader_factory, factories.tracker_factory)
    camera_id = uuid4()

    started = await manager.start(camera_id, request())
    await wait_until(lambda: manager.status_now(camera_id).frame_sequence >= 2)

    assert await manager.frame(camera_id, started.viewer_lease_id) == b"jpeg:2"
    assert factories.trackers[0].annotated_values[:2] == [1, 2]
    assert (await manager.status(camera_id, started.viewer_lease_id)).state is SessionState.LIVE
    await manager.force_stop(camera_id)


@pytest.mark.asyncio
async def test_second_start_for_same_camera_is_idempotent():
    factories = RecordingFactories.with_sequences([(True, frame(1))])
    manager = SessionManager(factories.reader_factory, factories.tracker_factory)
    camera_id = uuid4()

    first, second = await asyncio.gather(
        manager.start(camera_id, request()),
        manager.start(camera_id, request()),
    )
    await wait_until(lambda: manager.status_now(camera_id).state is SessionState.LIVE)

    assert first.camera_id == second.camera_id == camera_id
    assert len(factories.readers) == 1
    assert len(factories.trackers) == 1
    assert factories.supplied_passwords == ["secret"]
    await manager.force_stop(camera_id)


@pytest.mark.asyncio
async def test_stop_is_idempotent_and_releases_reader():
    factories = RecordingFactories.with_sequences([(True, frame(3))])
    manager = SessionManager(factories.reader_factory, factories.tracker_factory)
    camera_id = uuid4()
    await manager.start(camera_id, request())
    await wait_until(lambda: bool(factories.readers))

    first = await manager.force_stop(camera_id)
    second = await manager.force_stop(camera_id)

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

    status = manager.status_now(camera_id)
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
    await manager.force_stop(camera_id)
    restarted = await manager.start(camera_id, request("new-password"))
    await wait_until(lambda: manager.status_now(camera_id).frame_sequence >= 1)

    assert factories.supplied_passwords == ["old-password", "new-password"]
    assert len(factories.trackers) == 2
    assert factories.trackers[0] is not factories.trackers[1]
    assert await manager.frame(camera_id, restarted.viewer_lease_id) == b"jpeg:5"
    await manager.force_stop(camera_id)


@pytest.mark.asyncio
async def test_concurrent_start_and_stop_leave_no_orphan_worker():
    factories = RecordingFactories.with_sequences([(True, frame(6))])
    manager = SessionManager(factories.reader_factory, factories.tracker_factory)
    camera_id = uuid4()

    start_task = asyncio.create_task(manager.start(camera_id, request()))
    await asyncio.sleep(0)
    stop_task = asyncio.create_task(manager.force_stop(camera_id))
    await asyncio.gather(start_task, stop_task)

    assert manager.status_now(camera_id).state is SessionState.STOPPED
    assert manager.active_worker_count == 0
    assert all(reader.closed for reader in factories.readers)


@pytest.mark.asyncio
async def test_start_waits_for_inflight_stop_then_creates_new_worker():
    factories = RecordingFactories.with_sequences([(True, frame(7))], [(True, frame(8))])
    manager = SessionManager(factories.reader_factory, factories.tracker_factory)
    camera_id = uuid4()
    await manager.start(camera_id, request("old-password"))
    await wait_until(lambda: manager.status_now(camera_id).state is SessionState.LIVE)

    stop_task = asyncio.create_task(manager.force_stop(camera_id))
    restart_task = asyncio.create_task(manager.start(camera_id, request("new-password")))
    await asyncio.gather(stop_task, restart_task)
    await wait_until(lambda: manager.status_now(camera_id).state is SessionState.LIVE)

    assert factories.supplied_passwords == ["old-password", "new-password"]
    assert manager.active_worker_count == 1
    await manager.force_stop(camera_id)


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
    started = await manager.start(camera_id, request())
    await wait_until(lambda: manager.status_now(camera_id).state is SessionState.RECONNECTING)

    with pytest.raises(FrameNotReadyError):
        await manager.frame(camera_id, started.viewer_lease_id)
    await wait_until(lambda: manager.status_now(camera_id).state is SessionState.ERROR)
    with pytest.raises(FrameNotReadyError):
        await manager.frame(camera_id, started.viewer_lease_id)


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
    await manager.force_stop(camera_id)


@pytest.mark.asyncio
async def test_only_one_gpu_session_can_run_at_a_time():
    factories = RecordingFactories.with_sequences([(True, frame(13))], [(True, frame(14))])
    manager = SessionManager(
        factories.reader_factory,
        factories.tracker_factory,
        max_active_camera_sessions=1,
    )
    first, second = uuid4(), uuid4()
    await manager.start(first, request())
    await wait_until(lambda: manager.status_now(first).state is SessionState.LIVE)

    with pytest.raises(SessionCapacityError):
        await manager.start(second, request())

    await manager.force_stop(first)
    await manager.start(second, request())
    await manager.force_stop(second)


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


@pytest.mark.asyncio
async def test_two_viewers_share_one_camera_worker_and_have_distinct_leases():
    factories = RecordingFactories.with_sequences([(True, frame(21))])
    manager = SessionManager(factories.reader_factory, factories.tracker_factory)
    camera_id = uuid4()

    first = await manager.start(camera_id, request())
    second = await manager.start(camera_id, request())
    await wait_until(lambda: manager.status_now(camera_id).state is SessionState.LIVE)

    assert first.viewer_lease_id is not None
    assert second.viewer_lease_id is not None
    assert first.viewer_lease_id != second.viewer_lease_id
    assert first.session_id == second.session_id
    assert len(factories.readers) == len(factories.trackers) == 1
    await manager.shutdown()


@pytest.mark.asyncio
async def test_releasing_one_viewer_keeps_the_other_live():
    factories = RecordingFactories.with_sequences([(True, frame(22))])
    manager = SessionManager(factories.reader_factory, factories.tracker_factory)
    camera_id = uuid4()
    first = await manager.start(camera_id, request())
    second = await manager.start(camera_id, request())
    await wait_until(lambda: manager.status_now(camera_id).state is SessionState.LIVE)

    await manager.release_viewer(camera_id, first.viewer_lease_id)

    assert (await manager.status(camera_id, second.viewer_lease_id)).state is SessionState.LIVE
    assert await manager.frame(camera_id, second.viewer_lease_id) == b"jpeg:22"
    assert manager.active_worker_count == 1
    await manager.shutdown()


@pytest.mark.asyncio
async def test_two_distinct_cameras_run_with_isolated_trackers():
    factories = RecordingFactories.with_sequences(
        [(True, frame(31))],
        [(True, frame(32))],
    )
    manager = SessionManager(
        factories.reader_factory,
        factories.tracker_factory,
        max_active_camera_sessions=2,
    )
    first_camera, second_camera = uuid4(), uuid4()

    first = await manager.start(first_camera, request())
    second = await manager.start(second_camera, request())
    await wait_until(lambda: manager.status_now(first_camera).state is SessionState.LIVE)
    await wait_until(lambda: manager.status_now(second_camera).state is SessionState.LIVE)

    assert len(factories.readers) == len(factories.trackers) == 2
    assert factories.trackers[0] is not factories.trackers[1]
    assert await manager.frame(first_camera, first.viewer_lease_id) == b"jpeg:31"
    assert await manager.frame(second_camera, second.viewer_lease_id) == b"jpeg:32"
    await manager.shutdown()


@pytest.mark.asyncio
async def test_same_numeric_track_id_on_two_cameras_remains_camera_local():
    factories = RecordingFactories.with_sequences(
        [(True, frame(34))],
        [(True, frame(34))],
    )
    manager = SessionManager(
        factories.reader_factory,
        factories.tracker_factory,
        max_active_camera_sessions=2,
    )
    first_camera, second_camera = uuid4(), uuid4()

    first = await manager.start(first_camera, request())
    second = await manager.start(second_camera, request())
    await wait_until(lambda: manager.status_now(first_camera).frame_sequence > 0)
    await wait_until(lambda: manager.status_now(second_camera).frame_sequence > 0)

    assert await manager.frame(first_camera, first.viewer_lease_id) == b"jpeg:34"
    assert await manager.frame(second_camera, second.viewer_lease_id) == b"jpeg:34"
    assert factories.trackers[0] is not factories.trackers[1]
    await manager.shutdown()


@pytest.mark.asyncio
async def test_third_camera_is_rejected_without_stopping_the_first_two():
    factories = RecordingFactories.with_sequences(
        [(True, frame(41))],
        [(True, frame(42))],
        [(True, frame(43))],
    )
    manager = SessionManager(
        factories.reader_factory,
        factories.tracker_factory,
        max_active_camera_sessions=2,
    )
    first_camera, second_camera, third_camera = uuid4(), uuid4(), uuid4()
    first = await manager.start(first_camera, request())
    second = await manager.start(second_camera, request())
    await wait_until(lambda: manager.status_now(first_camera).state is SessionState.LIVE)
    await wait_until(lambda: manager.status_now(second_camera).state is SessionState.LIVE)

    with pytest.raises(SessionCapacityError, match="AI_SESSION_CAPACITY"):
        await manager.start(third_camera, request())

    assert (await manager.status(first_camera, first.viewer_lease_id)).state is SessionState.LIVE
    assert (await manager.status(second_camera, second.viewer_lease_id)).state is SessionState.LIVE
    assert manager.active_worker_count == 2
    await manager.shutdown()


@pytest.mark.asyncio
async def test_final_release_stops_preview_owned_worker():
    factories = RecordingFactories.with_sequences([(True, frame(61))])
    manager = SessionManager(factories.reader_factory, factories.tracker_factory)
    camera_id = uuid4()
    started = await manager.start(camera_id, request())
    await wait_until(lambda: manager.status_now(camera_id).state is SessionState.LIVE)

    released = await manager.release_viewer(camera_id, started.viewer_lease_id)

    assert released.state is SessionState.STOPPED
    assert manager.active_worker_count == 0
    assert factories.readers[0].closed


@pytest.mark.asyncio
async def test_abandoned_final_lease_expires_and_releases_capacity():
    factories = RecordingFactories.with_sequences([(True, frame(62))], [(True, frame(63))])
    manager = SessionManager(
        factories.reader_factory,
        factories.tracker_factory,
        session_idle_timeout_seconds=.03,
        max_active_camera_sessions=1,
    )
    first_camera, second_camera = uuid4(), uuid4()
    await manager.start(first_camera, request())

    await wait_until(lambda: manager.status_now(first_camera).state is SessionState.STOPPED)
    second = await manager.start(second_camera, request())

    assert second.viewer_lease_id is not None
    await manager.shutdown()


@pytest.mark.asyncio
async def test_completed_or_error_session_lease_is_pruned_by_sweeper():
    factories = RecordingFactories.with_sequences([(False, None)], [(True, frame(64))])
    manager = SessionManager(
        factories.reader_factory,
        factories.tracker_factory,
        reconnect_attempts=0,
        session_idle_timeout_seconds=.03,
        max_active_camera_sessions=1,
    )
    failed_camera, next_camera = uuid4(), uuid4()
    failed = await manager.start(failed_camera, request())
    await wait_until(lambda: manager.status_now(failed_camera).state is SessionState.ERROR)

    await wait_until(lambda: manager.status_now(failed_camera).state is SessionState.STOPPED)
    with pytest.raises(RuntimeError, match="AI_VIEWER_LEASE_EXPIRED"):
        await manager.status(failed_camera, failed.viewer_lease_id)
    assert (await manager.start(next_camera, request())).viewer_lease_id is not None
    await manager.shutdown()


@pytest.mark.asyncio
async def test_old_generation_lease_cannot_access_restarted_session():
    factories = RecordingFactories.with_sequences([(True, frame(65))], [(True, frame(66))])
    manager = SessionManager(factories.reader_factory, factories.tracker_factory)
    camera_id = uuid4()
    old = await manager.start(camera_id, request())
    await wait_until(lambda: manager.status_now(camera_id).state is SessionState.LIVE)
    await manager.force_stop(camera_id)
    fresh = await manager.start(camera_id, request())

    with pytest.raises(RuntimeError, match="AI_VIEWER_LEASE_EXPIRED"):
        await manager.status(camera_id, old.viewer_lease_id)
    assert (await manager.status(camera_id, fresh.viewer_lease_id)).session_id == fresh.session_id
    await manager.shutdown()


@pytest.mark.asyncio
async def test_unknown_release_is_idempotent():
    factories = RecordingFactories.with_sequences([(True, frame(67))])
    manager = SessionManager(factories.reader_factory, factories.tracker_factory)
    camera_id = uuid4()
    started = await manager.start(camera_id, request())
    await wait_until(lambda: manager.status_now(camera_id).state is SessionState.LIVE)

    status = await manager.release_viewer(camera_id, uuid4())

    assert status.state is SessionState.LIVE
    assert (await manager.status(camera_id, started.viewer_lease_id)).state is SessionState.LIVE
    await manager.shutdown()


@pytest.mark.asyncio
async def test_configuration_mismatch_does_not_restart_shared_worker():
    factories = RecordingFactories.with_sequences([(True, frame(68))])
    manager = SessionManager(factories.reader_factory, factories.tracker_factory)
    camera_id = uuid4()
    started = await manager.start(camera_id, request("first-secret"))
    await wait_until(lambda: manager.status_now(camera_id).state is SessionState.LIVE)

    with pytest.raises(RuntimeError, match="AI_PREVIEW_CONFIGURATION_CONFLICT"):
        await manager.start(camera_id, request("other-secret"))

    assert (await manager.status(camera_id, started.viewer_lease_id)).session_id == started.session_id
    assert len(factories.readers) == len(factories.trackers) == 1
    await manager.shutdown()


@pytest.mark.asyncio
async def test_failure_of_camera_one_does_not_change_camera_two():
    factories = RecordingFactories.with_sequences([(False, None)], [(True, frame(69))])
    manager = SessionManager(
        factories.reader_factory,
        factories.tracker_factory,
        reconnect_attempts=0,
        max_active_camera_sessions=2,
    )
    failed_camera, live_camera = uuid4(), uuid4()
    failed = await manager.start(failed_camera, request())
    live = await manager.start(live_camera, request())
    await wait_until(lambda: manager.status_now(failed_camera).state is SessionState.ERROR)
    await wait_until(lambda: manager.status_now(live_camera).state is SessionState.LIVE)

    assert (await manager.status(failed_camera, failed.viewer_lease_id)).state is SessionState.ERROR
    assert await manager.frame(live_camera, live.viewer_lease_id) == b"jpeg:69"
    await manager.shutdown()


@pytest.mark.asyncio
async def test_concurrent_final_release_and_attach_leaves_no_orphan_worker():
    factories = RecordingFactories.with_sequences([(True, frame(70))], [(True, frame(71))])
    manager = SessionManager(factories.reader_factory, factories.tracker_factory)
    camera_id = uuid4()
    first = await manager.start(camera_id, request())
    await wait_until(lambda: manager.status_now(camera_id).state is SessionState.LIVE)

    release = asyncio.create_task(manager.release_viewer(camera_id, first.viewer_lease_id))
    attach = asyncio.create_task(manager.start(camera_id, request()))
    _, attached = await asyncio.gather(release, attach)
    await wait_until(lambda: manager.status_now(camera_id).state is SessionState.LIVE)

    assert (await manager.status(camera_id, attached.viewer_lease_id)).state is SessionState.LIVE
    assert manager.active_worker_count == 1
    assert len([reader for reader in factories.readers if not reader.closed]) == 1
    await manager.shutdown()
