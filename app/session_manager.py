from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol
from uuid import UUID,uuid4

import numpy as np

from app.schemas import SessionState, SessionStatusResponse, StartSessionRequest
from app.monitoring_schemas import StartMonitoringRequest,MeasurementBatch,MeasurementPage
from app.measurement_buffer import MeasurementBuffer
from app.source_clock import SourceClock


class Reader(Protocol):
    def read(self) -> tuple[bool, np.ndarray | None]: ...
    def close(self) -> None: ...


class Tracker(Protocol):
    def annotate(self, frame: np.ndarray) -> np.ndarray: ...
    def encode_jpeg(self, frame: np.ndarray) -> bytes: ...


class SessionNotRunningError(RuntimeError):
    pass


class FrameNotReadyError(RuntimeError):
    pass


class SessionCapacityError(RuntimeError):
    pass

class SessionOwnershipError(RuntimeError):
    pass


@dataclass
class _Session:
    status: SessionStatusResponse
    cancel: asyncio.Event
    task: asyncio.Task[None] | None = None
    reader: Reader | None = None
    latest_jpeg: bytes | None = None
    last_access: float = 0.0
    monitoring_request: StartMonitoringRequest | None = None
    pending_request: StartMonitoringRequest | None = None
    owner_access: float = 0.0
    buffer: MeasurementBuffer | None = None
    continuity_id: UUID | None = None
    frame_condition: asyncio.Condition = field(default_factory=asyncio.Condition)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SessionManager:
    def __init__(
        self,
        reader_factory: Callable[[StartSessionRequest], Reader],
        tracker_factory: Callable[[StartSessionRequest], Tracker],
        *,
        reconnect_attempts: int = 5,
        reconnect_delay_seconds: float = 1.0,
        session_idle_timeout_seconds: float = 30.0,
        monitoring_tracker_factory: Callable | None = None,
        max_observation_gap_ms: int = 2000,
    ) -> None:
        self._reader_factory = reader_factory
        self._tracker_factory = tracker_factory
        self._monitoring_tracker_factory = monitoring_tracker_factory
        self._max_observation_gap_ms=max_observation_gap_ms
        self._reconnect_attempts = max(0, reconnect_attempts)
        self._reconnect_delay_seconds = max(0, reconnect_delay_seconds)
        self._session_idle_timeout_seconds = max(0.01, session_idle_timeout_seconds)
        self._sessions: dict[UUID, _Session] = {}
        self._locks: dict[UUID, asyncio.Lock] = {}
        self._capacity_lock = asyncio.Lock()

    @property
    def active_worker_count(self) -> int:
        return sum(1 for session in self._sessions.values() if session.task and not session.task.done())

    def _lock_for(self, camera_id: UUID) -> asyncio.Lock:
        return self._locks.setdefault(camera_id, asyncio.Lock())

    async def start(self, camera_id: UUID, request: StartSessionRequest) -> SessionStatusResponse:
        async with self._capacity_lock:
            async with self._lock_for(camera_id):
                existing = self._sessions.get(camera_id)
                if existing and existing.monitoring_request:
                    # Viewer attach never changes source, confidence or owner lease,
                    # including a completed recorded session.
                    return existing.status.model_copy(deep=True)
                if existing and existing.task and not existing.task.done():
                    existing.last_access = asyncio.get_running_loop().time()
                    return existing.status.model_copy(deep=True)

                if any(
                    other_id != camera_id
                    and session.task
                    and not session.task.done()
                    for other_id, session in self._sessions.items()
                ):
                    raise SessionCapacityError("AI_SESSION_CAPACITY")

                now = _utc_now()
                session = _Session(
                    status=SessionStatusResponse(
                        camera_id=camera_id,
                        state=SessionState.STARTING,
                        started_at=now,
                        updated_at=now,
                        session_id=uuid4(),
                    ),
                    cancel=asyncio.Event(),
                    last_access=asyncio.get_running_loop().time(),
                )
                self._sessions[camera_id] = session
                session.task = asyncio.create_task(
                    self._run(camera_id, session, request),
                    name=f"ai-preview-{camera_id}",
                )
                return session.status.model_copy(deep=True)

    async def start_monitoring(self,camera_id: UUID,request: StartMonitoringRequest) -> SessionStatusResponse:
        async with self._capacity_lock:
            async with self._lock_for(camera_id):
                existing=self._sessions.get(camera_id)
                if existing and existing.monitoring_request:
                    expired=asyncio.get_running_loop().time()-existing.owner_access>=self._session_idle_timeout_seconds
                    if not expired and existing.monitoring_request.owner_id!=request.owner_id:
                        raise SessionOwnershipError("AI_SESSION_OWNER_MISMATCH")
                    if not expired and existing.status.state not in (SessionState.STOPPED,SessionState.ERROR):
                        prior=existing.pending_request or existing.monitoring_request
                        fields=("stream_url","source_type","username","password","model","tracker","device","half","classes")
                        if any(getattr(prior,f)!=getattr(request,f) for f in fields): raise SessionOwnershipError("AI_SESSION_MONITORING_OWNED")
                        existing.owner_access=asyncio.get_running_loop().time()
                        if prior.configuration_fingerprint!=request.configuration_fingerprint:
                            if existing.status.state == SessionState.COMPLETED:
                                # There is no next frame to apply a queued request. A new
                                # version at EOF invalidates old metrics but never replays.
                                existing.monitoring_request=request.model_copy(deep=True)
                                existing.pending_request=None
                                existing.continuity_id=uuid4()
                                if existing.buffer is not None: existing.buffer.clear()
                                existing.status.configuration_fingerprint=request.configuration_fingerprint
                                existing.status.annotation_context=None
                            else:
                                existing.pending_request=request.model_copy(deep=True)
                        return existing.status.model_copy(deep=True)
                if any(other!=camera_id and s.task and not s.task.done() for other,s in self._sessions.items()):
                    raise SessionCapacityError("AI_SESSION_CAPACITY")
                if existing: await self._stop_session(existing)
                now=_utc_now(); session_id=uuid4()
                session=_Session(status=SessionStatusResponse(camera_id=camera_id,state=SessionState.STARTING,started_at=now,updated_at=now,
                    session_id=session_id,purpose="MONITORING",configuration_fingerprint=request.configuration_fingerprint),cancel=asyncio.Event(),
                    monitoring_request=request.model_copy(deep=True),owner_access=asyncio.get_running_loop().time(),buffer=MeasurementBuffer(session_id),continuity_id=uuid4())
                self._sessions[camera_id]=session
                session.task=asyncio.create_task(self._run(camera_id,session,request),name=f"ai-monitoring-{camera_id}")
                return session.status.model_copy(deep=True)

    async def measurements(self,camera_id: UUID,owner_id: UUID,after_session_id: UUID|None,after_sequence: int,limit: int) -> MeasurementPage:
        async with self._lock_for(camera_id):
            session=self._sessions.get(camera_id)
            if not session or not session.monitoring_request or session.buffer is None: raise SessionNotRunningError("AI_PREVIEW_NOT_RUNNING")
            if session.monitoring_request.owner_id!=owner_id: raise SessionOwnershipError("AI_SESSION_OWNER_MISMATCH")
            if self._is_idle(session):
                await self._stop_session(session)
                raise SessionNotRunningError("AI_PREVIEW_NOT_RUNNING")
            session.owner_access=asyncio.get_running_loop().time()
            return session.buffer.read(after_sequence,limit,session.status.state,wrong_session=after_session_id is not None and after_session_id!=session.status.session_id,error_code=session.status.error_code)

    async def stop_monitoring(self,camera_id: UUID,owner_id: UUID) -> SessionStatusResponse:
        async with self._lock_for(camera_id):
            session=self._sessions.get(camera_id)
            if not session: return self.status_now(camera_id).model_copy(deep=True)
            if session.monitoring_request is None or session.monitoring_request.owner_id!=owner_id: raise SessionOwnershipError("AI_SESSION_OWNER_MISMATCH")
            await self._stop_session(session)
            session.monitoring_request=None; session.pending_request=None; session.buffer=None
            session.status.purpose="PREVIEW"; session.status.configuration_fingerprint=None; session.status.annotation_context=None
            return session.status.model_copy(deep=True)

    async def status(self, camera_id: UUID) -> SessionStatusResponse:
        session = self._sessions.get(camera_id)
        if session:
            session.last_access = asyncio.get_running_loop().time()
        return self.status_now(camera_id).model_copy(deep=True)

    def status_now(self, camera_id: UUID) -> SessionStatusResponse:
        session = self._sessions.get(camera_id)
        if session:
            return session.status
        return SessionStatusResponse(
            camera_id=camera_id,
            state=SessionState.STOPPED,
            updated_at=_utc_now(),
        )

    async def frame(self, camera_id: UUID) -> bytes:
        session = self._sessions.get(camera_id)
        if not session or session.status.state is SessionState.STOPPED:
            raise SessionNotRunningError("AI_PREVIEW_NOT_RUNNING")
        session.last_access = asyncio.get_running_loop().time()
        if session.status.state not in (SessionState.LIVE, SessionState.COMPLETED) or session.latest_jpeg is None:
            raise FrameNotReadyError("AI_FRAME_NOT_READY")
        return session.latest_jpeg

    async def next_frame(self, camera_id: UUID, after_sequence: int, after_session_id: UUID | None) -> tuple[bytes, int, UUID] | None:
        session = self._sessions.get(camera_id)
        if not session or session.status.state is SessionState.STOPPED:
            raise SessionNotRunningError("AI_PREVIEW_NOT_RUNNING")
        session.last_access = asyncio.get_running_loop().time()
        async with session.frame_condition:
            try:
                async with asyncio.timeout(1.0):
                    while True:
                        if session.status.state in (SessionState.STOPPED, SessionState.ERROR):
                            return None
                        if session.latest_jpeg is not None and (
                            session.status.session_id != after_session_id
                            or session.status.frame_sequence > after_sequence
                        ):
                            return session.latest_jpeg, session.status.frame_sequence, session.status.session_id
                        if session.status.state is SessionState.COMPLETED:
                            return None
                        await session.frame_condition.wait()
            except TimeoutError:
                return None

    async def stop(self, camera_id: UUID) -> SessionStatusResponse:
        async with self._lock_for(camera_id):
            session = self._sessions.get(camera_id)
            if not session:
                return self.status_now(camera_id).model_copy(deep=True)
            if session.monitoring_request: raise SessionOwnershipError("AI_SESSION_MONITORING_OWNED")
            await self._stop_session(session)
            return session.status.model_copy(deep=True)

    async def _stop_session(self,session: _Session):
        session.cancel.set()
        task=session.task
        if task and task is not asyncio.current_task(): await task
        self._set_state(session,SessionState.STOPPED)
        session.latest_jpeg=None

    async def shutdown(self):
        for session in list(self._sessions.values()): await self._stop_session(session)

    async def _run(
        self,
        camera_id: UUID,
        session: _Session,
        request: StartSessionRequest,
    ) -> None:
        tracker: Tracker | None = None
        try:
            if session.monitoring_request:
                if self._monitoring_tracker_factory is None: raise ValueError("AI_MONITORING_UNAVAILABLE")
                tracker=await asyncio.to_thread(self._monitoring_tracker_factory,request)
            else: tracker = await asyncio.to_thread(self._tracker_factory, request)
            clock=SourceClock(request.source_type,self._max_observation_gap_ms)
            attempt = 0
            while not session.cancel.is_set():
                if self._is_idle(session):
                    session.cancel.set()
                    break
                reader = await asyncio.to_thread(self._reader_factory, request)
                session.reader = reader
                disconnected = False
                try:
                    while not session.cancel.is_set():
                        if self._is_idle(session):
                            session.cancel.set()
                            break
                        frame_started = asyncio.get_running_loop().time()
                        # Only the worker replaces tracker contexts, between frames.
                        # Requests merely queue a new snapshot; no concurrent tracker mutation.
                        if session.pending_request:
                            request=session.pending_request; session.pending_request=None
                            await asyncio.to_thread(tracker.reconfigure,request.zones)
                            session.monitoring_request=request
                            session.status.configuration_fingerprint=request.configuration_fingerprint
                            session.continuity_id=uuid4()
                            session.buffer.clear()
                        ok, frame = await asyncio.to_thread(reader.read)
                        if not ok or frame is None:
                            if request.source_type == "RECORDED":
                                if session.status.frame_sequence:
                                    self._set_state(session, SessionState.COMPLETED)
                                else:
                                    session.status.error_code = "AI_RECORDED_VIDEO_INVALID"
                                    self._set_state(session, SessionState.ERROR)
                                return
                            disconnected = True
                            break
                        if session.monitoring_request:
                            source_time=clock.observe(getattr(reader,"position_ms",None),getattr(reader,"fps",None),asyncio.get_running_loop().time())
                            if source_time.continuity_broken:
                                session.continuity_id=uuid4()
                                await asyncio.to_thread(tracker.reset_tracking)
                            result=await asyncio.to_thread(tracker.process,frame,source_time.elapsed_ms,session.continuity_id)
                            annotated=result.annotated_frame
                            session.status.annotation_context=result.annotation_context
                        else: annotated = await asyncio.to_thread(tracker.annotate, frame)
                        encoded = await asyncio.to_thread(tracker.encode_jpeg, annotated)
                        session.latest_jpeg = encoded
                        session.status.frame_sequence += 1
                        if session.monitoring_request:
                            session.buffer.append(MeasurementBatch(session_id=session.status.session_id,continuity_id=session.continuity_id,
                                sequence=session.status.frame_sequence,captured_at=_utc_now(),source_elapsed_ms=source_time.elapsed_ms,
                                configuration_fingerprint=request.configuration_fingerprint,zones=result.measurements))
                        self._set_state(session, SessionState.LIVE)
                        async with session.frame_condition:
                            session.frame_condition.notify_all()
                        attempt = 0
                        if request.source_type == "RECORDED":
                            delay = max(0, getattr(reader, "frame_interval", 1 / 25) - (asyncio.get_running_loop().time() - frame_started))
                            try:
                                await asyncio.wait_for(session.cancel.wait(), timeout=delay)
                            except TimeoutError:
                                pass
                        else:
                            await asyncio.sleep(0)
                finally:
                    self._close_reader(session, reader)

                if session.cancel.is_set():
                    break
                if not disconnected:
                    continue
                if attempt >= self._reconnect_attempts:
                    session.status.error_code = "CAMERA_STREAM_UNAVAILABLE"
                    session.latest_jpeg = None
                    self._set_state(session, SessionState.ERROR)
                    return

                attempt += 1
                if session.monitoring_request:
                    session.continuity_id=uuid4()
                    await asyncio.to_thread(tracker.reset_tracking)
                session.latest_jpeg = None
                self._set_state(session, SessionState.RECONNECTING)
                try:
                    await asyncio.wait_for(
                        session.cancel.wait(), timeout=self._reconnect_delay_seconds
                    )
                except TimeoutError:
                    pass
        except asyncio.CancelledError:
            raise
        except Exception as error:
            code = str(error)
            session.status.error_code = code if code in ("AI_RECORDED_VIDEO_INVALID", "AI_SOURCE_INVALID","AI_SOURCE_TIME_INVALID","AI_TRACKER_UNSUPPORTED","AI_MONITORING_UNAVAILABLE") else "AI_PREVIEW_FAILED"
            self._set_state(session, SessionState.ERROR)
        finally:
            self._close_reader(session, session.reader)
            if session.cancel.is_set():
                self._set_state(session, SessionState.STOPPED)
            session.task = None

    def _is_idle(self, session: _Session) -> bool:
        return (
            asyncio.get_running_loop().time() - (session.owner_access if session.monitoring_request else session.last_access)
            >= self._session_idle_timeout_seconds
        )

    @staticmethod
    def _set_state(session: _Session, state: SessionState) -> None:
        session.status.state = state
        session.status.updated_at = _utc_now()
        if state not in (SessionState.ERROR, SessionState.RECONNECTING):
            session.status.error_code = None

    @staticmethod
    def _close_reader(session: _Session, reader: Reader | None) -> None:
        if reader is None or session.reader is not reader:
            return
        session.reader = None
        reader.close()
