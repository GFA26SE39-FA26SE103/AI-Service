from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol
from uuid import UUID

import numpy as np

from app.schemas import SessionState, SessionStatusResponse, StartSessionRequest


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


@dataclass
class _Session:
    status: SessionStatusResponse
    cancel: asyncio.Event
    task: asyncio.Task[None] | None = None
    reader: Reader | None = None
    latest_jpeg: bytes | None = None


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
    ) -> None:
        self._reader_factory = reader_factory
        self._tracker_factory = tracker_factory
        self._reconnect_attempts = max(0, reconnect_attempts)
        self._reconnect_delay_seconds = max(0, reconnect_delay_seconds)
        self._sessions: dict[UUID, _Session] = {}
        self._locks: dict[UUID, asyncio.Lock] = {}

    @property
    def active_worker_count(self) -> int:
        return sum(1 for session in self._sessions.values() if session.task and not session.task.done())

    def _lock_for(self, camera_id: UUID) -> asyncio.Lock:
        return self._locks.setdefault(camera_id, asyncio.Lock())

    async def start(self, camera_id: UUID, request: StartSessionRequest) -> SessionStatusResponse:
        async with self._lock_for(camera_id):
            existing = self._sessions.get(camera_id)
            if existing and existing.task and not existing.task.done():
                return existing.status.model_copy(deep=True)

            now = _utc_now()
            session = _Session(
                status=SessionStatusResponse(
                    camera_id=camera_id,
                    state=SessionState.STARTING,
                    started_at=now,
                    updated_at=now,
                ),
                cancel=asyncio.Event(),
            )
            self._sessions[camera_id] = session
            session.task = asyncio.create_task(
                self._run(camera_id, session, request),
                name=f"ai-preview-{camera_id}",
            )
            return session.status.model_copy(deep=True)

    async def status(self, camera_id: UUID) -> SessionStatusResponse:
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
        if session.latest_jpeg is None:
            raise FrameNotReadyError("AI_FRAME_NOT_READY")
        return session.latest_jpeg

    async def stop(self, camera_id: UUID) -> SessionStatusResponse:
        task: asyncio.Task[None] | None = None
        async with self._lock_for(camera_id):
            session = self._sessions.get(camera_id)
            if not session:
                return self.status_now(camera_id).model_copy(deep=True)
            session.cancel.set()
            self._close_reader(session, session.reader)
            task = session.task

        if task and task is not asyncio.current_task():
            await task

        async with self._lock_for(camera_id):
            session = self._sessions[camera_id]
            self._set_state(session, SessionState.STOPPED)
            session.latest_jpeg = None
            return session.status.model_copy(deep=True)

    async def _run(
        self,
        camera_id: UUID,
        session: _Session,
        request: StartSessionRequest,
    ) -> None:
        del camera_id
        tracker: Tracker | None = None
        try:
            tracker = await asyncio.to_thread(self._tracker_factory, request)
            attempt = 0
            while not session.cancel.is_set():
                reader = await asyncio.to_thread(self._reader_factory, request)
                session.reader = reader
                disconnected = False
                try:
                    while not session.cancel.is_set():
                        ok, frame = await asyncio.to_thread(reader.read)
                        if not ok or frame is None:
                            disconnected = True
                            break
                        annotated = await asyncio.to_thread(tracker.annotate, frame)
                        encoded = await asyncio.to_thread(tracker.encode_jpeg, annotated)
                        session.latest_jpeg = encoded
                        session.status.frame_sequence += 1
                        self._set_state(session, SessionState.LIVE)
                        await asyncio.sleep(0)
                finally:
                    self._close_reader(session, reader)

                if session.cancel.is_set():
                    break
                if not disconnected:
                    continue
                if attempt >= self._reconnect_attempts:
                    session.status.error_code = "CAMERA_STREAM_UNAVAILABLE"
                    self._set_state(session, SessionState.ERROR)
                    return

                attempt += 1
                self._set_state(session, SessionState.RECONNECTING)
                try:
                    await asyncio.wait_for(
                        session.cancel.wait(), timeout=self._reconnect_delay_seconds
                    )
                except TimeoutError:
                    pass
        except asyncio.CancelledError:
            raise
        except Exception:
            session.status.error_code = "AI_PREVIEW_FAILED"
            self._set_state(session, SessionState.ERROR)
        finally:
            self._close_reader(session, session.reader)
            if session.cancel.is_set():
                self._set_state(session, SessionState.STOPPED)
            session.task = None

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

