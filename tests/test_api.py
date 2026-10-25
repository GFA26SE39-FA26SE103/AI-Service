from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import cv2
import numpy as np
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.main import create_app
from app.schemas import SessionState, SessionStatusResponse
from app.session_manager import (
    FrameNotReadyError,
    SessionCapacityError,
    SessionConfigurationConflictError,
    SessionNotRunningError,
    ViewerLeaseError,
)
from app.settings import Settings


class FakeSessionManager:
    def __init__(self) -> None:
        self.state = SessionState.STOPPED
        self.camera_id: UUID | None = None
        self.frame_error: Exception | None = None
        self.start_error: Exception | None = None
        self.viewer_leases: set[UUID] = set()
        self.released: list[tuple[UUID, UUID]] = []
        self.force_stopped: list[UUID] = []

    def _status(
        self, camera_id: UUID, viewer_lease_id: UUID | None = None
    ) -> SessionStatusResponse:
        return SessionStatusResponse(
            camera_id=camera_id,
            state=self.state,
            started_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            frame_sequence=1 if self.state is SessionState.LIVE else 0,
            viewer_lease_id=viewer_lease_id,
        )

    async def start(self, camera_id, _request):
        if self.start_error:
            raise self.start_error
        self.camera_id = camera_id
        self.state = SessionState.LIVE
        viewer_lease_id = uuid4()
        self.viewer_leases.add(viewer_lease_id)
        return self._status(camera_id, viewer_lease_id)

    async def attach_viewer(self, camera_id):
        if self.state is not SessionState.LIVE:
            raise SessionNotRunningError("AI_PREVIEW_NOT_RUNNING")
        viewer_lease_id = uuid4()
        self.viewer_leases.add(viewer_lease_id)
        return self._status(camera_id, viewer_lease_id)

    def _require_lease(self, viewer_lease_id: UUID) -> None:
        if viewer_lease_id not in self.viewer_leases:
            raise ViewerLeaseError("AI_VIEWER_LEASE_EXPIRED")

    async def status(self, camera_id, viewer_lease_id):
        self._require_lease(viewer_lease_id)
        return self._status(camera_id, viewer_lease_id)

    async def frame(self, _camera_id, viewer_lease_id):
        self._require_lease(viewer_lease_id)
        if self.frame_error:
            raise self.frame_error
        return b"\xff\xd8tracked-jpeg\xff\xd9"

    async def next_frame(
        self, camera_id, viewer_lease_id, after_sequence, after_session_id
    ):
        self._require_lease(viewer_lease_id)
        if self.frame_error:
            raise self.frame_error
        if after_sequence >= 1 and after_session_id == self._status(camera_id).session_id:
            return None
        return b"\xff\xd8tracked-jpeg\xff\xd9", 1, self._status(camera_id).session_id

    async def release_viewer(self, camera_id, viewer_lease_id):
        self.released.append((camera_id, viewer_lease_id))
        self.viewer_leases.discard(viewer_lease_id)
        if not self.viewer_leases:
            self.state = SessionState.STOPPED
        return self._status(camera_id)

    async def force_stop(self, camera_id):
        self.force_stopped.append(camera_id)
        self.viewer_leases.clear()
        self.state = SessionState.STOPPED
        return self._status(camera_id)


def client(manager=None, settings=None) -> TestClient:
    app = create_app(settings or Settings(), manager or FakeSessionManager())
    return TestClient(app, raise_server_exceptions=False)


def payload() -> dict:
    return {
        "stream_url": "http://camera.local:8080/video",
        "username": "viewer",
        "password": "secret",
    }


def viewer_headers(viewer_lease_id: str) -> dict[str, str]:
    return {"X-AI-Viewer-Lease": viewer_lease_id}


def test_health_returns_ready():
    response = client().get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_frame_health_endpoint_returns_cpu_only_visual_result():
    cells = np.indices((256, 256)).sum(axis=0) // 16
    frame = cv2.cvtColor(((cells % 2) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok

    response = client().post(
        "/frame-health",
        content=encoded.tobytes(),
        headers={"Content-Type": "image/jpeg"},
    )

    assert response.status_code == 200
    assert response.json() == {"issues": []}


def test_frame_health_endpoint_rejects_oversized_body_before_decode():
    settings = Settings(health_frame_max_bytes=1024)

    response = client(settings=settings).post(
        "/frame-health",
        content=b"x" * 1025,
        headers={"Content-Type": "image/jpeg"},
    )

    assert response.status_code == 413
    assert response.json()["code"] == "AI_FRAME_TOO_LARGE"


def test_start_status_frame_release_contract():
    camera_id = uuid4()
    test_client = client()

    start = test_client.post(f"/sessions/{camera_id}/start", json=payload())
    lease = start.json()["viewer_lease_id"]
    headers = viewer_headers(lease)
    status = test_client.get(f"/sessions/{camera_id}/status", headers=headers)
    frame = test_client.get(f"/sessions/{camera_id}/frame", headers=headers)
    stop = test_client.delete(f"/sessions/{camera_id}/viewer", headers=headers)

    assert start.status_code == 200
    assert start.json()["state"] == "LIVE"
    assert UUID(lease)
    assert status.json()["camera_id"] == str(camera_id)
    assert frame.status_code == 200
    assert frame.headers["content-type"] == "image/jpeg"
    assert frame.headers["cache-control"] == "no-store"
    assert frame.content.startswith(b"\xff\xd8")
    assert stop.json()["state"] == "STOPPED"


def test_next_frame_exposes_sequence_and_returns_no_content_for_duplicate():
    camera_id = uuid4()
    test_client = client()
    lease = test_client.post(f"/sessions/{camera_id}/start", json=payload()).json()[
        "viewer_lease_id"
    ]
    headers = viewer_headers(lease)
    first = test_client.get(
        f"/sessions/{camera_id}/frame/next?after_sequence=0", headers=headers
    )
    assert first.status_code == 200
    assert first.headers["x-frame-sequence"] == "1"
    assert first.headers["cache-control"] == "no-store"
    duplicate = test_client.get(
        f"/sessions/{camera_id}/frame/next?after_sequence=1", headers=headers
    )
    assert duplicate.status_code == 204


def test_frame_returns_409_when_stopped_and_503_when_not_ready():
    camera_id = uuid4()
    manager = FakeSessionManager()
    lease = uuid4()
    manager.viewer_leases.add(lease)
    manager.frame_error = SessionNotRunningError("AI_PREVIEW_NOT_RUNNING")
    test_client = client(manager)

    stopped = test_client.get(
        f"/sessions/{camera_id}/frame", headers=viewer_headers(str(lease))
    )
    manager.frame_error = FrameNotReadyError("AI_FRAME_NOT_READY")
    not_ready = test_client.get(
        f"/sessions/{camera_id}/frame", headers=viewer_headers(str(lease))
    )

    assert stopped.status_code == 409
    assert stopped.json() == {"code": "AI_PREVIEW_NOT_RUNNING", "message": "AI preview is not running."}
    assert not_ready.status_code == 503
    assert not_ready.json() == {"code": "AI_FRAME_NOT_READY", "message": "AI preview frame is not ready."}


def test_internal_key_is_required_when_configured():
    camera_id = uuid4()
    settings = Settings(internal_service_key=SecretStr("internal-key"))
    test_client = client(settings=settings)

    missing = test_client.get(f"/sessions/{camera_id}/status")
    accepted = test_client.get(
        f"/sessions/{camera_id}/status",
        headers={
            "X-AI-Service-Key": "internal-key",
            "X-AI-Viewer-Lease": str(uuid4()),
        },
    )

    assert missing.status_code == 401
    assert missing.json()["code"] == "AI_SERVICE_UNAUTHORIZED"
    assert accepted.status_code == 409
    assert accepted.json()["code"] == "AI_VIEWER_LEASE_EXPIRED"


def test_start_returns_stable_capacity_error():
    camera_id = uuid4()
    manager = FakeSessionManager()
    manager.start_error = SessionCapacityError("AI_SESSION_CAPACITY")

    response = client(manager).post(f"/sessions/{camera_id}/start", json=payload())

    assert response.status_code == 409
    assert response.json()["code"] == "AI_SESSION_CAPACITY"


def test_status_frame_and_next_frame_require_viewer_lease_header():
    camera_id = uuid4()
    test_client = client()

    responses = [
        test_client.get(f"/sessions/{camera_id}/status"),
        test_client.get(f"/sessions/{camera_id}/frame"),
        test_client.get(f"/sessions/{camera_id}/frame/next"),
    ]

    assert all(response.status_code == 422 for response in responses)
    assert all(
        response.json()["code"] == "AI_VIEWER_LEASE_REQUIRED"
        for response in responses
    )


def test_unknown_viewer_lease_returns_stable_expired_error_without_leaking_data():
    camera_id = uuid4()
    unknown_lease = uuid4()
    response = client().get(
        f"/sessions/{camera_id}/status",
        headers=viewer_headers(str(unknown_lease)),
    )

    assert response.status_code == 409
    assert response.json() == {
        "code": "AI_VIEWER_LEASE_EXPIRED",
        "message": "AI viewer lease is missing or expired.",
    }
    assert all(
        secret not in response.text
        for secret in (
            "camera.local",
            "secret",
            str(unknown_lease),
            "X-AI-Service-Key",
        )
    )


def test_attach_existing_creates_viewer_lease_without_starting_new_worker():
    camera_id = uuid4()
    manager = FakeSessionManager()
    manager.state = SessionState.LIVE

    response = client(manager).post(f"/sessions/{camera_id}/viewers/attach")

    assert response.status_code == 200
    assert UUID(response.json()["viewer_lease_id"]) in manager.viewer_leases


def test_release_unknown_lease_is_idempotent_and_force_stop_is_separate():
    camera_id = uuid4()
    manager = FakeSessionManager()
    unknown = uuid4()
    test_client = client(manager)

    release = test_client.delete(
        f"/sessions/{camera_id}/viewer", headers=viewer_headers(str(unknown))
    )
    forced = test_client.delete(f"/sessions/{camera_id}/force")

    assert release.status_code == 200
    assert manager.released == [(camera_id, unknown)]
    assert forced.status_code == 200
    assert manager.force_stopped == [camera_id]


def test_start_returns_stable_configuration_conflict_without_sensitive_data():
    camera_id = uuid4()
    manager = FakeSessionManager()
    manager.start_error = SessionConfigurationConflictError(
        "AI_PREVIEW_CONFIGURATION_CONFLICT"
    )

    response = client(manager).post(f"/sessions/{camera_id}/start", json=payload())

    assert response.status_code == 409
    assert response.json()["code"] == "AI_PREVIEW_CONFIGURATION_CONFLICT"
    assert all(
        secret not in response.text
        for secret in ("camera.local", "viewer", "secret")
    )
