from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.main import create_app
from app.schemas import SessionState, SessionStatusResponse
from app.session_manager import FrameNotReadyError, SessionNotRunningError
from app.settings import Settings


class FakeSessionManager:
    def __init__(self) -> None:
        self.state = SessionState.STOPPED
        self.camera_id: UUID | None = None
        self.frame_error: Exception | None = None

    def _status(self, camera_id: UUID) -> SessionStatusResponse:
        return SessionStatusResponse(
            camera_id=camera_id,
            state=self.state,
            started_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            frame_sequence=1 if self.state is SessionState.LIVE else 0,
        )

    async def start(self, camera_id, _request):
        self.camera_id = camera_id
        self.state = SessionState.LIVE
        return self._status(camera_id)

    async def status(self, camera_id):
        return self._status(camera_id)

    async def frame(self, _camera_id):
        if self.frame_error:
            raise self.frame_error
        return b"\xff\xd8tracked-jpeg\xff\xd9"

    async def stop(self, camera_id):
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


def test_health_returns_ready():
    response = client().get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_start_status_frame_stop_contract():
    camera_id = uuid4()
    test_client = client()

    start = test_client.post(f"/sessions/{camera_id}/start", json=payload())
    status = test_client.get(f"/sessions/{camera_id}/status")
    frame = test_client.get(f"/sessions/{camera_id}/frame")
    stop = test_client.delete(f"/sessions/{camera_id}")

    assert start.status_code == 200
    assert start.json()["state"] == "LIVE"
    assert status.json()["camera_id"] == str(camera_id)
    assert frame.status_code == 200
    assert frame.headers["content-type"] == "image/jpeg"
    assert frame.headers["cache-control"] == "no-store"
    assert frame.content.startswith(b"\xff\xd8")
    assert stop.json()["state"] == "STOPPED"


def test_frame_returns_409_when_stopped_and_503_when_not_ready():
    camera_id = uuid4()
    manager = FakeSessionManager()
    manager.frame_error = SessionNotRunningError("AI_PREVIEW_NOT_RUNNING")
    test_client = client(manager)

    stopped = test_client.get(f"/sessions/{camera_id}/frame")
    manager.frame_error = FrameNotReadyError("AI_FRAME_NOT_READY")
    not_ready = test_client.get(f"/sessions/{camera_id}/frame")

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
        headers={"X-AI-Service-Key": "internal-key"},
    )

    assert missing.status_code == 401
    assert missing.json()["code"] == "AI_SERVICE_UNAUTHORIZED"
    assert accepted.status_code == 200

