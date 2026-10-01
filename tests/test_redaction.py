from __future__ import annotations

import logging
from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import create_app
from app.settings import Settings


class ExplodingManager:
    async def start(self, _camera_id, _request):
        raise RuntimeError("http://viewer:super-secret@camera.local/video")

    async def status(self, _camera_id):
        raise RuntimeError("super-secret")

    async def frame(self, _camera_id):
        raise RuntimeError("super-secret")

    async def stop(self, _camera_id):
        raise RuntimeError("super-secret")


def test_errors_and_logs_never_contain_password_or_authenticated_uri(caplog):
    caplog.set_level(logging.ERROR)
    test_client = TestClient(
        create_app(Settings(), ExplodingManager()),
        raise_server_exceptions=False,
    )
    camera_id = uuid4()

    response = test_client.post(
        f"/sessions/{camera_id}/start",
        json={
            "stream_url": "http://viewer:super-secret@camera.local/video",
            "username": "viewer",
            "password": "super-secret",
        },
    )

    combined = response.text + caplog.text
    assert response.status_code == 500
    assert response.json() == {"code": "AI_PREVIEW_FAILED", "message": "AI preview failed."}
    assert "super-secret" not in combined
    assert "viewer:" not in combined


def test_validation_errors_do_not_echo_secret_input():
    test_client = TestClient(create_app(Settings(), ExplodingManager()))
    camera_id = uuid4()

    response = test_client.post(
        f"/sessions/{camera_id}/start",
        json={
            "stream_url": "http://viewer:validation-secret@camera.local/video",
            "password": "validation-secret",
            "confidence": 2,
        },
    )

    assert response.status_code == 422
    assert response.json()["code"] == "AI_REQUEST_INVALID"
    assert "validation-secret" not in response.text

