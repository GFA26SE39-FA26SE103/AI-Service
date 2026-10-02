from __future__ import annotations

import logging
import secrets
from typing import Any
from uuid import UUID

from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from app.camera_reader import CameraReader
from app.schemas import SessionStatusResponse, StartSessionRequest
from app.session_manager import (
    FrameNotReadyError,
    SessionCapacityError,
    SessionManager,
    SessionNotRunningError,
)
from app.settings import Settings
from app.tracker import FrameTracker

logger = logging.getLogger("ai_preview")


class ApiError(RuntimeError):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(code)
        self.status_code = status_code
        self.code = code
        self.message = message


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"code": code, "message": message},
    )


def create_app(settings: Settings | None = None, session_manager: Any | None = None) -> FastAPI:
    resolved_settings = settings or Settings()
    manager = session_manager or SessionManager(
        lambda request: CameraReader(request, resolved_settings.frame_timeout_seconds, recorded_root=resolved_settings.recorded_root),
        lambda request: FrameTracker(
            request,
            resolved_settings.jpeg_quality,
            resolved_settings.max_frame_dimension,
        ),
        reconnect_attempts=resolved_settings.reconnect_attempts,
        reconnect_delay_seconds=resolved_settings.reconnect_delay_seconds,
        session_idle_timeout_seconds=resolved_settings.session_idle_timeout_seconds,
    )
    app = FastAPI(title="S.H.E.P.H.E.R.D AI Preview", version="0.1.0")

    async def require_internal_key(
        supplied_key: str | None = Header(default=None, alias="X-AI-Service-Key"),
    ) -> None:
        configured = resolved_settings.internal_service_key
        if configured is None or not configured.get_secret_value():
            return
        if supplied_key is None or not secrets.compare_digest(
            supplied_key, configured.get_secret_value()
        ):
            raise ApiError(401, "AI_SERVICE_UNAUTHORIZED", "AI service authentication failed.")

    protected = [Depends(require_internal_key)]

    @app.exception_handler(ApiError)
    async def handle_api_error(_request: Request, error: ApiError) -> JSONResponse:
        return _error_response(error.status_code, error.code, error.message)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _request: Request, _error: RequestValidationError
    ) -> JSONResponse:
        return _error_response(422, "AI_REQUEST_INVALID", "AI preview request is invalid.")

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, _error: Exception) -> JSONResponse:
        logger.error(
            "AI preview request failed path=%s code=AI_PREVIEW_FAILED",
            request.url.path,
        )
        return _error_response(500, "AI_PREVIEW_FAILED", "AI preview failed.")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ready"}

    @app.post(
        "/sessions/{camera_id}/start",
        response_model=SessionStatusResponse,
        dependencies=protected,
    )
    async def start_session(
        camera_id: UUID, body: StartSessionRequest
    ) -> SessionStatusResponse:
        if "model" not in body.model_fields_set:
            body.model = resolved_settings.model_path
        if "tracker" not in body.model_fields_set:
            body.tracker = resolved_settings.tracker
        if "device" not in body.model_fields_set:
            body.device = resolved_settings.device
        if "half" not in body.model_fields_set:
            body.half = resolved_settings.half
        if "confidence" not in body.model_fields_set:
            body.confidence = resolved_settings.confidence
        try:
            return await manager.start(camera_id, body)
        except SessionCapacityError as error:
            raise ApiError(
                409,
                "AI_SESSION_CAPACITY",
                "Another AI preview session is already using the GPU.",
            ) from error

    @app.get(
        "/sessions/{camera_id}/status",
        response_model=SessionStatusResponse,
        dependencies=protected,
    )
    async def session_status(camera_id: UUID) -> SessionStatusResponse:
        return await manager.status(camera_id)

    @app.get("/sessions/{camera_id}/frame", dependencies=protected)
    async def session_frame(camera_id: UUID) -> Response:
        try:
            jpeg = await manager.frame(camera_id)
        except SessionNotRunningError as error:
            raise ApiError(
                409, "AI_PREVIEW_NOT_RUNNING", "AI preview is not running."
            ) from error
        except FrameNotReadyError as error:
            raise ApiError(503, "AI_FRAME_NOT_READY", "AI preview frame is not ready.") from error
        return Response(
            content=jpeg,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    @app.delete(
        "/sessions/{camera_id}",
        response_model=SessionStatusResponse,
        dependencies=protected,
    )
    async def stop_session(camera_id: UUID) -> SessionStatusResponse:
        return await manager.stop(camera_id)

    return app


app = create_app()
