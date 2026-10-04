from __future__ import annotations

import asyncio
import logging
import secrets
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID

from fastapi import Depends, FastAPI, Header, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from app.camera_reader import CameraReader
from app.frame_health import FrameHealthAnalyzer
from app.monitoring_schemas import MeasurementPage, StartMonitoringRequest
from app.monitoring_tracker import MonitoringFrameTracker
from app.schemas import FrameHealthResponse, SessionStatusResponse, StartSessionRequest
from app.session_manager import (
    FrameNotReadyError,
    SessionCapacityError,
    SessionManager,
    SessionNotRunningError,
    SessionOwnershipError,
)
from app.settings import Settings

logger = logging.getLogger("ai_preview")


def _frame_tracker(request: StartSessionRequest, settings: Settings):
    # Keep CPU-only health checks available without importing the YOLO runtime.
    from app.tracker import FrameTracker

    return FrameTracker(request, settings.jpeg_quality, settings.max_frame_dimension)


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
        lambda request: _frame_tracker(request, resolved_settings),
        reconnect_attempts=resolved_settings.reconnect_attempts,
        reconnect_delay_seconds=resolved_settings.reconnect_delay_seconds,
        session_idle_timeout_seconds=resolved_settings.session_idle_timeout_seconds,
        max_observation_gap_ms=resolved_settings.max_observation_gap_ms,
        monitoring_tracker_factory=lambda request: MonitoringFrameTracker(request,resolved_settings.jpeg_quality,resolved_settings.max_frame_dimension),
    )
    @asynccontextmanager
    async def lifespan(_app):
        yield
        if hasattr(manager,"shutdown"): await manager.shutdown()
    app = FastAPI(title="S.H.E.P.H.E.R.D AI Monitoring", version="0.2.0",lifespan=lifespan)

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
    frame_health = FrameHealthAnalyzer(
        blur_variance_threshold=resolved_settings.health_blur_variance_threshold,
        blocked_stddev_threshold=resolved_settings.health_blocked_stddev_threshold,
        blocked_edge_ratio_threshold=resolved_settings.health_blocked_edge_ratio_threshold,
        max_dimension=resolved_settings.health_frame_max_dimension,
    )

    @app.exception_handler(SessionOwnershipError)
    async def handle_owner_error(_request,error):
        code=str(error)
        if code not in ("AI_SESSION_OWNER_MISMATCH","AI_SESSION_MONITORING_OWNED"): code="AI_SESSION_MONITORING_OWNED"
        return _error_response(409,code,"This session is owned by backend monitoring; deactivate before changing it.")

    @app.exception_handler(SessionCapacityError)
    async def handle_capacity(_request,_error):
        return _error_response(409,"AI_SESSION_CAPACITY","Another camera session is using the GPU.")

    @app.exception_handler(SessionNotRunningError)
    async def handle_not_running(_request,_error):
        return _error_response(409,"AI_PREVIEW_NOT_RUNNING","AI session is not running.")

    @app.post("/monitoring/sessions/{camera_id}/start",response_model=SessionStatusResponse,dependencies=protected)
    async def start_monitoring(camera_id: UUID,body: StartMonitoringRequest):
        for name,value in (("model",resolved_settings.model_path),("tracker",resolved_settings.tracker),("device",resolved_settings.device),("half",resolved_settings.half)):
            if name not in body.model_fields_set: setattr(body,name,value)
        return await manager.start_monitoring(camera_id,body)

    @app.get("/monitoring/sessions/{camera_id}/measurements",response_model=MeasurementPage,dependencies=protected)
    async def measurements(camera_id: UUID,owner_id: UUID=Header(alias="X-AI-Monitoring-Owner"),after_session_id: UUID|None=None,
                           after_sequence: int=Query(default=0,ge=0),limit: int=Query(default=64,ge=1,le=64)):
        return await manager.measurements(camera_id,owner_id,after_session_id,after_sequence,limit)

    @app.delete("/monitoring/sessions/{camera_id}",response_model=SessionStatusResponse,dependencies=protected)
    async def stop_monitoring(camera_id: UUID,owner_id: UUID=Header(alias="X-AI-Monitoring-Owner")):
        return await manager.stop_monitoring(camera_id,owner_id)

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
        "/frame-health",
        response_model=FrameHealthResponse,
        dependencies=protected,
    )
    async def analyze_frame_health(request: Request) -> FrameHealthResponse:
        encoded = bytearray()
        async for chunk in request.stream():
            if len(encoded) + len(chunk) > resolved_settings.health_frame_max_bytes:
                raise ApiError(413, "AI_FRAME_TOO_LARGE", "Camera frame exceeded the health-check size limit.")
            encoded.extend(chunk)
        result = await asyncio.to_thread(frame_health.analyze_encoded, bytes(encoded))
        return FrameHealthResponse(issues=result.issues)

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

    @app.get("/sessions/{camera_id}/frame/next", dependencies=protected)
    async def next_session_frame(
        camera_id: UUID,
        after_sequence: int = Query(default=0, ge=0),
        after_session_id: UUID | None = None,
    ) -> Response:
        try:
            frame = await manager.next_frame(camera_id, after_sequence, after_session_id)
        except SessionNotRunningError as error:
            raise ApiError(409, "AI_PREVIEW_NOT_RUNNING", "AI preview is not running.") from error
        if frame is None:
            return Response(status_code=204, headers={"Cache-Control": "no-store"})
        jpeg, sequence, session_id = frame
        return Response(
            content=jpeg,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-store",
                "X-Frame-Sequence": str(sequence),
                "X-Session-Id": str(session_id) if session_id is not None else "",
            },
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
