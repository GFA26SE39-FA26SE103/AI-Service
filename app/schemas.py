from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class SessionState(str, Enum):
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    LIVE = "LIVE"
    RECONNECTING = "RECONNECTING"
    ERROR = "ERROR"
    COMPLETED = "COMPLETED"


class StartSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stream_url: str = Field(min_length=1, repr=False)
    source_type: Literal["LIVE", "RECORDED"] = "LIVE"
    username: str | None = Field(default=None, repr=False)
    password: SecretStr | None = Field(default=None, repr=False)
    model: str = "yolo26n.pt"
    tracker: str = "bytetrack.yaml"
    classes: list[int] = Field(default_factory=lambda: [0])
    confidence: Annotated[float, Field(ge=0, le=1)] = 0.50
    device: str = "cuda:0"
    half: bool = True


class SessionStatusResponse(BaseModel):
    camera_id: UUID
    state: SessionState
    started_at: datetime | None = None
    updated_at: datetime
    frame_sequence: int = 0
    error_code: str | None = None


class FrameHealthResponse(BaseModel):
    issues: list[
        Literal[
            "CAMERA_VIEW_BLOCKED",
            "CAMERA_VIEW_BLURRED",
            "CAMERA_VIEW_FROZEN",
            "CAMERA_FRAME_INVALID",
        ]
    ] = Field(default_factory=list)
