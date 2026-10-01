from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class SessionState(str, Enum):
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    LIVE = "LIVE"
    RECONNECTING = "RECONNECTING"
    ERROR = "ERROR"


class StartSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stream_url: str = Field(min_length=1, repr=False)
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

