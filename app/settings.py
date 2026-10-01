from __future__ import annotations

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "127.0.0.1"
    port: int = 8090
    internal_service_key: SecretStr | None = None
    model_path: str = "../yolo26n.pt"
    tracker: str = "bytetrack.yaml"
    device: str = "cuda:0"
    half: bool = True
    confidence: float = Field(default=0.50, ge=0, le=1)
    jpeg_quality: int = Field(default=85, ge=1, le=100)
    reconnect_attempts: int = Field(default=5, ge=0)
    reconnect_delay_seconds: float = Field(default=1.0, ge=0)
    frame_timeout_seconds: float = Field(default=5.0, gt=0)

