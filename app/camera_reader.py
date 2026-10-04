from __future__ import annotations

import os
import math
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import url2pathname

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("OPENCV_FFMPEG_DEBUG", "0")

import cv2
import numpy as np

from app.schemas import StartSessionRequest


def _authenticated_url(request: StartSessionRequest) -> str:
    if not request.username:
        return request.stream_url

    parts = urlsplit(request.stream_url)
    password = request.password.get_secret_value() if request.password else ""
    credentials = f"{quote(request.username, safe='')}:{quote(password, safe='')}@"
    return urlunsplit((parts.scheme, credentials + parts.netloc, parts.path, parts.query, parts.fragment))


class CameraReader:
    def __init__(self, request: StartSessionRequest, timeout_seconds: float = 5.0, *, recorded_root: Path | None = None) -> None:
        self.frame_interval = 0.0
        self.position_ms: float | None = None
        self.fps: float | None = None
        parts = urlsplit(request.stream_url)
        if request.source_type == "RECORDED":
            if recorded_root is None or parts.scheme != "file" or parts.netloc or parts.query or parts.fragment or request.username or request.password:
                raise ValueError("AI_RECORDED_VIDEO_INVALID")
            path = Path(url2pathname(parts.path)).resolve()
            if not path.is_relative_to(recorded_root.resolve()) or path.suffix.lower() != ".mp4" or not path.is_file():
                raise ValueError("AI_RECORDED_VIDEO_INVALID")
            self._capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
            fps = self._capture.get(cv2.CAP_PROP_FPS)
            self.fps = fps if math.isfinite(fps) and 1 <= fps <= 240 else None
            self.frame_interval = 1.0 / (fps if math.isfinite(fps) and 1 <= fps <= 240 else 25)
            self._closed = False
            return
        if parts.scheme not in ("http", "https", "rtsp") or not parts.hostname:
            raise ValueError("AI_SOURCE_INVALID")
        timeout_ms = max(1, round(timeout_seconds * 1_000))
        self._capture = cv2.VideoCapture(
            _authenticated_url(request),
            cv2.CAP_FFMPEG,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                timeout_ms,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                timeout_ms,
            ],
        )
        self._closed = False

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._closed:
            return False, None
        ok, frame = self._capture.read()
        if ok:
            position = self._capture.get(cv2.CAP_PROP_POS_MSEC)
            self.position_ms = position if math.isfinite(position) and position >= 0 else None
        return bool(ok), frame if ok else None

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._capture.release()
