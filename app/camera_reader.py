from __future__ import annotations

from urllib.parse import quote, urlsplit, urlunsplit

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
    def __init__(self, request: StartSessionRequest) -> None:
        self._capture = cv2.VideoCapture(_authenticated_url(request), cv2.CAP_FFMPEG)
        self._capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5_000)
        self._capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5_000)
        self._closed = False

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._closed:
            return False, None
        ok, frame = self._capture.read()
        return bool(ok), frame if ok else None

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._capture.release()

