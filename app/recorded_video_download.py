from __future__ import annotations

import http.client
import re
import time
from pathlib import Path
from urllib.parse import urlsplit


MAX_BYTES = 200 * 1024 * 1024
CONTENT_PATH = re.compile(
    r"/api/cameras/[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}/recorded-video/content/[0-9a-f]{32}",
)


def download_recorded_video(url: str, backend_origin: str | None, key: str | None, destination: Path) -> None:
    if not backend_origin or not key:
        raise ValueError("AI_RECORDED_VIDEO_INVALID")

    try:
        source = urlsplit(url)
        backend = urlsplit(backend_origin)
        if (
            source.scheme != "https"
            or backend.scheme != "https"
            or not source.hostname
            or (source.hostname, source.port or 443) != (backend.hostname, backend.port or 443)
            or source.username is not None or source.password is not None
            or backend.username is not None or backend.password is not None
            or source.query or source.fragment
            or backend.query or backend.fragment or backend.path not in ("", "/")
            or not CONTENT_PATH.fullmatch(source.path)
        ):
            raise ValueError
    except ValueError:
        raise ValueError("AI_RECORDED_VIDEO_INVALID") from None

    connection = http.client.HTTPSConnection(source.hostname, source.port, timeout=15)
    try:
        deadline = time.monotonic() + 120
        connection.request("GET", source.path, headers={"X-AI-Service-Key": key, "Accept": "video/mp4"})
        with connection.getresponse() as response:
            length = int(response.getheader("Content-Length", "0"))
            if response.status != 200 or response.getheader("Content-Type") != "video/mp4" or not 0 < length <= MAX_BYTES:
                raise ValueError
            total = 0
            with destination.open("wb") as output:
                while chunk := response.read1(65536):
                    total += len(chunk)
                    if total > length or time.monotonic() > deadline:
                        raise ValueError
                    output.write(chunk)
            if total != length:
                raise ValueError
    except (OSError, http.client.HTTPException, ValueError):
        raise ValueError("AI_RECORDED_VIDEO_DOWNLOAD_FAILED") from None
    finally:
        connection.close()
