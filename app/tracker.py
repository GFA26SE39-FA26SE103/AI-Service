from __future__ import annotations

import cv2
import numpy as np
from ultralytics import YOLO

from app.schemas import StartSessionRequest


class FrameTracker:
    def __init__(self, request: StartSessionRequest, jpeg_quality: int = 85) -> None:
        self._request = request
        self._model = YOLO(request.model)
        self._jpeg_quality = jpeg_quality

    def annotate(self, frame: np.ndarray) -> np.ndarray:
        results = self._model.track(
            frame,
            persist=True,
            tracker=self._request.tracker,
            classes=self._request.classes,
            conf=self._request.confidence,
            device=self._request.device,
            half=self._request.half,
            verbose=False,
        )
        return results[0].plot(labels=True, conf=True)

    def encode_jpeg(self, frame: np.ndarray) -> bytes:
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality],
        )
        if not ok:
            raise RuntimeError("AI_FRAME_ENCODE_FAILED")
        return encoded.tobytes()

