from __future__ import annotations

import cv2
import numpy as np
from ultralytics import YOLO

from app.schemas import StartSessionRequest


class FrameTracker:
    def __init__(
        self,
        request: StartSessionRequest,
        jpeg_quality: int = 85,
        max_frame_dimension: int = 1280,
    ) -> None:
        self._request = request
        self._model = YOLO(request.model)
        self._jpeg_quality = jpeg_quality
        self._max_frame_dimension = max_frame_dimension

    def annotate(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        largest_dimension = max(height, width)
        if largest_dimension > self._max_frame_dimension:
            scale = self._max_frame_dimension / largest_dimension
            frame = cv2.resize(
                frame,
                (round(width * scale), round(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
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
