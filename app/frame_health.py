from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameHealthResult:
    issues: list[str]


class FrameHealthAnalyzer:
    """Deterministic, CPU-only frame usability checks; no person inference."""

    def __init__(
        self,
        *,
        blur_variance_threshold: float = 80.0,
        blocked_stddev_threshold: float = 4.0,
        blocked_edge_ratio_threshold: float = 0.001,
        max_dimension: int = 640,
    ) -> None:
        self._blur_variance_threshold = blur_variance_threshold
        self._blocked_stddev_threshold = blocked_stddev_threshold
        self._blocked_edge_ratio_threshold = blocked_edge_ratio_threshold
        self._max_dimension = max_dimension

    def analyze_encoded(self, encoded: bytes) -> FrameHealthResult:
        if not encoded:
            return FrameHealthResult(["CAMERA_FRAME_INVALID"])
        frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return FrameHealthResult(["CAMERA_FRAME_INVALID"])
        return self.analyze_frame(frame)

    def analyze_frame(self, frame: np.ndarray) -> FrameHealthResult:
        if frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 32:
            return FrameHealthResult(["CAMERA_FRAME_INVALID"])
        height, width = frame.shape[:2]
        largest = max(height, width)
        if largest > self._max_dimension:
            scale = self._max_dimension / largest
            frame = cv2.resize(
                frame,
                (max(32, round(width * scale)), max(32, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        contrast = float(np.std(gray))
        edge_ratio = float(np.count_nonzero(cv2.Canny(gray, 50, 150))) / gray.size
        if contrast <= self._blocked_stddev_threshold and edge_ratio <= self._blocked_edge_ratio_threshold:
            return FrameHealthResult(["CAMERA_VIEW_BLOCKED"])
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if sharpness < self._blur_variance_threshold:
            return FrameHealthResult(["CAMERA_VIEW_BLURRED"])
        return FrameHealthResult([])
