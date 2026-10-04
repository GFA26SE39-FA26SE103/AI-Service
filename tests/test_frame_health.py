from __future__ import annotations

import cv2
import numpy as np

from app.frame_health import FrameHealthAnalyzer


def checkerboard(size: int = 256) -> np.ndarray:
    cells = np.indices((size, size)).sum(axis=0) // 16
    gray = ((cells % 2) * 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def test_invalid_encoded_frame_is_reported_without_throwing() -> None:
    result = FrameHealthAnalyzer().analyze_encoded(b"not-an-image")

    assert result.issues == ["CAMERA_FRAME_INVALID"]


def test_uniform_frame_is_conservatively_reported_as_blocked() -> None:
    frame = np.full((256, 256, 3), 120, dtype=np.uint8)

    result = FrameHealthAnalyzer().analyze_frame(frame)

    assert result.issues == ["CAMERA_VIEW_BLOCKED"]


def test_sharp_detailed_frame_is_clear() -> None:
    result = FrameHealthAnalyzer().analyze_frame(checkerboard())

    assert result.issues == []


def test_blurred_detailed_frame_is_reported_as_blurred_not_blocked() -> None:
    frame = cv2.GaussianBlur(checkerboard(), (31, 31), 0)

    result = FrameHealthAnalyzer(blur_variance_threshold=100).analyze_frame(frame)

    assert result.issues == ["CAMERA_VIEW_BLURRED"]
