from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np

from app.camera_reader import CameraReader
from app.schemas import StartSessionRequest
from app.tracker import FrameTracker


def request() -> StartSessionRequest:
    return StartSessionRequest(stream_url="http://camera.local:8080/video")


def test_camera_reader_applies_configured_native_timeout_before_open():
    capture = Mock()
    with patch("app.camera_reader.cv2.VideoCapture", return_value=capture) as open_capture:
        reader = CameraReader(request(), timeout_seconds=2.5)
        reader.close()

    assert open_capture.call_args.args[2] == [
        53, 2500, 54, 2500
    ]


def test_tracker_downscales_oversized_frames_before_inference():
    tracker = FrameTracker.__new__(FrameTracker)
    tracker._request = request()
    tracker._jpeg_quality = 85
    tracker._max_frame_dimension = 1280
    model = Mock()
    result = Mock()
    result.plot.return_value = np.zeros((1280, 640, 3), dtype=np.uint8)
    model.track.return_value = [result]
    tracker._model = model

    tracker.annotate(np.zeros((2000, 1000, 3), dtype=np.uint8))

    inference_frame = model.track.call_args.args[0]
    assert inference_frame.shape[:2] == (1280, 640)

