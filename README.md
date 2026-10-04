# S.H.E.P.H.E.R.D AI Preview Service

Private native Windows service for MF-01 controlled preview. It reads an IP Webcam stream or an uploaded MP4, runs Ultralytics YOLO person detection with ByteTrack, and exposes only the latest annotated JPEG to the ASP.NET backend.

The React frontend must never call this service or the phone camera directly. This service does not write detections, operational events, incidents, or images to the database.

## Local runtime

- Python: `C:\FPT University\CAPSTONE\setup\.venv\Scripts\python.exe`
- Model: `C:\FPT University\CAPSTONE\setup\yolo26n.pt`
- FFmpeg: portable build under `C:\FPT University\CAPSTONE\setup\tools\ffmpeg`
- Default API: `http://127.0.0.1:8090`

PyTorch CUDA is installed separately from `requirements.txt` so a normal package install cannot silently replace it with a CPU-only wheel:

```powershell
$python = 'C:\FPT University\CAPSTONE\setup\.venv\Scripts\python.exe'
$uv = 'C:\FPT University\CAPSTONE\setup\tools\uv\uv.exe'
& $uv pip install --python $python torch torchvision --index-url https://download.pytorch.org/whl/cu118
& $uv pip install --python $python -r 'C:\FPT University\CAPSTONE\setup\ai-service\requirements-dev.txt'
& $python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Copy `.env.example` to `.env` for local overrides. Do not commit camera URLs, usernames, passwords, JWTs, or the internal service key.

Start the private service from this directory:

```powershell
& '..\.venv\Scripts\python.exe' -m uvicorn app.main:app --host 127.0.0.1 --port 8090
```

Internal contract:

- `GET /health`
- `POST /frame-health` (private CPU-only JPEG usability analysis)
- `POST /sessions/{cameraId}/start`
- `GET /sessions/{cameraId}/status`
- `GET /sessions/{cameraId}/frame`
- `DELETE /sessions/{cameraId}`

Session states are `STOPPED`, `STARTING`, `LIVE`, `RECONNECTING`, `COMPLETED`, and `ERROR`. Frame reads use `409 AI_PREVIEW_NOT_RUNNING` or `503 AI_FRAME_NOT_READY` when appropriate. Unexpected errors return only `AI_PREVIEW_FAILED`; camera credentials and authenticated stream URLs are never echoed.

`POST /frame-health` is called only by the backend health worker. It accepts one bounded encoded frame and returns whole-camera visual issue codes (`CAMERA_VIEW_BLOCKED`, `CAMERA_VIEW_BLURRED`, or `CAMERA_FRAME_INVALID`). It downsizes before deterministic OpenCV checks and does not load YOLO or use the GPU. Darkness is intentionally not classified. `CAMERA_VIEW_FROZEN` remains a supported health-event contract but this endpoint does not guess frozen state from an empty/static supermarket scene; a future source adapter must supply reliable freshness/sequence metadata.

The native preview intentionally runs one GPU session at a time. Input frames are downscaled to a maximum dimension of `AI_MAX_FRAME_DIMENSION` (1280 by default), OpenCV open/read calls use `AI_FRAME_TIMEOUT_SECONDS`, and a session self-stops after `AI_SESSION_IDLE_TIMEOUT_SECONDS` without status/frame polling. These limits protect the laptop if the browser closes or authentication expires.

## IP Webcam prerequisite

Put the phone and laptop on the same LAN. Start IP Webcam and use the exact MJPEG URL advertised by the app, commonly `http://PHONE_IP:8080/video`. Validate the URL in a browser before starting an AI session.

### Native startup order

1. In IP Webcam, tap **Start server** and note the LAN address shown at the bottom of the phone screen.
2. Open `http://PHONE_IP:8080` on the laptop and verify the browser can display video. Windows Firewall must allow the private-network connection.
3. Configure the video endpoint (`http://PHONE_IP:8080/video`, not the base HTML page) in the backend camera connection (`LIVE` + `HTTP`). Keep username/password separate, never inside the URL.
4. Start this service with the Uvicorn command above.
5. Start ASP.NET at `http://localhost:5080`, then React at `http://localhost:5173`.
6. In React **Cameras**, click **Test & enable**, then **Start AI preview**. A person in view should have a YOLO box and a camera-local ByteTrack ID.

Run the opt-in real-camera contract without printing its URL or credentials:

```powershell
$env:AI_TEST_STREAM_URL = 'http://PHONE_IP:8080/video'
$env:AI_TEST_STREAM_USERNAME = ''
$env:AI_TEST_STREAM_PASSWORD = ''
& '..\.venv\Scripts\python.exe' -m pytest tests/test_ip_webcam_contract.py -q
```

If the phone is disconnected, status transitions through `RECONNECTING` and then `ERROR`; this smoke-test feature does not create an Operational Incident. Polling returns the newest annotated JPEG (roughly 3–10 FPS depending on GPU/network), not browser-native video. Stop the preview before editing camera connection details so the restarted session uses only the current credentials.

## Recorded video fallback (2026-10-02)

No IP Webcam server is required for this mode. In React **Cameras**, create an **Uploaded video (test source)** camera or select an existing ACTIVE camera → **Upload video**, choose an MP4 up to 200 MB → **Save video source → Test & enable → Start AI preview**. Backend stores/validates the file and forwards `source_type: "RECORDED"` with its local `file:///...` URI. AI has no public upload/static-file endpoint. Live requests cannot open local files.

Default recorded storage is resolved from this checkout: `CAPSTONE/Backend/Back-End/src/Supermarket.Api/.local/videos`. Set `AI_RECORDED_ROOT` to the same absolute directory as backend `Video:RecordedRoot` when moving directories, publishing or using custom storage. Keep both services on the same machine for this native workflow. Reader rejects files outside that root, non-local file URIs and non-MP4 paths. Do not put secrets or uploaded videos in Git.

Recorded files are processed sequentially, paced by source FPS (not faster than source playback; slow inference may take longer). Every frame reaches the tracker, but React only displays the newest JPEG. EOF changes state to `COMPLETED`, releases the worker and retains the final frame; it does not reconnect/loop or report a camera outage. Stop/start replays from frame one with fresh camera-local IDs. Invalid input returns `ERROR` without exposing file paths. A live request omitting `source_type` remains backwards compatible.

This is detection/tracking preview only, not configured continuous monitoring. No ROI measurements, per-zone incident threshold evaluation, sustain/cooldown or OperationalEvents are written. Backend health probes still test file readability independently of playback completion. Review zone ROIs when changing scenes.

Tests (including a generated three-frame MP4 through real OpenCV reader, pacing/EOF/path guards):

```powershell
& '..\.venv\Scripts\python.exe' -m pytest -q
```
