# S.H.E.P.H.E.R.D AI Preview Service

Private native Windows service for MF-01 controlled preview. It reads an IP Webcam stream or an uploaded MP4, runs Ultralytics YOLO person detection with ByteTrack, and exposes only the latest annotated JPEG to the ASP.NET backend.

The React frontend must never call this service or the phone camera directly. This service does not write detections, operational events, incidents, or images to the database.

## Local runtime

- Python: `C:\FPT University\CAPSTONE\setup\.venv\Scripts\python.exe`
- Model: `C:\FPT University\CAPSTONE\setup\yolo26s.pt` (YOLO26s official checkpoint; keep the weight outside Git)
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
If the checkpoint is absent on another machine, download `yolo26s.pt` from the official Ultralytics release into `CAPSTONE/setup` before starting a preview; changing the model name alone does not provide the weight file. Both AI's `AI_MODEL_PATH` and backend's `AiPreview:Model` must point to the same checkpoint, and both services need a restart.

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
- `GET /sessions/{cameraId}/frame/next?after_sequence=N&after_session_id=<uuid>`: waits up to 1 second for a newer annotated JPEG; `200` includes `X-Frame-Sequence` and `X-Session-Id`, `204` means no new frame. A changed session ID sends the current frame even if its sequence reset.
- `DELETE /sessions/{cameraId}`

Session states are `STOPPED`, `STARTING`, `LIVE`, `RECONNECTING`, `COMPLETED`, and `ERROR`. Frame reads use `409 AI_PREVIEW_NOT_RUNNING` or `503 AI_FRAME_NOT_READY` when appropriate. Unexpected errors return only `AI_PREVIEW_FAILED`; camera credentials and authenticated stream URLs are never echoed.

`POST /frame-health` is called only by the backend health worker. It accepts one bounded encoded frame and returns whole-camera visual issue codes (`CAMERA_VIEW_BLOCKED`, `CAMERA_VIEW_BLURRED`, or `CAMERA_FRAME_INVALID`). It downsizes before deterministic OpenCV checks and does not load YOLO or use the GPU. Darkness is intentionally not classified. `CAMERA_VIEW_FROZEN` remains a supported health-event contract but this endpoint does not guess frozen state from an empty/static supermarket scene; a future source adapter must supply reliable freshness/sequence metadata.

The native service runs one active GPU session at a time. Frames downscale to `AI_MAX_FRAME_DIMENSION` (1280 by default), OpenCV open/read calls use `AI_FRAME_TIMEOUT_SECONDS`, and a session self-stops after `AI_SESSION_IDLE_TIMEOUT_SECONDS` without status/frame polling. Preview-only lease renews on viewer access. Monitoring lease renews **only** on owner measurement reads (default 30 seconds); viewer access cannot keep an abandoned monitoring owner alive. These limits protect the laptop if the browser closes or authentication expires.

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

If the phone is disconnected, status transitions through `RECONNECTING` and then `ERROR`; connection health errors do not create an Operational Incident. The React viewer requests only newer annotated JPEGs; its frame rate depends on source FPS, YOLO/ByteTrack time, JPEG/network transfer and browser decoding, not a fixed 250 ms UI delay. It is not browser-native video. For preview-only sessions, stop the preview before editing camera connection details. For owned monitoring, deactivate the active zone configurations first; public viewer Stop only detaches the view.

## Recorded video fallback (2026-10-02)

No IP Webcam server is required for this mode. In React **Cameras**, create an **Uploaded video (test source)** camera or select an existing ACTIVE camera → **Upload video**, choose an MP4 up to 200 MB → **Save video source → Test & enable → Start AI preview**. Backend stores/validates the file and forwards `source_type: "RECORDED"` with its local `file:///...` URI. AI has no public upload/static-file endpoint. Live requests cannot open local files.

Default recorded storage is resolved from this checkout: `CAPSTONE/Backend/Back-End/src/Supermarket.Api/.local/videos`. Set `AI_RECORDED_ROOT` to the same absolute directory as backend `Video:RecordedRoot` when moving directories, publishing or using custom storage. Keep both services on the same machine for this native workflow. Reader rejects files outside that root, non-local file URIs and non-MP4 paths. Do not put secrets or uploaded videos in Git.

Recorded files are processed sequentially, paced by source FPS (not faster than source playback; slow inference may take longer). Every frame reaches the tracker, but React only displays the newest JPEG. EOF changes state to `COMPLETED`, releases the worker and retains the final frame; it does not reconnect/loop or report a camera outage. Preview-only stop/start replays from frame one with fresh camera-local IDs. Owned monitoring requires the backend owner to stop before a fresh start; public viewer start/stop never replays it. Invalid input returns `ERROR` without exposing file paths. A live request omitting `source_type` remains backwards compatible.

Preview-only still does detection/tracking. The monitoring protocol below adds ROI aggregates; threshold/sustain/cooldown and SQL persistence remain entirely BE-owned. Health probes test file readability independently of playback completion. Review ROIs when changing sources/scenes.

## BE-owned monitoring protocol v1 (03/10/2026)

- `POST /monitoring/sessions/{camera_id}/start`: authenticated internal request with owner UUID, configuration fingerprint and zone contexts (config ID/versionUTC, normalized ROI, confidence, queue_enabled). Credentials only in this private start body; never in status/measurements/errors.
- `GET .../measurements`: `X-AI-Monitoring-Owner`, `after_session_id`, `after_sequence`, `limit`1–64. Ordered batches have session/continuity IDs, source elapsed ms, capturedUTC, fingerprint and aggregate people/queue counts only. Bounded256 buffer reports `gap` on overrun/session mismatch. Do not poll only latest counts for sustained conditions.
- `DELETE /monitoring/sessions/{camera_id}`: owner-only stop. Existing `X-AI-Service-Key` applies. No FE direct access.

YOLO predicts once/frame, then confidence filters detections **before** distinct ByteTrack contexts (shared for equal-confidence zones). Count current confirmed person tracks whose bottom-center footpoint is inside/on ROI; not lost/predicted tracks. Queue membership requires ≥5000ms continuously observed in ROI; exits/misses/continuity changes reset dwell. Track IDs remain transient camera/context-local, never persisted as identity.

Recorded source uses PTS or validated FPS/frame-index fallback, never inference wall time for sustain. Live uses monotonic time with gap guard (`AI_MAX_OBSERVATION_GAP_MS` default2000). Reconnect/gap/reconfigure reset tracking/dwell. Same-source zone reconfigure happens between frames without reader reopen or recorded rewind, retaining sequence. EOF drains once and remains COMPLETED; new configuration at EOF invalidates old metrics but does not replay. Public preview start attaches; public stop409 cannot stop an owned monitor. BE viewer proxy handles this as detach/no-op. Replay requires owner stop then new start.

Remaining: physical density/calibration, waiting/checkout measurement, SQL/incident routing and tasks are not AI-service responsibilities. Dependency warnings (Starlette/httpx and installed Ultralytics `half` deprecation) remain; no dependency upgrade performed.

Tests (including a generated three-frame MP4 through real OpenCV reader, pacing/EOF/path guards):

```powershell
& '..\.venv\Scripts\python.exe' -m pytest -q
```

Verification 03/10/2026: 52 tests passed, 1 IP Webcam test skipped without `AI_TEST_STREAM_URL`. On this Windows sandbox, pytest was run with `-p no:cacheprovider` and a unique writable `--basetemp` to avoid unrelated temp/cache ACL errors. A separate native RTX4060 smoke used existing YOLO26n/ByteTrack and the first 8s of an uploaded video (read-only): 240 frames, two confidence contexts, final JPEG/COMPLETED/no replay. Positive queue dwell was covered by unit tests, not observed in that bounded sample. SQL/runtime/browser integration is verified separately, not as one native end-to-end session.
