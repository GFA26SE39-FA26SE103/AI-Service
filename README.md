# S.H.E.P.H.E.R.D AI Preview Service

Private native Windows service for the MF-01 camera preview smoke test. It reads an IP Webcam stream, runs Ultralytics YOLO person detection with ByteTrack, and exposes only the latest annotated JPEG to the ASP.NET backend.

The React frontend must never call this service or the phone camera directly. This service does not write detections, operational events, incidents, or images to the database.

## Local runtime

- Python: `C:\FPT University\CAPSTONE\setup\.venv\Scripts\python.exe`
- Model: `C:\FPT University\CAPSTONE\setup\yolo26n.pt`
- FFmpeg: portable build under `C:\FPT University\CAPSTONE\setup\tools\ffmpeg`
- Default API: `http://127.0.0.1:8090`

PyTorch CUDA is installed separately from `requirements.txt` so a normal package install cannot silently replace it with a CPU-only wheel:

```powershell
$python = 'C:\FPT University\CAPSTONE\setup\.venv\Scripts\python.exe'
& $python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
& $python -m pip install -r 'C:\FPT University\CAPSTONE\setup\ai-service\requirements-dev.txt'
& $python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Copy `.env.example` to `.env` for local overrides. Do not commit camera URLs, usernames, passwords, JWTs, or the internal service key.

Start the private service from this directory:

```powershell
& '..\.venv\Scripts\python.exe' -m uvicorn app.main:app --host 127.0.0.1 --port 8090
```

Internal contract:

- `GET /health`
- `POST /sessions/{cameraId}/start`
- `GET /sessions/{cameraId}/status`
- `GET /sessions/{cameraId}/frame`
- `DELETE /sessions/{cameraId}`

Session states are `STOPPED`, `STARTING`, `LIVE`, `RECONNECTING`, and `ERROR`. Frame reads use `409 AI_PREVIEW_NOT_RUNNING` or `503 AI_FRAME_NOT_READY` when appropriate. Unexpected errors return only `AI_PREVIEW_FAILED`; camera credentials and authenticated stream URLs are never echoed.

## IP Webcam prerequisite

Put the phone and laptop on the same LAN. Start IP Webcam and use the exact MJPEG URL advertised by the app, commonly `http://PHONE_IP:8080/video`. Validate the URL in a browser before starting an AI session.
