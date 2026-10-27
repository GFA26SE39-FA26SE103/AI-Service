# MF-02 Lane B runtime and fallback runbook

Validated on 27/10/2026 for the Review 2 contract. This runbook owns AI measurements only; the Backend remains authoritative for `OperationalEvent` rule evaluation and persistence.

## Frozen demo contract

- Event/type: `LONG_QUEUE` / `QUEUE_LENGTH`, unit `PEOPLE`.
- Confidence: `0.50`; queue dwell: `5000 ms`.
- Warning `>=3`; Critical `>=5`; sustain `5 s`; recovery `5 s`; cooldown `30 s`.
- One camera, one Zone, full-frame normalized ROI `0,0;1,0;1,1;0,1`.
- Stable non-secret fingerprint: `mf02-lane-b-r2-v1`.
- Recorded playback is non-looping and must end in `COMPLETED`.

## Environment and tests

From `setup/ai-service`:

```powershell
& '..\tools\uv\uv.exe' pip install --python '..\.venv\Scripts\python.exe' -r requirements-dev.txt
& '..\.venv\Scripts\python.exe' -m pytest -q -p no:cacheprovider --basetemp .superpowers/tmp-mf02-tests
```

Validated environment: Python 3.11.16, PyTorch 2.7.1+cu118, CUDA available. The dependency check reported 10 packages satisfied. Final AI suite: `91 passed, 1 skipped, 1 warning`; the skip is the opt-in live-camera test and the warning is documented below.

## Build the local fallback clip

The source footage and derived clip are local-only assets. Do not add them or model weights to Git. From `setup/ai-service`, create an 18-second high-queue segment followed by a 10-second low-occupancy segment:

```powershell
$ffmpeg = '..\tools\ffmpeg\ffmpeg-9.0.2-essentials_build\bin\ffmpeg.exe'
$high = '..\..\Video\hoitruong.mp4'
$low = '..\..\Video\Videos_MERL_Shopping_Dataset\Videos_MERL_Shopping_Dataset\1_1_crop.mp4'
$output = '..\..\Backend\Back-End\src\Supermarket.Api\.local\videos\mf02-lane-b-deterministic.mp4'

& $ffmpeg -hide_banner -loglevel error `
  -ss 0 -t 18 -i $high -ss 0 -t 10 -i $low `
  -filter_complex '[0:v]scale=854:480:force_original_aspect_ratio=decrease,pad=854:480:(ow-iw)/2:(oh-ih)/2,fps=30,setsar=1[v0];[1:v]scale=854:480:force_original_aspect_ratio=decrease,pad=854:480:(ow-iw)/2:(oh-ih)/2,fps=30,setsar=1[v1];[v0][v1]concat=n=2:v=1:a=0[outv]' `
  -map '[outv]' -an -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p $output
```

Expected media: H.264, 854x480, 30 FPS, 28 seconds. Keep the same aspect/crop and re-check the ROI if either source changes.

## GPU smoke

```powershell
& '..\.venv\Scripts\python.exe' -m tools.mf02_recorded_smoke `
  --video '..\..\Backend\Back-End\src\Supermarket.Api\.local\videos\mf02-lane-b-deterministic.mp4' `
  --confidence 0.50 --fingerprint mf02-lane-b-r2-v1 `
  --model '..\yolo26s.pt' --device cuda:0 `
  --roi '0,0;1,0;1,1;0,1' --warning-threshold 3 --critical-threshold 5 `
  --wall-timeout-sec 90
```

Observed sanitized result:

```json
{"batch_count":840,"continuity_count":1,"counts_valid":true,"critical_batches":138,"critical_reached":true,"fingerprint":"mf02-lane-b-r2-v1","fingerprint_consistent":true,"healthy":true,"ok":true,"page_gap":false,"people_max":10,"people_min":0,"queue_max":5,"queue_min":0,"sequence_contiguous":true,"sequence_first":1,"sequence_last":840,"source_elapsed_ms_last":27967,"state":"COMPLETED","warning_batches":246,"warning_reached":true}
```

Exit code `0` requires healthy ordered data and both thresholds. Missing/invalid observations, cursor gaps, non-contiguous sequences, fingerprint mismatch, invalid count relations, or unmet thresholds return a sanitized non-zero result. Video paths, session/owner IDs and exception text are never emitted by the tool.

## Backend bridge procedure and current result

Service order:

1. Start AI with exactly one worker. If Docker already owns 8090, use 8091 and set Backend `AiPreview__BaseUrl=http://127.0.0.1:8091` for that process.
2. Start Backend commit `3470519ab8799ead2a2e37a6c6aef27b8444930b` against a disposable V6-compatible SQL database.
3. Configure one ACTIVE recorded camera, one same-floor mapping/ROI and one ACTIVE Zone configuration using the frozen values above.
4. Confirm start/read uses protocol v1, the same configuration fingerprint/version, `gap=false`, then observe one episode transition `WARNING -> CRITICAL -> CLEARED`.
5. Deactivate the configuration, wait for owner stop, stop both services and remove the disposable database/video.

Fresh Backend contract/evaluator coverage passed 19/19 tests (`AiMonitoringClientTests`, `MonitoringWorkerTests`, `MonitoringIncidentWriterTests`, and `MonitoringEvaluationTests`). This proves payload, gap/missing-data, sustain/recovery and episode logic independently of the real model.

The combined Backend-to-AI database E2E is currently **blocked and not claimed**. The repository's isolated SQL bootstrap files are behind the Backend V6 scaffold: they lack required V6 runtime schema including the `OperationalEventType` catalog (and initially lacked current account/Zone/Camera columns). A disposable SQL Express attempt was stopped and cleaned; the shared Dev database was deliberately not mutated. Re-run steps 1-5 only after the approved V6 database bootstrap/migration is available for an isolated database.

## Resilience evidence

The targeted suite covers owner lease, cursor gap, reconnect continuity reset, recorded EOF, reconfiguration without stale measurements, and disconnect without a synthetic zero batch. A missing frame does not become `people_count=0` or `queue_count=0`; only an actual observed frame may drive recovery.

For an opt-in live camera contract, keep the URL and credentials in environment variables and run:

```powershell
$env:AI_TEST_STREAM_URL = 'http://PHONE_IP:8080/video'
$env:AI_TEST_STREAM_USERNAME = ''
$env:AI_TEST_STREAM_PASSWORD = ''
& '..\.venv\Scripts\python.exe' -m pytest tests/test_ip_webcam_contract.py -q
```

Known non-blocking warnings remain unchanged: Starlette's current TestClient requests `httpx2`, and installed Ultralytics warns that `half` is deprecated in favor of `quantize`. Do not upgrade dependencies solely to hide these warnings.
