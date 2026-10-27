# MF-02 Lane B AI Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce reusable, sanitized evidence that the existing AI runtime supplies ordered queue measurements and a deterministic Review 2 fallback through the frozen Backend bridge contract.

**Architecture:** Keep production measurement behavior unchanged. Add one explicit disconnect regression, a thin CLI that composes existing runtime classes and summarizes their output, and a runbook/evidence record for real GPU and Backend bridge checks.

**Tech Stack:** Python 3.11, FastAPI/Pydantic, OpenCV, Ultralytics YOLO/ByteTrack, pytest, existing ASP.NET Core monitoring bridge.

**Spec:** `docs/superpowers/specs/2026-10-27-mf02-ai-runtime-design.md`

## Global Constraints

- Demo contract: `LONG_QUEUE`, `QUEUE_LENGTH`, `PEOPLE`, confidence `0.50`, queue dwell `5000 ms`, warning `>=3`, critical `>=5`, sustain `5 s`, recovery `5 s`, cooldown `30 s`.
- Keep protocol v1 and every existing endpoint/DTO unchanged; OperationalEvent never becomes Incident.
- Do not edit Backend bridge implementation, Task/Mobile code, SQL schema or generated persistence code.
- Do not commit video, model weights, credentials, stream URLs, caches, build output or absolute machine paths.
- Work only on `feature/mf02-ai-runtime`; do not merge/rebase/force-push.
- Generated fallback video is local and non-looping; EOF remains `COMPLETED`.

## Review Focus

- A disconnect or missing frame must append no synthetic zero batch: covered in Task 1.
- A cursor/session/sequence gap must be explicit rather than summarized as continuous: covered in Task 2.
- CLI success and failure output must not disclose the video path, owner/session IDs or credentials: covered in Task 2.
- EOF/reconfiguration must not replay or serve old measurements: existing tests re-run in Tasks 1 and 3.
- Footage that does not reach warning/critical thresholds must return an explicit unmet result/non-zero exit: covered in Task 2.

---

### Task 1: Missing-observation regression

**Files:**
- Modify: `tests/test_monitoring_sessions.py`

**Interfaces:**
- Consumes: `SessionManager.start_monitoring(...)`, `measurements(...)`, `SessionState.RECONNECTING`, `MeasurementPage.batches`.
- Produces: regression `test_disconnect_emits_no_synthetic_zero_measurement`.

- [ ] **Step 1: Add the regression test**

Create one successful frame followed by a disconnect with a non-zero `MonitoringFake` measurement. Pause the reconnect with a bounded delay, read measurements during `RECONNECTING`, and assert the page contains only the original batch and no generated `people_count=0`/`queue_count=0` batch.

- [ ] **Step 2: Run the focused test**

Run: `..\.venv\Scripts\python.exe -m pytest tests/test_monitoring_sessions.py::test_disconnect_emits_no_synthetic_zero_measurement -q -p no:cacheprovider --basetemp <unique-workspace-temp>`

Expected: PASS if existing behavior is correct. If it fails, make the smallest `app/session_manager.py` correction that prevents append-on-read-failure; do not synthesize zero or change reconnect semantics.

- [ ] **Step 3: Run related resilience tests**

Run the full `tests/test_monitoring_sessions.py`, `tests/test_measurement_buffer.py`, `tests/test_source_clock.py`, `tests/test_roi_measurements.py`, and `tests/test_recorded_video.py` set.

Expected: all pass.

- [ ] **Step 4: Commit**

Commit: `test: prove missing observations do not emit zero`

---

### Task 2: Sanitized recorded-video smoke CLI

**Files:**
- Create: `tools/mf02_recorded_smoke.py`
- Create: `tests/test_mf02_recorded_smoke.py`

**Interfaces:**
- Consumes: `CameraReader`, `MonitoringFrameTracker`, `SessionManager`, `StartMonitoringRequest`, `MonitoringZoneInput`, `MeasurementBatch`.
- Produces: `summarize_batches(batches, state, fingerprint, page_gap, warning_threshold=3, critical_threshold=5) -> dict[str, object]`; async CLI runner accepting `--video`, `--duration-sec`, `--confidence`, `--fingerprint`, `--model`, `--device`, and normalized `--roi`.

- [ ] **Step 1: Write failing summary/CLI tests**

Test names and required assertions:

- `test_summary_reports_contiguous_threshold_measurements_without_identifiers`: contiguous sequences and counts produce warning/critical flags, one continuity count, and JSON without session/owner/video values.
- `test_summary_marks_gap_or_noncontiguous_sequence_unhealthy`: `page_gap=True` or skipped sequence makes `healthy=False`.
- `test_cli_reports_unmet_threshold_without_echoing_video_path`: mocked runtime below threshold exits non-zero and output contains neither the supplied path nor raw identifiers.

- [ ] **Step 2: Run tests to verify they fail**

Run: `..\.venv\Scripts\python.exe -m pytest tests/test_mf02_recorded_smoke.py -q -p no:cacheprovider --basetemp <unique-workspace-temp>`

Expected: FAIL because the tool does not exist.

- [ ] **Step 3: Implement the thin CLI**

Use the production reader/tracker/session manager; do not duplicate detection or queue logic. Poll ordered pages until source duration, EOF/error or wall timeout. Aggregate only sanitized values, set `healthy` from page-gap/sequence/count invariants, and exit non-zero when healthy data does not reach warning and critical thresholds. Catch errors into a stable code without printing exception text/path.

- [ ] **Step 4: Run focused and full AI tests**

Run the Task 2 test file, then `..\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp <unique-workspace-temp>` outside the restricted event-loop environment.

Expected baseline: at least `82 passed, 1 skipped`, plus the new tests; no failures.

- [ ] **Step 5: Commit**

Commit: `feat: add sanitized MF-02 recorded smoke tool`

---

### Task 3: Deterministic fallback and bridge evidence

**Files:**
- Create: `docs/MF02_LANE_B_RUNBOOK.md`
- Modify: `README.md`
- Modify: shared Google Doc lane B lines only after evidence exists.

**Interfaces:**
- Consumes: Task 2 CLI, local ignored footage, FFmpeg, AI service protocol v1, Backend `dev` commit `3470519ab8799ead2a2e37a6c6aef27b8444930b`.
- Produces: repeatable clip recipe, sanitized AI smoke result, bridge commands/config fingerprint and pass/blocker record.

- [ ] **Step 1: Prepare the ignored deterministic clip**

Create one normalized MP4 from a high-queue segment long enough for 5-second AI dwell plus 5-second Backend sustain, followed by a low-occupancy segment at least 5 seconds long for recovery. Keep the source and generated file outside Git.

- [ ] **Step 2: Run the real GPU smoke**

Run Task 2 CLI with confidence `0.50`, full-frame or documented normalized ROI, required warning `3`, critical `5`, and a stable non-secret fingerprint.

Expected: healthy contiguous measurements, `queue_count <= people_count`, warning and critical reached. Record the summary only.

- [ ] **Step 3: Validate the existing Backend bridge**

Run the AI service with one Uvicorn worker and the existing Backend at the pinned commit. Use one recorded camera and one mapped Zone/config. Verify start/read requests, matching fingerprint/config version, normal-run `gap=false`, and Event WARNING → CRITICAL → CLEARED on the same episode. Do not modify Backend code or shared schema; use existing/rollback-safe fixture data.

If auth/SQL/demo prerequisites are unavailable, record the exact missing prerequisite and leave the bridge/E2E checklist unticked.

- [ ] **Step 4: Write runbook and evidence**

Document environment setup, clip recipe, smoke command, service startup order, non-secret config values, observed result, cleanup, live-camera opt-in command, and known warnings (`httpx2`/Ultralytics `half`) without upgrading dependencies.

- [ ] **Step 5: Verify docs and repository scope**

Run: `git diff --check`, full AI tests, `git status --short`, and secret/path searches over changed files. Confirm no video/model/cache or Backend file is staged.

- [ ] **Step 6: Commit and update shared plan**

Commit: `docs: add MF-02 lane B integration runbook`

Update only completed lane B checklist items with commit/test evidence using the current Google Docs revision guard. Push the lane branch and mark it ready, but do not merge.

