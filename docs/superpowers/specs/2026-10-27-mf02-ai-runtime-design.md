# MF-02 Lane B AI Runtime Design

## Intent

Deliver Đặng Nguyễn Phước Lộc's bounded MF-02 lane: prove that the existing Python AI runtime produces trustworthy ordered `people_count` and `queue_count` measurements for one camera and one Zone, remains safe across lease/reconnect/gap/EOF/reconfiguration cases, interoperates with the existing Backend monitoring bridge, and has a repeatable recorded-video fallback for Review 2.

The goal is integration evidence and minimal hardening, not new analytics or a new API.

## Goals

- Reproduce the Python environment from `requirements-dev.txt` and keep the full test suite green.
- Add explicit regression coverage that a disconnect or missing observation emits no synthetic zero measurement.
- Provide a reusable recorded-video smoke command that reports sequence, continuity, fingerprint, people and queue aggregates without printing source paths or secrets.
- Prepare a local deterministic Review 2 clip/config that can drive `LONG_QUEUE` through WARNING, CRITICAL and sustained recovery when consumed by the Backend.
- Validate the existing Backend-to-AI monitoring contract for one camera/Zone without editing Backend-owned bridge code.
- Record exact commands, results, limitations and known warnings for integration and PR review.

## Non-goals and anti-goals

- No new AI/Backend endpoints or DTO changes.
- No Incident-targeting AI, waiting time, density, checkout utilization, FCM/SignalR, Task API or Mobile work.
- No cross-camera identity/deduplication, looped recorded playback or synthetic missing-as-zero behavior.
- No committed camera credentials, stream URLs, model weights, uploaded videos, generated clips, cache files or machine-specific absolute paths.
- No dependency upgrades solely to silence current deprecation warnings.
- No edits to Backend bridge implementation; discrepancies are reported to its owner.

## Constraints

- Contract baseline is the 27/10/2026 MF-02 plan: `LONG_QUEUE`, `QUEUE_LENGTH`, `PEOPLE`, confidence `0.50`, queue dwell `5000 ms`, warning `>=3`, critical `>=5`, sustain `5 s`, recovery `5 s`, cooldown `30 s`.
- OperationalEvent is the only demo outcome and never becomes an Incident.
- AI owns observations only; Backend owns rule evaluation, persistence and business workflow.
- Work occurs on `feature/mf02-ai-runtime` in the existing checkout. No worktree and no merge into `AI-Service`/`dev`/`main`.
- Video artifacts remain local/ignored; only scripts, tests, runbook and evidence may be committed.

## Approaches considered

1. **Minimal hardening plus reproducible evidence — chosen.** Preserve the runtime/API, add one missing regression, a small smoke harness and a runbook, then exercise the real model and existing bridge. This gives a reviewable lane-B commit without expanding the frozen contract.
2. **Evidence-only execution.** Run existing commands and update the shared plan without code. Faster, but leaves no reusable smoke tool and weak branch/PR evidence.
3. **Demo simulator or new runtime endpoint.** Easier to force scenarios, but it would create a second behavior path and expand the contract; rejected.

## Design

### 1. Regression coverage

Extend monitoring-session tests with a controlled reader sequence that emits a valid measured frame, then disconnects. Assert the measurement buffer does not append a zero-valued batch during `RECONNECTING`/unavailable state. Existing tests remain authoritative for owner lease, ordered cursor/gap, reconnect continuity reset, EOF and reconfigure behavior.

### 2. Recorded smoke harness

Add a small CLI under `tools/` that uses the same `CameraReader`, `MonitoringFrameTracker`, `SessionManager` and monitoring schemas as production. Inputs are a local MP4, normalized ROI, confidence and bounded source duration. Output is one sanitized JSON summary containing state, sequence range/contiguity, continuity count, source elapsed time, min/max counts, threshold batch counts and fingerprint. It must never echo the video path, credentials or owner/session tokens.

The harness is validation tooling, not a public API and not a second measurement implementation.

### 3. Deterministic fallback material

Use ignored local source footage to prepare a short MP4 with:

- a sufficiently long high-queue segment to satisfy AI dwell plus Backend sustain for WARNING and CRITICAL;
- a sufficiently long low-queue segment to satisfy Backend recovery;
- one resolution/FPS and no loop.

The runbook records the reproducible FFmpeg recipe and expected measurement envelope. The generated MP4 is not committed.

### 4. Backend bridge validation

Start the existing AI service with one worker and use the existing Backend configuration/monitoring APIs for one recorded camera and one mapped Zone. Capture only non-secret identifiers and the configuration fingerprint. Verify ordered reads, matching fingerprint/config version, no gap in the normal run, and Backend Event open/escalate/clear behavior. If local SQL/auth/demo fixture prerequisites are unavailable, stop without modifying Backend/DB and report the exact blocker; AI-only and contract tests remain separate evidence.

### 5. Documentation and shared-plan update

Document setup, smoke/bridge commands, expected outputs, warning notes and cleanup. Update only lane B checklist lines in the shared Google Doc using a revision guard: mark Doing at start, then Done only for items backed by command output/commit evidence. Do not touch other lanes or the contract baseline.

## Verification

- Full AI suite: expected `82 passed, 1 skipped` or better; the live-camera test remains opt-in.
- Focused new regression fails before implementation and passes after it.
- Real GPU recorded smoke shows contiguous sequences, one continuity epoch during the stable segment, valid `queue_count <= people_count`, and queue values reaching both demo thresholds.
- Sanitization review confirms no sensitive path/URI/token in output or diff.
- Backend bridge/E2E is reported as passed only with fresh runtime evidence; otherwise it remains explicitly blocked/not run.
- Final self-review includes `git diff`, out-of-scope files, tests, generated files and contract preservation.

## Builder decisions left open

- Exact ignored filenames and local locations for generated clips.
- CLI implementation details and JSON field naming, provided they remain backward-independent and sanitized.
- Whether bridge evidence is captured through the running API or an existing focused integration test, based on available local SQL/auth prerequisites.

## Builder-check resolutions

- Local footage is available under the shared workspace. `hoitruong.mp4` has already produced real queue measurements up to 5; a separate low-occupancy local clip is available for recovery. Derived material stays ignored and is not redistributed or committed.
- The checkout has Python 3.11.16, PyTorch 2.7.1 CUDA 11.8, an available CUDA GPU and the local `yolo26s.pt` weight. A real smoke already produced 313 contiguous batches with one continuity epoch.
- The integration target is the current Backend `dev` baseline at `3470519ab8799ead2a2e37a6c6aef27b8444930b`, which is also the plan's pinned Backend commit. Existing local auth/SQL/demo configuration may be used read-only or with rollback-safe demo data; unavailable prerequisites are reported rather than recreated by lane B.

