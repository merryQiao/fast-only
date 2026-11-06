# Bounded KV adaptation and paired diagnostics implementation plan

> **For agentic workers:** Use superpowers:executing-plans / subagent-driven-development with failing tests before implementation and independent review.

**Goal:** Test whether the real six-layer interface can learn on existing tiny NAVSIM data, comparing identical Fast checkpoints with absent, fresh, genuinely older and mismatched KV.

**Architecture:** Keep the accepted model and P0 branch intact. Add a bounded offline runner, causal raw-observation builder and paired reporting. Pretrain tiny planning modules and adapt frozen-Slow-cache consumption; do not implement or claim actual asynchronous closed-loop validation.

**Tech Stack:** Existing Qwen3.5-0.8B/2B, native original-resolution processor, PyTorch/PEFT, one H800, existing 12 TRAIN / 4 VAL mini manifests and metadata.

## Fixed experiment contract

- No download. Preserve 2 Hz, four original CAM_F0 images, native pad-only processing, official Ego/command and 8×3 mean-L1 pose supervision.
- Seed 42. Fixed budgets, no VAL-selected checkpoint: Fast no-KV warmup 80 updates; Slow frozen-language input/head warmup 20; Slow LoRA adaptation 40; Fast frozen-Slow-cache consumption 120. MB1, no augmentation/dropout, AdamW base-LoRA LR1e-4/new LR5e-4, weight decay .01, clip1, constant LR for this diagnostic (not the formal 8k schedule).
- Select TRAIN only for optimizer updates. Keep all 12/4 rows for descriptive paired evaluation, including stationary cases. No TEST access, no VAL-dependent budget or hyperparameter changes.
- Create source observations only from source/current/history raw fields. Their world poses come from raw ego2global poses already observed at those times, never from future GT. Source history requires four distinct existing files and consistent 2 Hz timestamps, log, navigation command.
- Compare source observation ages 0, .5, 1 and 1.5 seconds using actual timestamps with .001s tolerance. Missing history/navigation mismatch is counted and excluded, never replaced with a fresh cache. Report each eligible subset and its paired no-KV baseline.
- Fresh KV is an offline diagnostic with zero observed age, not deployable zero-latency Slow. Older KV uses real historical images/poses but is not an actual asynchronous-delivery measurement. No invented ready times or H1 delivery claims.
- Generate numerical feedback using the completed Fast-warmup predictor on the corresponding source input. Freeze this feedback-producer version for all subsequent Slow and cache diagnostics; never insert expert trajectories as feedback. Record feedback-source checkpoint hash.
- Train Slow on its own TRAIN trajectory mean L1. Stage A freezes language parameters but preserves autograd; Stage B opens existing language LoRA. Export immutable detached source snapshots after Slow training. No Slow updates during Fast consumption.
- Cache-consumer schedule repeats 10-step blocks: 3 absent, 4 fresh, 3 stale. Absent/fresh cycle all TRAIN rows; stale cycles explicitly eligible TRAIN pairs only; log actual exposures. No silent missing-pair fallback.
- Paired mismatched-KV diagnostic swaps only external KV/positions, keeping target input and target metadata fixed. Prefer another TRAIN log; VAL has only one log, so require another token at least 2s away, deterministic derangement. If impossible, report unavailable. This is deliberately invalid offline content injection, never an online admission policy.
- Report TRAIN and VAL separately before/after consumer adaptation: ADE/FDE/heading MAE/L2 horizons, per-row predictions, fresh/shuffled changes, and stale/no-KV/fresh on identical eligible rows. Include current-velocity baseline, moving/stationary counts. Four VAL rows from one log cannot establish generalization or closed-loop benefit.
- Save final fixed-budget checkpoint (trainable state, optimizer/scheduler if present, RNG, stage/step and sampler cursors) and prove independent-process reload on held-out VAL predictions. Do not claim exact mid-training continuation without an executed continuation check.

### Fairness amendment before inspecting paired results

Add a matched Fast-update-budget control: restore the exact Fast-warmup adapter/new-module state and train 120 further no-KV updates using the same target-token order as the consumer plan, with a reset optimizer and identical LR/loss. Each Fast path therefore has 200 total updates; shared warmup is counted once operationally. Main run remains 260 updates; the separate control adds 120. Slow's extra 60 supervised updates are reported separately, so this still does not establish equal total training compute or the full A1 auxiliary-supervision ablation. No control or primary budget is chosen using VAL metrics.

## Task 1: Causal diagnostic data

Files: `slowfast_v11/diagnostic_data.py`, `tests_slowfast/test_diagnostic_data.py`.

- [x] Write failing tests for raw current/history-only observation, relative pose, real timestamps, missing histories, navigation mismatch and split-preserving deterministic mismatches.
- [x] Implement observation/pair construction and count exclusions. Replacing future annotations/images must not change source input or metadata.
- [x] Verify tests and scan existing mini files before training. Record supported ages and scene/motion limitations.

## Task 2: Adaptation and reporting primitives

Files: `slowfast_v11/validation.py`, `tests_slowfast/test_validation.py`, `configs/slowfast_v11_validation.yaml`.

- [x] Test bounded config, TRAIN-only updates, stage trainability/gradient behavior, fixed sampling plan, and paired-subset metrics on real tiny native models.
- [x] Implement fixed-budget warmup/Slow/cache-consumer loops and strict config. Keep existing model implementation unchanged.
- [x] Implement paired no-KV/fresh/shuffled/stale evaluation and predictions; avoid label-derived inputs and report coverage rather than impute missing cases.
- [x] Verify checkpoint packing/restoration and held-out prediction reference.

## Task 3: Real run, review and publication

Files: `tools/validate_slowfast_v11.py`, `tools/reload_slowfast_validation.py`, `docs/SLOWFAST_V11_VALIDATION.md`, `evidence/slowfast_v11_validation/*.json`.

- [x] Run real native-resolution, fixed-budget stages on one H800; log loss, exposures and memory. Do not tune using VAL or extend after looking at results.
- [x] Execute fresh-process reload and verify held-out predictions. Report numerical facts whether KV helps, hurts or is ignored.
- [x] Run the predeclared matched-budget no-KV control; report all 12/4 predictions and both paths' data/order identity.
- [x] Independently review source and interpretational limits, run affected tests/regressions, and retain original Fast-only source unchanged.
- [x] Commit and push separate validation branch with portable JSON evidence/docs only; keep raw data, caches and checkpoints local/ignored.

## Execution record

2026-11-06: Completed fixed main260 + matched-control120 real updates on one H800. All495 tests passed; independent reviews passed. Fresh-process reload reproduced13 VAL conditions exactly. No inference-time XY benefit from correct KV established: same-checkpoint VAL ADE none3.1553/fresh3.2426/shuffled3.2481m. Matched no-KV control3.7618m indicates a training-regime difference, not proof of online KV utility. CV0.3503m and tiny coverage limit interpretation. No full-data download, test use, VAL checkpoint selection or budget extension after results. Original model/attention code unchanged. Results published separately from accepted main.
