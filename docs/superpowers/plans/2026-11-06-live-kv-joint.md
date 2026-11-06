# Stronger Slow adaptation and live-KV joint validation plan

> **For agentic workers:** Use test-driven implementation and independent review through superpowers:subagent-driven-development / executing-plans. User approved executing the reference-audit training recommendations.

**Goal:** Establish whether stronger task adaptation plus sustained Fast-loss→Slow-KV training changes useful KV consumption on the existing mini cohort.

**Architecture:** Preserve both full Qwen backbones, six-layer KV fusion, original images and the unique trajectory Head. Add a separate bounded trainer with three controlled gradient routes, shared data/model harness and new schema. No full dataset download, no Future/refresh architecture changes.

**Tech stack:** Existing verified PyTorch/PEFT/FLA runtime, single H800, original 1920×1080 four-image NAVSIM observations, 12 TRAIN /4 VAL.

## Predeclared protocol

- Initialize Fast from the previous verified 80-step warmup state and Slow from its 20+40-step state, both contained in the prior diagnostic final checkpoint. Verify checkpoint SHA, model assets, data identities and feedback-producer identity. This is diagnostic continuation, not formal-training initialization.
- TRAIN only for optimizer updates. No TEST access; no VAL selection, threshold tuning, early stopping or budget extension based on results.
- Shared Slow adaptation: **240 updates**, LoRA+new modules, existing trajectory mean L1. Keep Fast fixed. Retain the same detached numerical feedback from the saved Fast-warmup producer to isolate subsequent gradient routing; do not substitute expert labels as feedback. Feedback refresh/missing-feedback training remain deferred and explicit.
- Then run **three arms, each 240 paired updates**, from exactly the same Fast-warmup and strengthened Slow states: `no_kv`, `detached`, `joint`. Every arm trains Slow with its own trajectory loss on every step, including `no_kv`; that unused auxiliary compute equalizes Slow training opportunity.
- `no_kv`: Fast never reads Slow; `detached`: fresh KV with stop-gradient; `joint`: fresh live KV preserving Fast-loss→Slow. Detached and joint use exactly 30% no-KV /70% fresh conditions; no stale training yet.
- Fixed balanced TRAIN target sequence: moving speed≥1m/s and low-speed each 50%, no replacement within each stratum's shuffle cycles. Modes are independently shuffled per stratum so both have exactly30% absent. All arms have identical target order; detached/joint have identical modes. With240 steps:120 moving/120 low-speed,72 absent/168 fresh in reading arms.
- Objective per arm: `L_F + 1.0*L_S`, existing unweighted mean L1 pose. Separate AdamW per model, LoRA LR1e-4/new LR5e-4, weight decay.01, constant LR, **clip each model separately at1**. Separate clipping prevents Fast gradient magnitude from modifying Slow-only updates in the detached/no-KV controls.
- Architecture/LoRA rank16/alpha32/dropout0/float32 adapter masters/frozen vision-merger unchanged. Processor stays native pad-only. Seed42, strict deterministic settings, no augmentation.
- Probe Fast-loss-only gradients on four predeclared fresh-step quantiles: each Slow K and V separately, plus paired Slow K/V LoRA-B parameters. Joint must have finite nonzero gradients; detached and no-KV must have no such gradient. Do not mistake Slow's own-loss gradient for consumer alignment.
- Evaluate TRAIN-only curves at steps60/120/240; evaluate VAL only at each arm's final fixed budget. Compare same-checkpoint absent/fresh/shuffled, plus real-old subsets descriptively where images exist. Report coverage and moving/low-speed results; tiny VAL cannot establish generalization or asynchronous delivery benefit.
- Check no-KV and detached Slow final states for equality: with identical Slow inputs/loss/optimizer and separate clipping, they should agree under deterministic computation. Joint changes may originate from Fast supervision; no performance guarantee follows from this check.
- Save per-arm final model/optimizer/RNG/plan progress. Also save before/after one predeclared late fresh joint step; execute independent-process prediction reload and actual interrupted-step replay. Do not claim exact continuation unless replay evidence passes.
- Operational budget:240 shared Slow updates +3×240 paired iterations=960 iterations;1680 new optimizer.step calls (960 Slow+720 Fast). Each Fast arm has inherited80+240=320 total diagnostic updates. Each Slow arm has inherited60+240+240=540 module updates, LoRA40+240+240=520. Report shared/inherited budgets distinctly.

## Task 1: Plan and config

Files: `slowfast_v11/joint_protocol.py`, `configs/slowfast_v11_joint.yaml`, `tests_slowfast/test_joint_protocol.py`.

- [x] Observe failing tests for bounded schema, balanced TRAIN-only sampling, class-independent cache masks and fixed gradient-probe steps.
- [x] Implement strict config and deterministic plans, rejecting unsupported policy drift and empty strata.
- [x] Review protocol before real training.

## Task 2: Joint optimization

Files: `slowfast_v11/joint_training.py`, `tests_slowfast/test_joint_training.py`.

- [x] Test real tiny native models: live Fast-loss gradients, absent detached-control gradients, Slow own-loss updates and separate clipping/equal Slow states.
- [x] Implement one paired update and Fast-only gradient probes without modifying backbone/fusion code.
- [x] Verify finite gradients, frozen vision and feedback detach; no unused inference snapshot in joint mode.

## Task 3: Runner and restore

Files: `tools/train_slowfast_joint.py`, `tools/reload_slowfast_joint.py`; small backward-compatible harness injection in `tools/validate_slowfast_v11.py`.

- [x] Add constructor injection for new config/plan while preserving original default behavior.
- [x] Implement parent identity checks, stronger Slow phase, three arms, scheduled TRAIN curves and final paired VAL diagnostics.
- [x] Store both optimizer states, RNG, initial/final identities and sampler position; save the selected joint replay boundary.
- [x] Implement fresh-process reload and real next-step replay, comparing parameters, moments, predictions and input/order identities.

## Task 4: Real validation and delivery

Files: `docs/SLOWFAST_V11_JOINT.md`, selected portable `evidence/slowfast_v11_joint/`.

- [x] Run real original-resolution fixed budgets; log gradients, loss, exposures and memory. No extra data download.
- [x] Execute independent reload/replay and appropriate regression tests; obtain independent review.
- [x] Report observations without assuming KV utility. Keep checkpoints/data local; commit reviewable code/report branch, attempt authorized GitHub publication without blocking research on unavailable credentials.
