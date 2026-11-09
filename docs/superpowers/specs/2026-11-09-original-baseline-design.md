# Original Fast-only initialization without historical Slow artifacts

User correction: starting the baseline on full NAVSIM V1 must not require the old Mini experiment's source bundle. This correction is independent of whether the next experiment also initializes a fresh Slow arm.

Fast-only loads the original Qwen3.5-0.8B base, initializes its existing planning modules with the declared seed, and uses current TRAIN-only Ego statistics. It never loads or hashes a Slow model, a feedback teacher or source_bundle.pt. Existing six-layer-compatible Fast architecture, original-image processing, loss, optimizer, epochs and DDP/resume rules stay fixed.

Add explicit baseline-only formal/Smoke configurations with one arm, source_feedback=none, no Slow asset placeholders, full_train_fast normalization, and official default TRAIN/VAL. Since no upstream task-adapted model is used, the six-log source-contamination exclusion does not apply to this new protocol. Official preparation must expose an explicit way to retain the official VAL; old bundle-backed runs keep their exclusion rules and existing configuration identity.

Preflight and the baseline launch command depend only on the Fast base and prepared data. Fast-only inference verifies only its recorded Fast assets. A checkpoint from the old protocol remains an old experiment and must not silently resume under the new semantic identity.

Verification: reproduce the current unconditional load_source failure in a small constructor regression; config accepts baseline-only without Slow files; source guards distinguish new official VAL from legacy exclusion; native one-update Smoke with source/Slow paths absent; relevant regressions and remote commit verification. No full-data download or full training here.

This originally isolated dependency fix is now complete. The subsequent explicit Fast1/Slow4 request supersedes the pending fresh-Slow choice: see `2026-11-09-split-history-design.md` for the announced initialization seed and frozen-backbone/train-new-modules policy. This document remains the independent four-frame Fast-only protocol.
