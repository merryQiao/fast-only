# Original Baseline Dependency Fix Plan

**Goal:** Start full-data Fast-only from original Qwen3.5-0.8B without any previous task checkpoint.

**Architecture:** Keep the existing planner/DDP loop. Make source assets conditional on an actual Slow arm; provide a separate, explicit baseline-only protocol. Preserve legacy experiment configuration identity where possible and reject semantic resume changes.

- [x] Reproduce the unconditional source load and config rejection with focused tests.
- [x] Review the bounded Fast-only design; fresh-Slow decisions subsequently locked by the split-history spec.
- [x] Make Fast-only model construction and inference independent of source_bundle/Slow assets.
- [x] Add baseline-only config validation, formal/Smoke YAMLs and official default VAL guard.
- [x] Provide baseline launch/preflight and explicit official-data preparation flag.
- [x] Run CPU regression and one real native update with no source/Slow assets in the config.
- [x] Update migration instructions and review the corrected entry.

Released together with the later split-history update; its release operation commits these files and verifies the remote branch. See the split-history plan and Git history.
