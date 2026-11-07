# Reader Residual Fusion Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans task-by-task; request independent spec/code/evidence review. User supplied and authorized the design; implement in this isolated worktree without another permission loop.

**Goal:** Test incremental value of the existing readable Slow KV through a small optional residual into a frozen current Fast's original unique head.

**Architecture:** Frozen native no-KV Fast F plus frozen Reader R feed a zero-initialized gated residual. Only fusion parameters train; two arms correct/reference share all initialization/order/budget. Disabled path is the native Fast, not all-masked Reader.

**Tech Stack:** Existing PyTorch/Qwen3.5/FLA environment, original NAVSIM data/cache, CPU pytest, H800 bounded experiment. No downloads.

## Task 1: design, model and semantic tests

- [ ] Review spec `docs/superpowers/specs/2026-11-07-reader-residual-fusion-design.md` independently and fix blockers.
- [ ] Create failing semantic tests in `tests_slowfast/test_reader_fusion.py`: disabled lazy Reader not called; zero residual exact; gradients through frozen head only update fusion; source isolation and current feature conditioning; hook cleanup.
- [ ] Implement `slowfast_v11/reader_fusion.py` using existing head-input hooks to expose Reader features without editing old Reader and a scoped hook to fuse the original Fast head input.
- [ ] Run targeted tests (expected new cases pass and old readability tests unchanged), commit model/tests.

## Task 2: authenticated cache extraction and two-arm training

- [ ] Write `configs/slowfast_reader_fusion.yaml` with fixed sources, architecture and512x8 budget before GPU execution.
- [ ] Implement `tools/train_reader_fusion.py`: authenticate sources; extract full original Fast inputs/features/predictions; separately reproduce all972 historical Reader predictions in batch8/final4, then export the actual fusion bank in batch1 and record geometry differences. Final VAL inference is batch1. Keep GT out of fusion inputs. Unload Slow/rawKV after extraction; retain Fast, Reader and two raw packets through the post-training live check.
- [ ] Train correct/reference arms with equal initialized fusion modules and original frozen head. Save TRAIN32 curves, all per-step loss/grad/LR, identities, checkpoints, actual-last-step replay and all VAL conditions/pairs/per-log summaries.
- [ ] Verify online off/on integration on predeclared TRAIN+VAL tokens with original images, all324 native head identities, frozen states/feature tensor versions and absence of backbone updates. Never run future sources or VAL gradients.

## Task 3: verification, report and bounded publication

- [ ] Independent reload of both arms' full VAL predictions and actual last-step replay; validate SHA/RNG/scheduler/sampler.
- [ ] Request code and scientific evidence review; run applicable complete pytest suites, fixing actual failures.
- [ ] Create portable `evidence/slowfast_reader_fusion/` predictions/curves/hashes and `docs/SLOWFAST_READER_FUSION.md`, link README. Include source-specific gains, fixed-trained control, paired/per-log outcomes and all limits. Do not claim prior0.885 masked Fast is this native baseline.
- [ ] Commit only source/config/docs/portable evidence; preserve local checkpoints/cache and bounded push attempt under existing GitHub authorization. Report actual status and stop after fixed experiment.
