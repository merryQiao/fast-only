# Adapter Baseline and Paired New-Log Study Plan

> **For agentic workers:** Use superpowers:subagent-driven-development for independent data/bundle tasks and independent spec/code/evidence review. User supplied the design direction; do the bounded implementation/experiment without another permission loop.

**Goal:** Deliver a real no-Slow fixed-feature adapter and measure matched-Slow value over it on eight new drive groups with three paired fusion seeds.

**Architecture:** Existing frozen residual experiment, same TRAIN256 and512x8 budget; export constantR0 baseline, stream new original-image model features, evaluate once on locked new logs, paired log/seed bootstrap.

**Tech Stack:** Existing Qwen/FLA PyTorch environment, pinned OpenScene mini bounded Range acquisition, CPU pytest and H800; no new model weights/full dataset.

- [ ] Independently review spec and lock threshold/config/cohort protocol before model evaluation.
- [ ] Data task: failing selection/leakage/budget tests, minimal `slowfast_v11/adapter_data.py` and `tools/prepare_adapter_study.py`, fixed plan/hash/provenance, <=2GB new bodies,8x32 windows or explicit failure.
- [ ] Baseline task: failing constant-bundle parity/bypass tests, `slowfast_v11/fixed_adapter.py`, authenticated export/load helpers; no Reader import/inference dependency.
- [ ] Root extraction: `tools/train_adapter_study.py`, authenticate old feature bank/teacher/latest models/raw layout; new holdout only native batch1 features, no future inputs or new TRAINstats; preserve exact seed42 old behavior.
- [ ] Train six512x8 fusion arms, log loss/gradient/LR/order/initial state, export fixed bundles, actual checkpoint final-update replays and full prediction reload for all seeds. Actual original-image fixed-bundle inference with no Slow/Reader instantiated.
- [ ] Analysis/report: `tools/report_adapter_study.py`,3x8 effects, declared paired bootstrap/2cm gate, raw four conditions, per-log/speed and old diagnostics, cumulative/new download bytes, independent review.
- [ ] Run relevant full tests, verify artifact/source hashes and links, mark actual outcome without tuning, commit lightweight artifacts; report GitHub credential status accurately.
