# Frozen Cross-Input Diagnostic Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development for the independent input helper and review tasks; no training or downloads.

**Goal:** Measure Fast image dependence and Slow KV compensation with a frozen, metadata-controlled factorial input intervention.

**Architecture:** Existing full Fast-only and correct-KV endpoints; four frozen vision sources × four native KV sources; preserve target sensors/native layout and existing execution paths. No new neural modules.

**Tech Stack:** PyTorch/Transformers, existing native Processor and frozen vision cache, NumPy recording-paired analysis, Matplotlib.

- [x] Inspect existing endpoints, cache, GPU, user constraints; create isolated sibling worktree from207cdbf.
- [ ] Review the fixed config and design before implementation.
- [ ] Helper TDD: write meaningful failed tests for three bijective cross-log maps, label-independent selection, complete four-frame-only replacement and rejection of mismatching layout; implement `slowfast_v11/cross_inputs.py`; review.
- [ ] Runner TDD: fixed protocol identities/counts/cells, strict restore, canonical baseline/old offset4 equality,64 raw Processor equivalences, full parameter/input immutability; implement `tools/diagnose_cross_inputs.py`.
- [ ] Report TDD: average repeated donors within target window, log-paired effects/bootstrap and factorial interaction, complete ADE/FDE/trajectory metrics, metadata strata and per-log tables; implement `tools/report_cross_inputs.py`.
- [ ] Run CPU regression, actual H800 Fast-only then joint factorial inference; preserve all artifacts and zero-download/zero-training scope.
- [ ] Independently recompute raw predictions and compare all cells/effects; produce Chinese report and plots without forcing a diagnostic pattern or proposing a Head change.
- [ ] Commit lightweight source/evidence; preserve portable numerical/source packages and accurately report GitHub publication status.
