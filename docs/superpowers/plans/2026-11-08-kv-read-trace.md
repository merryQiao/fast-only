# Frozen Direct-KV Read Localization Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development for independently bounded inference helper and review tasks. Do not train or download.

**Goal:** Locate scene-source sensitivity through the six direct-KV attention layers and the existing planning head, with the entire Fast/Slow checkpoint frozen.

**Architecture:** Existing native six-layer planning-only bridge, full-language endpoint, frozen original-resolution visual packets. Observe actual outputs without replacing them, distinguish fixed-Q local counterfactuals from natural propagated differences, and perform single-layer interventions on a fixed metadata-selected subset.

**Tech Stack:** Python, PyTorch 2.8, Transformers 5.10.2, native FLA/TileLang, existing FP32-master/BF16 compute, NumPy bootstrap, Matplotlib.

- [x] Verify existing worktree/checkpoints, user steering, prior metrics and zero-new-download scope; preserve previous branch and all prior artifacts.
- [x] Review this design and the helper interface before implementation.
- [x] Helper: write failing meaningful CPU tests in `tests_slowfast/test_read_trace.py`; implement scoped probability/contribution and fixed-Q/gate/projection instrumentation in `slowfast_v11/read_trace.py`, preserving original return tensors and hook lifetime. Verify tests and review.
- [x] Runner: implement `tools/trace_direct_kv.py`; strict source hashes/restore, original donor map, all256 passive/local probes, 64 metadata-only per-layer intervention cohort, immutable protocol before any GPU inference, context/weights/source versions and old prediction exact equality checks.
- [x] Run targeted regression, then real H800 diagnostics. Keep limits unchanged, log progress, do not add training or change the experiment after looking at results.
- [x] Independently recompute aggregates from per-window records; evaluate whether evidence supports weak reading, gate/projection attenuation, or downstream attenuation without overstating a unique root cause.
- [x] Publish Chinese report/README/plots and lightweight verification evidence, commit; preserve source bundle if existing GitHub authentication remains unavailable.
