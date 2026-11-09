# Split-history Implementation Plan

**Goal:** Deliver explicit Fast1/Slow4 training/runtime with Fast1-only and Fast4-only controls, initialized from pretrained assets without legacy Mini dependencies.

- [x] Lock three-arm input/initialization/training scope and preserve legacy artifacts.
- [x] Independent spec/interface review.
- [x] Frame-aware collator/model/dataset, backwards-compatible defaults, meaningful causality/layout tests.
- [x] Per-image frozen-vision execution/cache, cache-enabled four-frame runtime, CPU tests.
- [x] New protocol configs/source guards/three-arm launcher and bundle-free preflight.
- [x] Fresh-source backend, detached current-plan feedback, train new Slow input modules through live KV.
- [x] Optimizer/source state/buffers/gradient/frozen audit/checkpoint/resume integration.
- [x] Native bounded1/4Fast/cooperative update and separate reload/resume; runtime/cache timing evidence.
- [x] Full regression, independent spec+quality review, migration docs/report.

Release action after these checks: commit and verify the remote branch. The final handoff and Git history record the resulting SHA; model/data artifacts stay outside Git.
