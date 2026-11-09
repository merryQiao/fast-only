# NAVSIM V1 Cluster Training and Async Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development for disjoint data/distributed/runtime tasks with review. Current server runs bounded Smoke only.

**Goal:** Deliver launchable full-official-V1 two-arm/two-seed training for2×8H200 plus a minimal fixed-period async runtime, verified by honest local Smoke.

**Architecture:** Existing frozen2B/full0.8B six-layer DirectKV; raw native four-image processing; new portable data contract, DDP loop/state, frozen source export and bounded process handoff. No new neural architecture.

**Tech Stack:** PyTorch2.8 DDP/NCCL/Gloo, Transformers5.10.2, existing Qwen/FLA, NumPy/PyYAML, spawn multiprocessing.

- [x] Close previous diagnosis; pin user scope/current singleH800/target16H200 and officialV1 source contract.
- [ ] Independent review of this spec and concrete interfaces.
- [ ] Official V1 data builder and relocatable dataset: tests first, full-role coverage/filter/14frame eligibility/stats/provenance; prepare no-download Mini Smoke.
- [ ] Distributed epoch plan, valid padding/loss scaling, local-rank RNG and complete atomic checkpoint with strict identity: tests first and actual2rank CPU smoke.
- [ ] Source bundle exporter plus planner backend using existing models, proper per-model normalization, full Fast scope/native visual buffer and fresh frozen source; real source parity and bounded GPU update tests.
- [ ] DDP training/eval/resume loop, twoarm/twoseed launch scripts, configuration/epoch budgets and final staticKV eval.
- [ ] Async mailbox/worker/version/clock lifecycle with input-only requests; continuous original images from local prefix and real same-device process Smoke.
- [ ] Full relevant regression and independent code/evidence review; explicit current-vs-target validation limits.
- [ ] Chinese handoff, requirements/launch/preflight/data/asset commands, compact numerical evidence and portable source/source-bundle receipts; GitHub status accurate.
