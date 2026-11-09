# NAVSIM V1 Cluster Training and Async Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development for disjoint data/distributed/runtime tasks with review. Current server runs bounded Smoke only.

**Goal:** Deliver launchable full-official-V1 two-arm/two-seed training for2×8H200 plus a minimal fixed-period async runtime, verified by honest local Smoke.

**Architecture:** Existing frozen2B/full0.8B six-layer DirectKV; raw native four-image processing; new portable data contract, DDP loop/state, frozen source export and bounded process handoff. No new neural architecture.

**Tech Stack:** PyTorch2.8 DDP/NCCL/Gloo, Transformers5.10.2, existing Qwen/FLA, NumPy/PyYAML, spawn multiprocessing.

- [x] Close previous diagnosis; pin user scope/current singleH800/target16H200 and officialV1 source contract.
- [x] Independent review of this spec and concrete interfaces.
- [x] Official V1 data builder and relocatable dataset: tests first, full-role coverage/filter/14frame eligibility/stats/provenance; prepare no-download Mini Smoke. Source-trained VAL recording group explicitly excluded from selection; full official VAL retained for labelled report only.
- [x] Distributed epoch plan, valid padding/loss scaling, local-rank RNG and complete atomic checkpoint with strict identity: tests first and actual2rank CPU smoke.
- [x] Source bundle exporter plus planner backend using existing models, proper per-model normalization, full Fast scope/native visual buffer and fresh frozen source; authenticated source/state identity, native six-layer KV forward parity (max difference zero), and bounded GPU update tests.
- [x] DDP training/eval/resume loop, twoarm/twoseed launch scripts, configuration/epoch budgets and final staticKV eval.
- [x] Async mailbox/worker/version/clock lifecycle with input-only requests; continuous original images from local prefix and real same-device process Smoke. Complete file-backed CPU transfer avoids 64MiB shm limit; completed plan remains paired with exact source observation/time.
- [x] Full relevant regression and independent code/evidence review; 1014 passed/1 expected skip; 55 cluster tests; native disk resume and separate-process reload prediction max difference zero. Target16-H200/NCCL remains explicit future preflight.
- [x] Chinese handoff, requirements/launch/preflight/data/asset commands, compact numerical evidence and portable source/source-bundle receipts; source/archive/GitHub status supplied in final delivery.
