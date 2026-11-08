# Direct KV Full-Language Training Plan

> **For agentic workers:** Use superpowers:subagent-driven-development with independent data/full-state implementation and design/code/evidence reviews. The user supplied the architecture and authorizes execution; do not repeat permission questions for routine implementation decisions.

**Goal:** Actually expand TRAIN and compare Fast-only/correct-direct-KV/fixed-direct-KV with the complete Fast language backbone trainable.

**Architecture:** Existing complete Fast24layer hybrid backbone, frozen native vision+merger and Slow, unchanged6-layer planning-only directKV bridge, single existing trajectory head. Full FP32 master parameters, native BF16 language compute with FP32head/LoRA.512TRAIN12groups/newVAL2568groups, same4096drawbudget.

- [ ] Review spec and pin config/normalizer/budget/data-role decisions before new evaluation.
- [ ] Data task: deterministic512TRAIN promotion and8freshVAL/finite12pose/groups; private320->520MB metadata,2GB newcap, frozenplan-beforecamera, realexpandedstats audit and preserved normalizer; TDD/provenance.
- [ ] Full-state task: configure real full-language scope, precision policy, all-base-weights checkpoint/restore/optimizer groups, CPU semantic tests; existing hooks stay noncheckpointed.
- [ ] Root P0: actual H800 original-resolution full forward/backward/AdamW update with no/fresh/fixedKV; print trainable names/counts, perblock base grads/deltas andfrozenvision; measurememory/time. Reject silently falling back toLoRA or lower resolution.
- [ ] Root cache/runner: authenticated original native vision/SlowKV exports, same causalFast512teacher, host/NVMe source cache and fullgradient consumer inputs, same per-arm init/order/schedule, 1024xacc4microB1 train withTRAIN64curves.
- [ ] Full-state actualresume/reload andsameVAL three-arm/legacyfixedAdapter evaluation; pairedlog effects/bootstrap with one-seed scope, independent review and complete relevanttests.
- [ ] Chinese report/README/portableevidence/hash+bytes ledger; stop residual tuning; commit only lightweight artifacts and accurately report publication status.
