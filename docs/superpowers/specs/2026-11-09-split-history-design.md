# Fast single-frame / Slow four-frame protocol

User-approved direction: align planning with low-frequency history processing, retain four-frame Fast as a strong control, and remove mandatory legacy Mini task checkpoints. This supersedes the four-frame collaborative Fast default. Current host runs bounded Smoke only; target is full official NAVSIM V1,2nodes×8H200.

## Experiments and initialization

New schema `navsim_split_history_v2` trains three arms, each seeds42/43 and8epochs: `fast_single` (Fast1,no Slow), `fast_history` (Fast4,no Slow), `slow_fast` (Fast1,Slow4). All use identical official TRAIN/VAL scenes and raw full-resolution front images. Real source frequency remains2Hz. Shared trajectory format8×3 and official L1 loss, six Direct-KV layers, fullFast0.8B and unique Fast trajectory Head stay unchanged. Legacy configs/checkpoints remain explicit old experiments.

Both models start from original local pretrained Qwen assets. Slow vision/language incl LoRA stay frozen; only existing new Ego/command/feedback encoders, feedback/missing embeddings and Planning Queries train through Fast loss. Slow trajectory Head remains unused/frozen; no new auxiliary/Future losses. Fast language/new modules train, vision+Merger frozen. Source initialization seed142 is explicit; no old feedback teacher or source bundle. All new modules and both models use current TRAIN statistics. This frozen-backbone/new-input adaptation choice is stated to the user and in config.

For a correct synchronous training source, run the same Fast's no-KV candidate under no_grad, detach its numeric plan, prefill Slow on the same four-frame observation, then run Fast1 with live KV. Frozen Fast vision is computed once per microbatch and reused across candidate/final forwards. Preserve Slow-new-module gradients through frozen Slow language; do not wrap that whole prefill in no_grad. Prototype KV dropout0.3, feedback dropout0.1 are explicit TRAIN sampling; evaluation disables both. All ranks still execute compatible DDP accumulation, including unused source modules on missing-KV cases.

Source-new trainable state and native buffers join full-state checkpoint/save-best/resume. Original model assets, source init seed/config, task initialization digest, per-model normalization, frame policy and trained scope join identity. Frozen audits cover only frozen parameters/buffers; source gradient probes separately show learning scope. Legacy checkpoint identity must remain stable; new protocol cannot silently resume an old four-frame/source-bundle run.

## Frame boundary and frozen visual cache

`NavsimCollator` supports exactly `(0.,)` or existing `(-1.5,-1.,-.5,0.)`, with unchanged original_pad32/no-resize. `FastWithSlowKV(image_frames=1|4)` validates native grid accordingly; Slow remains4. Old calls default4 and preserve exact prompt/state shape. Manifests always retain official four-history/ten-future eligibility and eight targets. Dataset can decode only the latest image for Fast-only1, but cooperative training reads4 for Slow. Future metadata never enters native model inputs.

New Fast protocol uses native per-image frozen ViT execution during training and inference; native placeholder/mRoPE assembly remains unchanged. This fixed execution policy lets four-frame inference cache each historical RGB/visual feature while preserving cache-on/off math. Four-frame language context is still processed in full each planning round. Cache is bounded, keyed by real image identity plus frozen provider version, and invalidated on provider/buffer changes. Single-frame Fast must decode/encode exactly one current frame, not four followed by slicing GPU tokens. No persistent language/GDN cache is introduced.

## Runtime

Existing spawn Slow and complete CPU-file-mmap transport remain. Slow worker loads its trained new modules from the same cooperative checkpoint, no external task bundle. Fast performs current vision first, then fixes one fully published cache version. Full transfer/synchronize precedes ready time; s<=t,r<=c and episode/nav/model checks remain. Slow requests retain the completed candidate's exact source observation/time. No action matching or adaptive refresh implementation is added.

Runtime benchmarks support the two Fast-only arms and a sync fresh-KV reference; existing fixed-period async uses the same cooperative weights. Measure decoding/Processor/vision/language+Head/transfer and source-to-submit latency, actual Slow compute and snapshot bytes. Four-frame baseline cache is enabled in comparison. Never infer4× speed from token count. Current singleH800 colocated result is functional evidence only, not16H200 or separate-GPU performance/quality.

## Locked interfaces and implementation ownership

- Root: `fast_cluster/models.py`, trainer/distributed source-state integration, source checkpoints, new smoke evidence/docs.
- Frame task: `fast_navsim/collate.py`, `slowfast_v11/model.py`, `fast_cluster/data.py`, focused frame/data tests. Dataset/PreparedData `dataset(split,image_frames=4)` defaults unchanged; cooperative reads4, fast_single reads1. Model Fast `image_frames` argument default4; Slow remains4.
- Cache/runtime task: new `fast_cluster/vision_cache.py`, `fast_cluster/async_runtime.py`, `tools/run_cluster_async.py`, new `tools/benchmark_split_history.py`, tests. Generic `FrozenFrameCache` provides RGB/feature caching and counters; `encode_frames(backbone,native,cache=None,keys=None)` returns tuple of native per-image pooler tensors. Runtime adds `config['runtime_checkpoint']=absolute_checkpoint` for worker source restoration. Factories signatures unchanged.
- Config task: `fast_cluster/config.py`, `fast_cluster/source_contract.py`, new split-history formal/Smoke YAMLs, launcher/preflight/data-preparation compatibility, tests. Helpers `split_history(config)`, `fast_frame_count(config,arm)`; old configs stay byte/semantic compatible. New config model frame map, source init/seed/freeze/train_new_modules and train feedback/dropouts are explicit.

## Acceptance

Tests first:1/4frame native layout; latest-only causal invariance; dataset1frame decode; original4 compatibility; KV capture/fuse1Fast/4Slow; bounded visual-cache hit/miss/cache-off equality; source gradients with frozen backbone; source task optimizer/checkpoint/RNG restore; official default VAL; distinct protocol resume guards. Native Smoke:1/4Fast forward plus a bounded real cooperative update and independent reload/resume, actual runtime/transport/cache flow. No whole dataset download, no formal training, no new architecture investigation. Final independent review, tests, docs and verified GitHub push required.
