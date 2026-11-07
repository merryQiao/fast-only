# Frozen Slow Dedicated Expert Control

> **For agentic workers:** Apply subagent-driven-development, test-first implementation and independent review. User explicitly requested this added architecture control while preserving the existing full Fast. This supersedes the previous conditional-only scheduling of a consumer experiment; small readability probes are diagnostic, not a gate used to reject the KV route.

**Goal:** Compare full-parameter dedicated Action Expert against complete Fast+LoRA with one stable frozen Slow provider and the same new sample exposure budget.

**Sources:** See docs/KV_EXPERT_REFERENCE_AUDIT.md, pinned Qwen-Drive/AutoMoT code and Think5Hz paper. This is a Qwen-Drive-inspired local regression adaptation, not a paper reproduction.

## Locked decisions before training

- Same original-resolution existingTRAIN256/fourlogs, newVAL64/twologs, legacy4/onelog. No downloads/full navtrain. Same frozen expanded joint_final Slow and saved numeric Fast512 feedback; verify all cache/model/data identities. Frozen source cache exports all six raw pre-RoPE K/V, no pooling or adapter; never train Slow.
- Dedicated Expert: 24 dense fully trainable blocks, width512, SwiGLU1536,8Qheads,2KVheads,head_dim256; block i reads nativeFA cache i//4. Direct [Kslow;Kexpert]/[Vslow;Vexpert], own per-head Q/K RMS, native Slow mRoPE exactly once, sigmoid gate then output projection. Inputs=current normalized Ego4, NAVSIM command4, cache metadata,8queries; only scene-vision path is Slow KV. No local Fast vision branch. Single8x3regression head and existingmeanL1pose, no diffusion/flow loss.
- Own source-relative positions match existing Fast sensor/query reserved-offset convention4096, including source native-prefix/Ego anchor. Qwen-Drive uses contiguous prefix continuation; this is a known retained protocol difference, not a claimed replication. Hold this position convention fixed in both consumers instead of changing it with parameter trainability.
- Complete Fast baseline starts from existing joint_final Fast; all native language/vision baseweights frozen, LoRA/newmodules trainable as before. Expert is newly randomly initialized seed42. This compares practical architectures/initializations, not a pure parameter-count or pretraining ablation. Record prior1312Fast exposures vs randomExpert as well as newbudget; frozenSlow itself inherited prior adaptation.
- Per arm4096optimizerupdates×MB1, noaccum=4096TRAINdraws=16epochs. Identical deterministic epoch-shuffle target order seed42. Allfresh KV, no dropout/age/async; no loss encouraging bad KV to fail. AdamW nativeblocks/LoRA1e-4, newconditioning/head5e-4,WD.01,clip1,128warmup/cosine ending10% peak. FullExpert FFN/projection/normparams are updated. Current256-window experiment does not establish sufficient convergence or a data-scaled ceiling.
- FixedTRAIN32curves at1024/2048/3072/4096; noVAL until complete budget per arm, noVALselection/extension. Evaluate correct/masked/fixedreference/cohort-localshuffled with fixed current inputs/positions/metadata, and same outputmetrics on bothcohorts. Wrong/fixed/masked are offline interventions, not deployed policies. Allstrategiesuse oneHead. Source metadata remainsvalid1 whenmasking externalcolumns.
- Cache frozen Fast vision features for baseline, alloriginal pixels processed usingunchangedProcessor/merger; no resize or compression. Verify same-checkpoint cached-vs-native exact prediction on predetermined TRAIN+VAL examples. This is frozen-compute reuse, not a change to visual inputs. NoSlowfeaturegrad, noFastvisiongrad. CPUstores smallnativeids/grid/ego/cmd; rawpixel tensors discarded after exactfrozenvisionoutput captured. GPUcap76GiB onH80080GB; fail explicitly ratherthan silently altering MB/resolution/cache.
- Record losses, gradclip stats, alltrainableparametercounts and nonzeroExpertlayergrads, sourceimmutability, model/statshashes, complete trainorder, optimizer/scheduler/RNG, windowsseen/epochs, FLOPs/runtime/memory proxies. Equal samples/updates do not equal FLOPs, trainableparams, inputpaths or initialization. Save checkpoint and prediction evidence, including4095/4096 optimizer/RNG boundaries and one exact raw/vision input packet for independent final-step replay. Never claim this bounded prototype proves/disproves KV ceiling.

## Tasks

- [ ] Native-compatible ActionExpert and CPU24layer geometry/grad/mask/sourceimmutability tests.
- [ ] Cache loader / readonlysnapshot / cachedfrozenvision equivalence / frozenSlowmatchedtraining runner and tests.
- [ ] Independent review and source pinning; run Expert and completeFastLoRA sequentially ononeGPU underfixedbudget.
- [ ] Portable results, trainingcurves, perlog/speed/conditionmetrics, checkpoint identities, limitations and research implications; report alongside empty-path/readability evidence. Commit without rawcache/images/weights; GitHub auth stillrequired.
