"""Native planners; full-TRAIN Fast normalization and frozen-source normalization.

No labels enter the frozen feedback teacher or Slow prefill. Base weights stay
external to the portable task-state bundle. All recurrent/cache state is reset
by the existing native forward (use_cache=False).
"""
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from PIL import Image

from fast_navsim.collate import NavsimCollator
from fast_only.common import fingerprint
from slowfast_v11.attention import check_compatibility
from slowfast_v11.full_training import (configure_full_fast, full_state,
    install_full_precision, restore_full_state)
from slowfast_v11.model import FastWithSlowKV, SlowPrefill
from slowfast_v11.protocol import KVSnapshot, LayerKV
from slowfast_v11.validation import canonical_state, restore_weights, state_digest
from slowfast_v11.vision_protocol import buffer_digest, restore_native_visual_rope
from .config import baseline_only, split_history, fast_frame_count
from .pretrained_source import PretrainedSource
from .vision_cache import FrozenFrameCache, encode_frames


def seed_local(seed, device):
    """Never initialize another rank's CUDA context just to seed RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    if torch.device(device).type == 'cuda':
        with torch.cuda.device(device):
            torch.cuda.manual_seed(seed)


def normalize_ego(raw, stats):
    mean = torch.as_tensor(stats['ego_mean'], device=raw.device, dtype=torch.float32)
    std = torch.as_tensor(stats['ego_std'], device=raw.device, dtype=torch.float32)
    if (raw.ndim != 2 or raw.shape[1] != 4 or mean.shape != (4,) or std.shape != (4,)
            or not torch.isfinite(raw).all() or not torch.isfinite(mean).all()
            or not torch.isfinite(std).all() or (std <= 0).any()):
        raise ValueError('Expected finite raw Ego[B,4] and positive four-dimensional statistics')
    return (raw.float() - mean) / std


def validate_source_bundle(bundle, checkpoint_sha):
    if bundle.get('schema') != 'navsim_frozen_source_v1':
        raise ValueError('Expected explicit frozen-source bundle schema')
    if bundle.get('source_checkpoint_sha256') != checkpoint_sha:
        raise ValueError('Frozen source checkpoint identity differs')
    for field in ('slow', 'feedback_teacher'):
        if state_digest(bundle[field]) != bundle[field+'_digest']:
            raise ValueError(f'Frozen source {field} digest differs')
    normalize_ego(torch.zeros(1, 4), bundle['stats'])
    return bundle


def asset_manifest(root):
    root = Path(root)
    paths = sorted(p for p in root.iterdir() if p.is_file()
                   and p.suffix in ('.json', '.safetensors', '.jinja', '.txt', '.model'))
    if not any(p.suffix == '.safetensors' for p in paths):
        raise ValueError(f'No local base safetensors under {root}')
    return {p.name: fingerprint(p) for p in paths}


def verify_assets(root, expected):
    actual = asset_manifest(root)
    if actual != expected:
        raise ValueError(f'Base assets changed under {root}; use the exported asset identity')
    return actual


def parameter_digest(module, *, frozen_only=False):
    """Streaming hash avoids simultaneously copying all 2B weights to CPU."""
    digest = hashlib.sha256()
    for name, tensor in module.named_parameters():
        if frozen_only and tensor.requires_grad:
            continue
        digest.update(name.encode())
        host = tensor.detach().to('cpu', non_blocking=False).contiguous()
        digest.update(str((tuple(host.shape), host.dtype)).encode())
        digest.update(host.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_source(config, *, verify=True):
    assets = config['assets']
    path = Path(assets['source_bundle'])
    bundle = validate_source_bundle(torch.load(path, map_location='cpu', weights_only=False),
                                    assets['source_checkpoint_sha256'])
    from .source_contract import validate_bundle_provenance
    validate_bundle_provenance(bundle)
    if verify:
        for role in ('fast', 'slow'):
            verify_assets(assets[role+'_model'], bundle['assets'][role])
    bundle['_file_sha256'] = fingerprint(path)
    return bundle


def expand_snapshot(snapshot, batch_size):
    if snapshot.layers[0].key.shape[0] != 1:
        raise ValueError('Fixed reference must be one TRAIN observation')
    def expand(value):
        return value.expand(batch_size, *value.shape[1:])
    return KVSnapshot(tuple(LayerKV(expand(z.key), expand(z.value)) for z in snapshot.layers),
                      snapshot.position_ids.expand(-1, batch_size, -1),
                      expand(snapshot.valid_mask), snapshot.metadata)


def _move(tree, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in tree.items()}


def _sample_images(sample, image_frames=4):
    if 'images' in sample and all(isinstance(x, Image.Image) for x in sample['images']):
        if len(sample['images']) < image_frames:
            raise ValueError('Missing required current/history images')
        return sample['images'][-image_frames:]
    paths = sample.get('image_paths', sample.get('images'))
    if paths is None or len(paths) < image_frames:
        raise ValueError('Missing required current/history image paths')
    images = []
    for path in paths[-image_frames:]:
        with Image.open(path) as image:
            images.append(image.convert('RGB'))
    return images


def pack_samples(samples, collator, stats, device, image_frames=4):
    raw = torch.tensor([s['ego'] for s in samples], dtype=torch.float32, device=device)
    ego = normalize_ego(raw, stats)
    items = [dict(images=_sample_images(s, image_frames), ego=ego[i].cpu(),
                  command=torch.tensor(s['command'], dtype=torch.long),
                  trajectory_gt=torch.as_tensor(s.get('trajectory_gt', []), dtype=torch.float32))
             for i, s in enumerate(samples)]
    packed = collator(items)
    return dict(native=_move(packed['native'], device), ego=ego,
                command=packed['command'].to(device),
                target=packed['trajectory_gt'].to(device), raw_ego=raw,
                valid=torch.tensor([s.get('valid', True) for s in samples], device=device),
                image_audit=packed['image_audit'])


def _kwargs(config):
    return {k: config['model'][k] for k in
            ('lora_rank', 'lora_alpha', 'lora_dropout', 'lora_parameter_dtype', 'position_offset')}


def _new_fast(config, device, image_frames=None):
    kwargs = _kwargs(config)
    if image_frames is not None:
        kwargs['image_frames'] = image_frames
    fast = FastWithSlowKV.from_pretrained(config['assets']['fast_model'], dtype=torch.bfloat16,
                                         **kwargs).to(device)
    scope = configure_full_fast(fast)
    rope = restore_native_visual_rope(fast.backbone.visual)
    install_full_precision(fast)
    return fast, scope, rope


def _collator(config, role, frames):
    kwargs = dict(config['processor'])
    if frames == 1:
        kwargs['history_times_s'] = (0.,)
    return NavsimCollator(config['assets'][role+'_model'], **kwargs)


def _pretrained_provider(config, stats, device):
    seed_local(config['source']['seed'], device)
    slow = SlowPrefill.from_pretrained(config['assets']['slow_model'],
        dtype=torch.bfloat16, **_kwargs(config)).to(device)
    return PretrainedSource(slow, stats)


def _restore_source(provider, state):
    if not isinstance(state, dict) or state.get('schema') != 'pretrained_source_task_v1':
        raise ValueError('Checkpoint is missing trained Slow input/query state')
    buffers = dict(provider.slow.named_buffers())
    recorded = state['buffers']
    if set(buffers) != set(recorded) or any(recorded[k].shape != v.shape or
            recorded[k].dtype != v.dtype for k,v in buffers.items()):
        raise ValueError('Slow native checkpoint buffers differ')
    provider.restore_task_state(state['parameters'])
    with torch.no_grad():
        for key, value in buffers.items():
            value.copy_(recorded[key])


def _source_runtime_id(identity, source_state, checkpoint_sha):
    payload = dict(assets=identity['assets'], source_seed=identity['source_seed'],
        stats=identity['source_stats'], task_digest=state_digest(source_state['parameters']),
        buffers=state_digest(source_state['buffers']), checkpoint=checkpoint_sha)
    return 'pretrained:'+hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False).encode()).hexdigest()


class FrozenProvider(nn.Module):
    def __init__(self, config, bundle, device, *, with_teacher=True):
        super().__init__()
        self.stats = bundle['stats']
        self.slow = SlowPrefill.from_pretrained(config['assets']['slow_model'],
            dtype=torch.bfloat16, **bundle['model_kwargs']).to(device)
        restore_weights(self.slow, bundle['slow'])
        self.teacher = None
        if with_teacher:
            self.teacher = FastWithSlowKV.from_pretrained(config['assets']['fast_model'],
                dtype=torch.bfloat16, **bundle['model_kwargs']).to(device)
            restore_weights(self.teacher, bundle['feedback_teacher'])
        self.requires_grad_(False)
        self.eval()
        self.slow_buffers = buffer_digest(self.slow)

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, fast_native, slow_native, raw_ego, command):
        ego = normalize_ego(raw_ego, self.stats)
        feedback = self.teacher(fast_native, ego, command)
        return self.slow(slow_native, ego, command, feedback=feedback).snapshot


class PlanningBackend(nn.Module):
    def __init__(self, config, stats, arm, seed, device):
        super().__init__()
        self.modern = split_history(config)
        allowed = ('fast_single', 'fast_history', 'slow_fast') if self.modern else ('fast_only', 'slow_fast')
        if arm not in allowed:
            raise ValueError('Unknown planning arm for this input protocol')
        self.config, self.stats, self.arm = config, stats, arm
        self.device = torch.device(device)
        self.image_frames = fast_frame_count(config, arm)
        seed_local(seed, device)
        self.fast, self.scope, self.visual_rope = (_new_fast(config, device, self.image_frames)
            if self.modern else _new_fast(config, device))
        self.initial_digest = state_digest(full_state(self.fast))
        self.initial_buffers = buffer_digest(self.fast)
        self.collator = _collator(config, 'fast', self.image_frames)
        self.provider = None
        self.slow_collator = None
        self.fixed_snapshot = None
        self.eval_source = 'correct'
        # Original Fast-only authenticates only its own pretrained assets. The
        # legacy paired recipe retains its previous identity for strict resume.
        bundle = None if self.modern or baseline_only(config) else load_source(config)
        self.model_identity = dict(assets={'fast': asset_manifest(config['assets']['fast_model'])} if bundle is None else bundle['assets'],
            source_bundle=None if bundle is None else bundle['_file_sha256'],
            source_checkpoint=None if bundle is None else bundle['source_checkpoint_sha256'],
            source_stats=None if bundle is None else bundle['stats'],
            fast_stats=stats, initial_fast_digest=self.initial_digest,
            initial_buffers=self.initial_buffers, processor='native_original_pad32_temporal_v1',
            loss='official_navsim_mean_l1_pose', feedback='none' if bundle is None else 'frozen_source_teacher')
        if arm == 'slow_fast':
            if self.modern:
                self.provider = _pretrained_provider(config, stats, device)
                self.source_initial_digest = state_digest(self.provider.task_state())
                self.source_buffers = buffer_digest(self.provider)
            else:
                if _kwargs(config) != bundle['model_kwargs']:
                    raise ValueError('Frozen source and Fast interface model kwargs differ')
                self.provider = FrozenProvider(config, bundle, device)
            self.interface = check_compatibility(self.fast.backbone.language_model,
                                                  self.provider.slow.backbone.language_model)
            self.slow_collator = _collator(config, 'slow', 4)
        if self.modern:
            if self.provider is not None:
                self.model_identity['assets']['slow'] = asset_manifest(config['assets']['slow_model'])
            self.model_identity.update(protocol=config['schema'], fast_image_frames=self.image_frames,
                slow_image_frames=4 if self.provider is not None else None,
                source_init='pretrained' if self.provider is not None else None,
                source_seed=config['source']['seed'] if self.provider is not None else None,
                source_stats=stats if self.provider is not None else None,
                source_task_init_digest=getattr(self,'source_initial_digest',None),
                source_initial_buffers=getattr(self,'source_buffers',None),
                source_trainable_modules=list(self.provider.task_names) if self.provider is not None else [],
                vision_execution='per_frame', feedback='completed_fast_detached' if self.provider is not None else 'none',
                training_feedback='same-observation detached no-KV candidate',
                runtime_feedback='latest completed main Fast plan; may use KV')
        del bundle
        # Source construction must not alter data/dropout RNG relative to baseline.
        seed_local(seed, device)

    def audit_frozen(self):
        self.check_frozen()
        return dict(visual_parameters=parameter_digest(self.fast.backbone.visual),
                    fast_buffers=buffer_digest(self.fast),
                    source_parameters=parameter_digest(self.provider, frozen_only=self.modern) if self.provider is not None else None,
                    source_buffers=buffer_digest(self.provider) if self.provider is not None else None)

    def prepare(self, samples):
        batch = pack_samples(samples, self.collator, self.stats, self.device, self.image_frames)
        if self.provider is not None:
            slow = pack_samples(samples, self.slow_collator, self.provider.stats, self.device)
            batch['slow_native'] = slow['native']
        return batch

    def _candidate(self, batch):
        with torch.no_grad():
            return self.fast(batch['native'],batch['ego'],batch['command']).detach()

    def _modern_forward(self, batch):
        from tools.train_frozen_expert import cached_vision
        vision = encode_frames(self.fast.backbone,batch['native'])
        with cached_vision(self.fast.backbone,vision,batch['native']['image_grid_thw']):
            kwargs = {}
            use = self.provider is not None and self.eval_source != 'none'
            if use and self.training:
                use = bool(torch.rand(()) >= self.config['train']['kv_dropout'])
            self.last_source_used = use
            if use:
                if self.eval_source == 'fixed':
                    if self.fixed_snapshot is None:
                        raise ValueError('Initialize fixed TRAIN source before fixed evaluation')
                    snapshot = expand_snapshot(self.fixed_snapshot,len(batch['ego']))
                else:
                    feedback = self._candidate(batch)
                    if self.training and bool(torch.rand(()) < self.config['train']['feedback_dropout']):
                        feedback = None
                    snapshot = self.provider(batch['slow_native'],
                        normalize_ego(batch['raw_ego'],self.provider.stats),batch['command'],feedback)
                metadata = batch['ego'].new_tensor([[0.,0.,0.,0.,1.,1.]]).expand(len(batch['ego']),-1)
                kwargs = dict(snapshot=snapshot,metadata=metadata)
            return self.fast(batch['native'],batch['ego'],batch['command'],**kwargs)

    def forward(self, batch):
        if self.modern:
            return self._modern_forward(batch)
        kwargs = {}
        if self.provider is not None and self.eval_source != 'none':
            if self.eval_source == 'fixed':
                if self.fixed_snapshot is None:
                    raise ValueError('Initialize fixed reference from TRAIN before fixed evaluation')
                snapshot = expand_snapshot(self.fixed_snapshot, len(batch['ego']))
            else:
                snapshot = self.provider(batch['native'], batch['slow_native'],
                                         batch['raw_ego'], batch['command'])
            metadata = batch['ego'].new_tensor([[0., 0., 0., 0., 1., 1.]]).expand(len(batch['ego']), -1)
            kwargs = dict(snapshot=snapshot, metadata=metadata)
        return self.fast(batch['native'], batch['ego'], batch['command'], **kwargs)

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.eval_source = 'correct'
        return self

    def set_eval_source(self, mode):
        if mode not in ('correct', 'fixed', 'none'):
            raise ValueError('Invalid source mode')
        self.eval_source = mode

    @torch.no_grad()
    def initialize_fixed_source(self, sample):
        if self.provider is None:
            return
        batch = self.prepare([sample])
        if self.modern:
            from tools.train_frozen_expert import cached_vision
            vision = encode_frames(self.fast.backbone,batch['native'])
            with cached_vision(self.fast.backbone,vision,batch['native']['image_grid_thw']):
                feedback = self._candidate(batch)
            self.fixed_snapshot = self.provider(batch['slow_native'],
                normalize_ego(batch['raw_ego'],self.provider.stats),batch['command'],feedback).inference_copy()
            return
        self.fixed_snapshot = self.provider(batch['native'], batch['slow_native'],
                                            batch['raw_ego'], batch['command'])

    def diagnostics(self):
        return dict(arm=self.arm, scope=self.scope, image_frames=self.image_frames, init_digest=self.initial_digest,
                    visual_rope=self.visual_rope, buffers=self.initial_buffers,
                    interface=getattr(self, 'interface', None), model_identity=self.model_identity)

    def check_frozen(self):
        modules = [self.fast.backbone.visual]
        if self.provider is not None and not self.modern:
            modules.append(self.provider)
        if self.provider is not None and self.modern:
            self.provider.check_frozen()
        if any(p.requires_grad or p.grad is not None for m in modules for p in m.parameters()):
            raise ValueError('Frozen vision/source acquired trainable parameters or gradients')
        if buffer_digest(self.fast) != self.initial_buffers:
            raise ValueError('Native Fast buffers changed')
        if self.provider is not None and (buffer_digest(self.provider) != self.source_buffers if self.modern
                else buffer_digest(self.provider.slow) != self.provider.slow_buffers):
            raise ValueError('Frozen Slow buffers changed')
        return True

    def fast_state(self):
        return full_state(self.fast)

    def restore_fast_state(self, state):
        restore_full_state(self.fast, state)

    def source_state(self):
        if not self.modern or self.provider is None:
            return None
        return dict(schema='pretrained_source_task_v1', parameters=self.provider.task_state(),
                    buffers={k:v.detach().cpu().clone() for k,v in self.provider.slow.named_buffers()})

    def restore_source_state(self, state):
        if self.modern and self.provider is not None:
            _restore_source(self.provider,state)
        elif state is not None:
            raise ValueError('Unexpected trained source state for source-free/legacy planner')

    def source_gradient_report(self):
        return self.provider.gradient_report() if self.modern and self.provider is not None else None

    def add_source_optimizer_group(self, optimizer, learning_rate):
        if self.modern and self.provider is not None:
            optimizer.add_param_group(dict(name='source_new',lr=learning_rate,
                params=[p for p in self.provider.parameters() if p.requires_grad]))


def build_backend(config, stats, arm, seed, device):
    return PlanningBackend(config, stats, arm, seed, device)


def _sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


class FastRuntime:
    def __init__(self, config, checkpoint, device):
        device = torch.device(device)
        if device.type == 'cuda':
            torch.cuda.set_device(device)
        saved = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
        # Training checkpoint field convention is validated below; no fallback
        # to an adapter-only checkpoint is permitted.
        identity = saved['identity']
        model_id = identity.get('model', identity.get('model_identity'))
        if model_id is None:
            raise ValueError('Runtime requires a cluster checkpoint with model identity')
        self.modern = split_history(config)
        self.image_frames = fast_frame_count(config,identity['arm'])
        self.checkpoint_sha = fingerprint(checkpoint)
        bundle = None
        if self.modern:
            if (model_id.get('protocol') != config['schema']
                    or model_id.get('fast_image_frames') != self.image_frames
                    or model_id.get('vision_execution') != 'per_frame'):
                raise ValueError('Runtime input/vision protocol differs from the trained checkpoint')
            for role, expected in model_id['assets'].items():
                verify_assets(config['assets'][role+'_model'],expected)
            if model_id.get('source_init') == 'pretrained':
                if not isinstance(saved.get('source_tasks'),dict):
                    raise ValueError('Cooperative runtime requires trained Slow input/query state')
                runtime_id = _source_runtime_id(model_id,saved['source_tasks'],self.checkpoint_sha)
            else:
                runtime_id = 'fast_only:'+self.checkpoint_sha
        elif model_id['source_bundle'] is None:
            if not baseline_only(config):
                raise ValueError('Original Fast-only checkpoint requires its baseline-only configuration')
            actual = {'fast': asset_manifest(config['assets']['fast_model'])}
            if model_id['assets'] != actual:
                raise ValueError('Runtime Fast base assets differ from trained checkpoint')
        else:
            bundle = load_source(config)
            if model_id['assets'] != bundle['assets'] or model_id['source_bundle'] != bundle['_file_sha256']:
                raise ValueError('Runtime source/base assets differ from trained checkpoint')
        self.device = torch.device(device)
        self.stats = model_id['fast_stats']
        self.fast, _, _ = (_new_fast(config,device,self.image_frames)
                           if self.modern else _new_fast(config, device))
        restore_full_state(self.fast, saved['model'])
        buffers = dict(self.fast.named_buffers())
        recorded = saved['fast_buffers']
        if set(buffers) != set(recorded) or any(recorded[k].shape != v.shape or
                recorded[k].dtype != v.dtype for k, v in buffers.items()):
            raise ValueError('Runtime checkpoint native buffers differ')
        with torch.no_grad():
            for name, tensor in buffers.items():
                tensor.copy_(recorded[name])
        self.fast.eval().requires_grad_(False)
        self.collator = _collator(config,'fast',self.image_frames)
        self.model_id = runtime_id if self.modern else ('fast_only:'+self.checkpoint_sha if bundle is None else bundle['_file_sha256'])
        self.interface_info = dict(snapshot_layers=6, kv_heads=2, head_dim=256,
                                   image_frames=self.image_frames,
                                   checkpoint_sha256=self.checkpoint_sha,
                                   source_bundle_sha256=None if bundle is None else self.model_id)
        self.frame_cache = None
        if self.modern:
            self._visual_assets = model_id['assets']['fast']
            self._processing_policy = dict(config['processor'],vision_execution='per_frame')
            self.frame_cache = FrozenFrameCache(self._visual_provider_id(),
                capacity=config['vision']['runtime_cache_capacity'],
                enabled=config['vision']['runtime_history_cache'])
        self._frozen_identity = parameter_digest(self.fast)

    def _visual_provider_id(self):
        payload = dict(assets=self._visual_assets,buffers=buffer_digest(self.fast.backbone.visual),
                       processor=self._processing_policy)
        return hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False).encode()).hexdigest()

    def audit_frozen(self):
        actual = parameter_digest(self.fast)
        if actual != self._frozen_identity:
            raise ValueError('Inference Fast parameters changed')
        return dict(parameters_sha256=actual, buffers_sha256=buffer_digest(self.fast))

    @torch.no_grad()
    def prepare(self, observation):
        start = time.monotonic()
        keys = None
        if self.frame_cache is not None:
            provider_id = self._visual_provider_id()
            if provider_id != self.frame_cache.provider_id:
                self.frame_cache.clear(provider_id=provider_id)
            paths = observation.get('image_paths')
            if paths is None and all(isinstance(p,(str,Path)) for p in observation.get('images',[])):
                paths = observation['images']
            if paths is not None:
                if len(paths) < self.image_frames:
                    raise ValueError('Missing required runtime current/history images')
                decoded = [self.frame_cache.image(p) for p in paths[-self.image_frames:]]
                images,keys = [p[0] for p in decoded],[p[1] for p in decoded]
            else:
                images = _sample_images(observation,self.image_frames)
        else:
            images = _sample_images(observation,self.image_frames)
        decoded = time.monotonic()
        batch = pack_samples([dict(observation, images=images)], self.collator, self.stats,self.device,self.image_frames)
        _sync(self.device)
        processed = time.monotonic()
        native = batch['native']
        vision = (encode_frames(self.fast.backbone,native,
                    cache=self.frame_cache if keys is not None else None,keys=keys) if self.modern else
                  self.fast.backbone.get_image_features(native['pixel_values'],
                     native['image_grid_thw'], return_dict=True).pooler_output)
        _sync(self.device)
        end = time.monotonic()
        batch['vision'] = tuple(vision)
        batch['timings'] = dict(decode_s=decoded-start, processor_s=processed-decoded,
                              vision_s=end-processed, preparation_total_s=end-start)
        return batch

    @torch.no_grad()
    def predict(self, prepared, snapshot, metadata):
        from tools.train_frozen_expert import cached_vision
        _sync(self.device)
        start = time.monotonic()
        kwargs = {} if snapshot is None else dict(snapshot=snapshot,
                        metadata=metadata.to(self.device).reshape(1, 6))
        with cached_vision(self.fast.backbone, prepared['vision'], prepared['native']['image_grid_thw']):
            output = self.fast(prepared['native'], prepared['ego'], prepared['command'], **kwargs)
        result = output.detach().to('cpu', non_blocking=False).clone()
        _sync(self.device)
        elapsed = time.monotonic()-start
        return result, dict(llm_head_s=elapsed, model_total_s=elapsed,
                           model_total_scope='language+trajectory_head+CPU_submission; vision in prepare')


class SlowRuntime:
    def __init__(self, config, device):
        self.device = torch.device(device)
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
        if split_history(config):
            checkpoint = config.get('runtime_checkpoint')
            if checkpoint is None:
                raise ValueError('Slow runtime requires the same cooperative checkpoint as Fast')
            saved = torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
            identity = saved['identity']['model']
            if identity.get('protocol') != config['schema'] or identity.get('source_init') != 'pretrained':
                raise ValueError('Slow runtime requires a split-history cooperative checkpoint')
            for role, expected in identity['assets'].items():
                verify_assets(config['assets'][role+'_model'],expected)
            self.provider = _pretrained_provider(config,identity['source_stats'],device)
            _restore_source(self.provider,saved.get('source_tasks'))
            self.model_id = _source_runtime_id(identity,saved['source_tasks'],fingerprint(checkpoint))
            self.provider.freeze_for_inference()
        else:
            bundle = load_source(config)
            self.model_id = bundle['_file_sha256']
            self.provider = FrozenProvider(config, bundle, device, with_teacher=False)
        self.collator = NavsimCollator(config['assets']['slow_model'], **config['processor'])
        self._frozen_identity = parameter_digest(self.provider)

    def audit_frozen(self):
        actual = parameter_digest(self.provider)
        if actual != self._frozen_identity:
            raise ValueError('Inference Slow parameters changed')
        return dict(parameters_sha256=actual, buffers_sha256=buffer_digest(self.provider))

    @torch.no_grad()
    def prefill(self, observation, feedback_cpu):
        start = time.monotonic()
        batch = pack_samples([observation], self.collator, self.provider.stats, self.device)
        feedback = None if feedback_cpu is None else torch.as_tensor(feedback_cpu,
                            device=self.device, dtype=torch.float32).reshape(1, 8, 3)
        _sync(self.device)
        prepared = time.monotonic()
        events = None
        if self.device.type == 'cuda':
            events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            events[0].record()
        result = self.provider.slow(batch['native'], batch['ego'], batch['command'], feedback=feedback).snapshot
        if events is not None:
            events[1].record()
        _sync(self.device)
        completed = time.monotonic()
        self.last_timings = dict(
            preparation_s=prepared-start, model_wall_s=completed-prepared,
            gpu_prefill_span_s=None if events is None else events[0].elapsed_time(events[1])/1000,
            total_s=completed-start,
            scope='preparation includes decode/Processor/H2D; model includes vision/language/heads; '
                  'CUDA stream elapsed span is not active kernel time; snapshot transfer is separate')
        return result


def make_runtime_fast(config, checkpoint, device):
    return FastRuntime(config, checkpoint, device)


def make_runtime_slow(config, device):
    return SlowRuntime(config, device)
