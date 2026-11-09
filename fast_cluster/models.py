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


def parameter_digest(module):
    """Streaming hash avoids simultaneously copying all 2B weights to CPU."""
    digest = hashlib.sha256()
    for name, tensor in module.named_parameters():
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


def _sample_images(sample):
    if 'images' in sample and all(isinstance(x, Image.Image) for x in sample['images']):
        return sample['images']
    paths = sample.get('image_paths', sample.get('images'))
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert('RGB'))
    return images


def pack_samples(samples, collator, stats, device):
    raw = torch.tensor([s['ego'] for s in samples], dtype=torch.float32, device=device)
    ego = normalize_ego(raw, stats)
    items = [dict(images=_sample_images(s), ego=ego[i].cpu(),
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


def _new_fast(config, device):
    fast = FastWithSlowKV.from_pretrained(config['assets']['fast_model'], dtype=torch.bfloat16,
                                         **_kwargs(config)).to(device)
    scope = configure_full_fast(fast)
    rope = restore_native_visual_rope(fast.backbone.visual)
    install_full_precision(fast)
    return fast, scope, rope


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
        if arm not in ('fast_only', 'slow_fast'):
            raise ValueError('Expected fast_only or slow_fast')
        self.config, self.stats, self.arm = config, stats, arm
        self.device = torch.device(device)
        seed_local(seed, device)
        self.fast, self.scope, self.visual_rope = _new_fast(config, device)
        self.initial_digest = state_digest(full_state(self.fast))
        self.initial_buffers = buffer_digest(self.fast)
        self.collator = NavsimCollator(config['assets']['fast_model'], **config['processor'])
        self.provider = None
        self.slow_collator = None
        self.fixed_snapshot = None
        self.eval_source = 'correct'
        # Both arms authenticate the same source/assets, even though baseline
        # does not instantiate or run Slow/teacher.
        bundle = load_source(config)
        self.model_identity = dict(assets=bundle['assets'], source_bundle=bundle['_file_sha256'],
            source_checkpoint=bundle['source_checkpoint_sha256'], source_stats=bundle['stats'],
            fast_stats=stats, initial_fast_digest=self.initial_digest,
            initial_buffers=self.initial_buffers, processor='native_original_pad32_temporal_v1',
            loss='official_navsim_mean_l1_pose', feedback='frozen_source_teacher')
        if arm == 'slow_fast':
            if _kwargs(config) != bundle['model_kwargs']:
                raise ValueError('Frozen source and Fast interface model kwargs differ')
            self.provider = FrozenProvider(config, bundle, device)
            self.interface = check_compatibility(self.fast.backbone.language_model,
                                                  self.provider.slow.backbone.language_model)
            self.slow_collator = NavsimCollator(config['assets']['slow_model'], **config['processor'])
        del bundle
        # Source construction must not alter data/dropout RNG relative to baseline.
        seed_local(seed, device)

    def audit_frozen(self):
        self.check_frozen()
        return dict(visual_parameters=parameter_digest(self.fast.backbone.visual),
                    fast_buffers=buffer_digest(self.fast),
                    source_parameters=parameter_digest(self.provider) if self.provider is not None else None,
                    source_buffers=buffer_digest(self.provider) if self.provider is not None else None)

    def prepare(self, samples):
        batch = pack_samples(samples, self.collator, self.stats, self.device)
        if self.provider is not None:
            slow = pack_samples(samples, self.slow_collator, self.provider.stats, self.device)
            batch['slow_native'] = slow['native']
        return batch

    def forward(self, batch):
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
        self.fixed_snapshot = self.provider(batch['native'], batch['slow_native'],
                                            batch['raw_ego'], batch['command'])

    def diagnostics(self):
        return dict(arm=self.arm, scope=self.scope, init_digest=self.initial_digest,
                    visual_rope=self.visual_rope, buffers=self.initial_buffers,
                    interface=getattr(self, 'interface', None), model_identity=self.model_identity)

    def check_frozen(self):
        modules = [self.fast.backbone.visual]
        if self.provider is not None:
            modules.append(self.provider)
        if any(p.requires_grad or p.grad is not None for m in modules for p in m.parameters()):
            raise ValueError('Frozen vision/source acquired trainable parameters or gradients')
        if buffer_digest(self.fast) != self.initial_buffers:
            raise ValueError('Native Fast buffers changed')
        if self.provider is not None and buffer_digest(self.provider.slow) != self.provider.slow_buffers:
            raise ValueError('Frozen Slow buffers changed')
        return True

    def fast_state(self):
        return full_state(self.fast)

    def restore_fast_state(self, state):
        restore_full_state(self.fast, state)


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
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        # Training checkpoint field convention is validated below; no fallback
        # to an adapter-only checkpoint is permitted.
        identity = saved['identity']
        model_id = identity.get('model', identity.get('model_identity'))
        if model_id is None:
            raise ValueError('Runtime requires a cluster checkpoint with model identity')
        bundle = load_source(config)
        if model_id['assets'] != bundle['assets'] or model_id['source_bundle'] != bundle['_file_sha256']:
            raise ValueError('Runtime source/base assets differ from trained checkpoint')
        self.device = torch.device(device)
        self.stats = model_id['fast_stats']
        self.fast, _, _ = _new_fast(config, device)
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
        self.collator = NavsimCollator(config['assets']['fast_model'], **config['processor'])
        self.model_id = bundle['_file_sha256']
        self.interface_info = dict(snapshot_layers=6, kv_heads=2, head_dim=256,
                                   checkpoint_sha256=fingerprint(checkpoint),
                                   source_bundle_sha256=self.model_id)
        self._frozen_identity = parameter_digest(self.fast)

    def audit_frozen(self):
        actual = parameter_digest(self.fast)
        if actual != self._frozen_identity:
            raise ValueError('Inference Fast parameters changed')
        return dict(parameters_sha256=actual, buffers_sha256=buffer_digest(self.fast))

    @torch.no_grad()
    def prepare(self, observation):
        start = time.monotonic()
        images = _sample_images(observation)
        decoded = time.monotonic()
        batch = pack_samples([dict(observation, images=images)], self.collator, self.stats, self.device)
        _sync(self.device)
        processed = time.monotonic()
        native = batch['native']
        vision = self.fast.backbone.get_image_features(native['pixel_values'],
                     native['image_grid_thw'], return_dict=True).pooler_output
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
        batch = pack_samples([observation], self.collator, self.provider.stats, self.device)
        feedback = None if feedback_cpu is None else torch.as_tensor(feedback_cpu,
                            device=self.device, dtype=torch.float32).reshape(1, 8, 3)
        return self.provider.slow(batch['native'], batch['ego'], batch['command'], feedback=feedback).snapshot


def make_runtime_fast(config, checkpoint, device):
    return FastRuntime(config, checkpoint, device)


def make_runtime_slow(config, device):
    return SlowRuntime(config, device)
