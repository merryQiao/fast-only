"""Configuration, native image batching and numerical reporting."""
from __future__ import annotations

import json
import hashlib
import os
import random
import threading
from pathlib import Path

import numpy as np
import torch
import yaml


def read_config(path):
    with open(path) as f:
        config = yaml.safe_load(f)
    from .configuration import validate_training_config
    return validate_training_config(config)


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def fingerprint(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def experiment_fingerprints(config):
    """Hash actual base weights, processor assets and fixed training evidence."""
    result = {key: fingerprint(config['data'][key]) for key in ['train_manifest', 'stats']}
    if config['data'].get('val_manifest'):
        result['val_manifest'] = fingerprint(config['data']['val_manifest'])
    if config['data'].get('source_audit'):
        from .navigation import load_source_audit
        evidence = load_source_audit(config['data']['source_audit'])
        result['source_audit'] = evidence['audit_sha256']
        for item in evidence['sources']:
            result[f'official_source/{Path(item["path"]).name}'] = item['sha256']
    if config.get('schema_version') == 2:
        # Pin the implementation as well as data/weight identities. New code
        # belongs to a new run rather than silently resuming a historical one.
        for path in sorted(Path(__file__).parent.glob('*.py')):
            result[f'implementation/{path.name}'] = fingerprint(path)
    path = Path(config['model']['path'])
    assets = sorted(p for p in path.iterdir() if p.is_file() and
                    p.suffix in {'.json', '.safetensors', '.jinja', '.txt'})
    if not any(p.suffix == '.safetensors' for p in assets):
        raise FileNotFoundError(f'No local safetensors weights in {path}')
    result.update({f'model/{p.name}': fingerprint(p) for p in assets})
    return result


def verify_checkpoint_inputs(state):
    current = experiment_fingerprints(state['config'])
    if current != state['data_fingerprints']:
        changed = sorted(k for k in current.keys() | state['data_fingerprints'].keys()
                         if current.get(k) != state['data_fingerprints'].get(k))
        raise ValueError(f'Checkpoint input fingerprint mismatch: {changed}')


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def trajectory_metrics(prediction, target):
    diff = prediction.detach().float().cpu() - target.detach().float().cpu()
    if diff.ndim != 3 or diff.shape[1:] != (6, 2) or not torch.isfinite(diff).all():
        raise ValueError('Metrics need finite [N,6,2] arrays')
    distance = torch.linalg.vector_norm(diff, dim=-1)
    metrics = {'ADE': distance.mean().item(), 'FDE': distance[:, -1].mean().item(),
               'loss_l1_xy': diff.abs().sum(-1).mean().item(), 'samples': len(diff)}
    metrics.update({f'L2@{(i+1)*.5:g}s': value for i, value in enumerate(distance.mean(0).tolist())})
    return metrics


def prepare_original_image(image, multiple=32, fill=(0, 0, 0)):
    """Preserve decoded source pixels; align native patches by padding only."""
    from PIL import ImageOps
    if not isinstance(multiple, int) or multiple <= 0:
        raise ValueError('Patch alignment multiple must be a positive integer')
    width, height = image.size
    right, bottom = (-width) % multiple, (-height) % multiple
    image = image.convert('RGB')
    padded = ImageOps.expand(image, border=(0, 0, right, bottom), fill=tuple(fill))
    return padded, {'source_size_wh': [width, height],
                    'processed_size_wh': list(padded.size), 'resized': False,
                    'padding_ltrb': [0, 0, right, bottom],
                    'policy': 'original_pad'}


class NativeCollator:
    """Keep four separate native images in chronological prompt order."""
    def __init__(self, model_path, min_pixels=None, max_pixels=None, brightness=0.,
                 policy='native_smart_resize', pad_multiple=32, pad_fill=(0, 0, 0),
                 prompt_template='temporal_task_v1'):
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.processor.tokenizer.padding_side = 'left'
        if policy not in {'native_smart_resize', 'original_pad'}:
            raise ValueError(f'Unknown image policy: {policy}')
        if prompt_template != 'temporal_task_v1':
            raise ValueError('Only the explicit temporal_task_v1 prompt is implemented')
        self.policy = policy
        self.pad_multiple = pad_multiple
        self.pad_fill = pad_fill
        self.prompt_template = prompt_template
        self._processor_lock = threading.RLock()
        native_multiple = self.processor.image_processor.patch_size * self.processor.image_processor.merge_size
        if policy == 'original_pad':
            if pad_multiple != native_multiple:
                raise ValueError(f'Original policy must align to native patch*merge={native_multiple}')
            self.processor.image_processor.do_resize = False
        else:
            self.processor.image_processor.size = {'shortest_edge': min_pixels or 16384,
                                                   'longest_edge': max_pixels or 147456}
        self.brightness = brightness

    def __call__(self, samples, seed=None):
        from PIL import ImageEnhance
        rng = random.Random(seed) if seed is not None else random
        contents = []; images = []; audits = []
        for sample in samples:
            frames = sample['images']
            if self.brightness:
                factor = rng.uniform(1-self.brightness, 1+self.brightness)
                frames = [ImageEnhance.Brightness(image).enhance(factor) for image in frames]
            if self.policy == 'original_pad':
                prepared = [prepare_original_image(image, self.pad_multiple, self.pad_fill) for image in frames]
                frames = [item[0] for item in prepared]
                audits.extend(item[1] for item in prepared)
            else:
                audits.extend({'source_size_wh': list(image.size), 'policy': self.policy} for image in frames)
            content = []
            for i, image in enumerate(frames):
                content += [{'type': 'text', 'text': f'Front camera t{(i-3)*.1:+.1f}s: '},
                            {'type': 'image', 'image': image}]
            content.append({'type': 'text', 'text': 'Plan six future ego trajectory points.'})
            contents.append(content)
            images.extend(frames)
        # Tokenizers can temporarily mutate padding/truncation configuration.
        # Lock native processing while worker threads overlap independent JPEG
        # reads/padding with GPU work; all augmentation randomness is batch-local.
        with self._processor_lock:
            prompts = [self.processor.apply_chat_template([{'role': 'user', 'content': content}],
                         tokenize=False, add_generation_prompt=False) for content in contents]
            kwargs = {'images_kwargs': {'do_resize': False}} if self.policy == 'original_pad' else {}
            native = self.processor(text=prompts, images=images, padding=True, return_tensors='pt', **kwargs)
        grids = native['image_grid_thw'].tolist()
        patch_size = self.processor.image_processor.patch_size
        for audit, grid in zip(audits, grids):
            actual_size = [grid[2]*patch_size, grid[1]*patch_size]
            if self.policy == 'original_pad' and audit['processed_size_wh'] != actual_size:
                raise RuntimeError('Native processor unexpectedly resized original-pixel input')
            audit['processed_size_wh'] = actual_size
            audit['image_grid_thw'] = grid
            audit['visual_tokens'] = int(np.prod(grid) // self.processor.image_processor.merge_size**2)
        return {'native': dict(native), 'image_audit': audits,
                **{k: torch.stack([sample[k] for sample in samples])
                   for k in ['ego', 'command', 'nav', 'trajectory_gt']}}


def to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {key: to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [to_device(value, device) for value in batch]
    if isinstance(batch, tuple):
        return tuple(to_device(value, device) for value in batch)
    return batch


def apply_reproducibility(config):
    """Apply the declared CUDA policy before model/device initialization."""
    policy = config.get('reproducibility')
    if policy is None:
        return None
    expected = policy['cublas_workspace_config']
    existing = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    if existing is not None and existing != expected:
        raise ValueError(f'CUBLAS_WORKSPACE_CONFIG must be {expected}, got {existing}')
    if existing is None:
        if torch.cuda.is_initialized():
            raise RuntimeError('Set CUBLAS_WORKSPACE_CONFIG before CUDA initialization; restart the process')
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = expected
    torch.use_deterministic_algorithms(policy['deterministic_algorithms'], warn_only=policy['warn_only'])
    torch.backends.cudnn.deterministic = policy['cudnn_deterministic']
    torch.backends.cudnn.benchmark = policy['cudnn_benchmark']
    torch.backends.cuda.matmul.allow_tf32 = policy['allow_tf32']
    torch.backends.cudnn.allow_tf32 = policy['allow_tf32']
    return dict(policy)


def make_model(config):
    apply_reproducibility(config)
    from .model import FastPlanner
    cfg = config['model']
    dtype = torch.bfloat16 if config.get('bf16', True) and torch.cuda.is_available() else torch.float32
    model = FastPlanner.from_pretrained(cfg['path'], dtype=dtype, lora_rank=cfg.get('lora_rank', 16),
                                        lora_alpha=cfg.get('lora_alpha', 32),
                                        lora_dropout=cfg.get('lora_dropout', .05),
                                        ego_dim=cfg.get('ego_dim', 3), num_commands=cfg.get('num_commands', 7),
                                        lora_target_suffixes=cfg.get('lora_target_suffixes'),
                                        lora_parameter_dtype=cfg.get('lora_parameter_dtype'))
    return model.to(config.get('device', 'cuda'))


def predict(model, batch):
    return model(batch['native'], batch['ego'], batch['command'], batch['nav'])
