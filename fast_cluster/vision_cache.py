"""Bounded frozen RGB/ViT frame reuse; never language or GDN state.

The owner must change provider_id whenever frozen weights, native buffers, or
the deterministic image processing policy changes. Training uses the same
per-image native execution with no persistent cache.
"""
from collections import OrderedDict
from pathlib import Path
import math

from PIL import Image
import torch


class FrozenFrameCache:
    def __init__(self, provider_id, capacity=8, enabled=True):
        if not isinstance(provider_id, str) or not provider_id:
            raise ValueError('Frozen provider identity must be a nonempty string')
        if type(capacity) is not int or capacity <= 0:
            raise ValueError('Frame cache capacity must be a positive integer')
        self.provider_id, self.capacity, self.enabled = provider_id, capacity, bool(enabled)
        self._frames = OrderedDict()
        self._counts = dict(image_hits=0, image_misses=0, feature_hits=0, feature_misses=0,
                            decode_calls=0, feature_compute_calls=0, evictions=0, invalidations=0)

    def _entry(self, key):
        identity = (self.provider_id, key)
        entry = self._frames.get(identity)
        if entry is not None:
            self._frames.move_to_end(identity)
        return identity, entry

    def _store(self, identity, entry):
        self._frames[identity] = entry
        self._frames.move_to_end(identity)
        while len(self._frames) > self.capacity:
            self._frames.popitem(last=False)
            self._counts['evictions'] += 1

    @staticmethod
    def _stat(path):
        value = path.stat()
        return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)

    def image(self, path):
        path = Path(path).resolve(strict=True)
        before = self._stat(path)
        key = ('frozen_rgb', self.provider_id, str(path), *before)
        identity, entry = self._entry(key) if self.enabled else (None, None)
        if entry is not None and 'rgb' in entry:
            self._counts['image_hits'] += 1
            return entry['rgb'].copy(), key
        self._counts['image_misses'] += 1
        self._counts['decode_calls'] += 1
        with Image.open(path) as source:
            owned = source.convert('RGB')
            owned.load()
        if self._stat(path) != before:
            raise ValueError('Image identity changed during JPEG decoding')
        if self.enabled:
            entry = {} if entry is None else entry
            entry['rgb'] = owned.copy()
            self._store(identity, entry)
        return owned, key

    def feature(self, key, compute):
        identity, entry = self._entry(key) if self.enabled else (None, None)
        if entry is not None and 'feature' in entry:
            self._counts['feature_hits'] += 1
            return entry['feature'].clone()
        self._counts['feature_misses'] += 1
        self._counts['feature_compute_calls'] += 1
        with torch.no_grad():
            value = compute()
        if not isinstance(value, torch.Tensor) or not value.is_floating_point() or not value.numel():
            raise ValueError('Frozen feature provider must return a nonempty floating tensor')
        owned = value.detach().clone()
        if self.enabled:
            entry = {} if entry is None else entry
            entry['feature'] = owned
            self._store(identity, entry)
            return owned.clone()
        return owned

    def clear(self, provider_id=None):
        if provider_id is not None:
            if not isinstance(provider_id, str) or not provider_id:
                raise ValueError('Frozen provider identity must be a nonempty string')
            self.provider_id = provider_id
        self._frames.clear()
        self._counts['invalidations'] += 1

    def stats(self):
        rgb_bytes = sum(entry['rgb'].width*entry['rgb'].height*3
                        for entry in self._frames.values() if 'rgb' in entry)
        feature_bytes = sum(entry['feature'].numel()*entry['feature'].element_size()
                            for entry in self._frames.values() if 'feature' in entry)
        return dict(self._counts, provider_id=self.provider_id, enabled=self.enabled,
            capacity=self.capacity, entries=len(self._frames),
            image_entries=sum('rgb' in entry for entry in self._frames.values()),
            feature_entries=sum('feature' in entry for entry in self._frames.values()),
            rgb_bytes=rgb_bytes, feature_bytes=feature_bytes, total_bytes=rgb_bytes+feature_bytes)


def encode_frames(backbone, native, cache=None, keys=None):
    """Return native per-image pooler tensors, with identical cache-on/off math.

    pixel_values has one row per temporal/spatial patch, before native spatial
    merging. Splits therefore use prod(grid_thw), while output token counts use
    prod(grid_thw) / spatial_merge_size**2. Every miss/off call processes exactly
    one image, even in a multi-sample training batch.
    """
    if any(key == 'trajectory_gt' or key.startswith('future') for key in native):
        raise ValueError('Vision cache inputs are input-only')
    pixels, grids = native.get('pixel_values'), native.get('image_grid_thw')
    if (not isinstance(pixels, torch.Tensor) or pixels.ndim != 2 or not pixels.is_floating_point()
            or not isinstance(grids, torch.Tensor) or grids.ndim != 2 or grids.shape[1] != 3
            or grids.dtype not in (torch.int32, torch.int64) or not len(grids)):
        raise ValueError('Expected native floating pixel_values[patches,width] and integer grid[N,3]')
    values = grids.detach().cpu().tolist()
    if any(any(value <= 0 for value in grid) for grid in values):
        raise ValueError('Native image grids must be positive')
    sizes = [math.prod(grid) for grid in values]
    if sum(sizes) != pixels.shape[0]:
        raise ValueError('Native pixel patch count must equal the sum of image grid products')
    merge = getattr(backbone.visual, 'spatial_merge_size', None)
    if type(merge) is not int or merge <= 0 or any(size % (merge*merge) for size in sizes):
        raise ValueError('Image grids must agree with native spatial merge size')
    if cache is not None and (keys is None or len(keys) != len(sizes)):
        raise ValueError('Frame caching requires one real image identity key per image grid')
    if keys is not None and len(keys) != len(sizes):
        raise ValueError('Frame key count must match native image grids')
    features, offset, width = [], 0, None
    for index, size in enumerate(sizes):
        part, grid = pixels[offset:offset+size], grids[index:index+1]

        def compute(part=part, grid=grid):
            with torch.no_grad():
                output = backbone.get_image_features(pixel_values=part, image_grid_thw=grid, return_dict=True)
            pooled = output.pooler_output
            if not isinstance(pooled, (tuple, list)) or len(pooled) != 1:
                raise ValueError('Per-image native provider must return exactly one pooler tensor')
            return pooled[0]

        feature = compute() if cache is None else cache.feature(keys[index], compute)
        if (not isinstance(feature, torch.Tensor) or feature.ndim != 2
                or feature.shape[0] != size//(merge*merge) or feature.shape[1] <= 0
                or feature.device != pixels.device or not feature.is_floating_point()):
            raise ValueError('Native per-image feature token shape/device is inconsistent with its grid')
        if width is None:
            width = feature.shape[1]
        elif width != feature.shape[1]:
            raise ValueError('Native per-image feature widths must agree')
        features.append(feature)
        offset += size
    return tuple(features)
