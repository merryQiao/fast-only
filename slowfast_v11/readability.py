"""Small diagnostic readers over frozen, unpooled Slow representations.

These modules never own or update Slow/Fast parameters. RMS scales are supplied
by the caller from TRAIN only; positive scalar division retains raw information.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


class KVReader(nn.Module):
    """Six parallel full-token attentions with explicit position/group priors.

    Inputs: K/V [B,L,T,D], positions [3,B,T], group IDs [B,T] and valid [B,T].
    Outputs: eight SE(2) poses [B,8,3] with the default query count. Group-count
    correction supplies equal group *prior* mass for equal content logits; it
    does not force equal posterior mass when actual K/Q content differs.
    """
    def __init__(self, kv_width=512, hidden=128, layers=6, queries=8, groups=9):
        super().__init__()
        self.kv_width = _positive_integer(kv_width, 'kv_width')
        self.hidden = _positive_integer(hidden, 'hidden')
        self.layers = _positive_integer(layers, 'layers')
        self.queries = _positive_integer(queries, 'queries')
        self.groups = _positive_integer(groups, 'groups')
        self.key_projections = nn.ModuleList(nn.Linear(kv_width, hidden, bias=False) for _ in range(layers))
        self.value_projections = nn.ModuleList(nn.Linear(kv_width, hidden, bias=False) for _ in range(layers))
        self.time_queries = nn.Parameter(torch.randn(layers, queries, hidden)*.02)
        self.position_projection = nn.Linear(96, hidden, bias=False)
        self.group_bias = nn.Parameter(torch.zeros(layers, groups))
        # Sixteen frequencies times sin/cos gives 32 dimensions for each axis.
        self.register_buffer('position_frequencies', torch.exp(-math.log(10000.)*torch.arange(16)/16))
        self.register_buffer('key_scales', torch.ones(layers))
        self.register_buffer('value_scales', torch.ones(layers))
        self.head = nn.Sequential(nn.Linear(layers*hidden, 256), nn.GELU(), nn.Linear(256, 3))

    @torch.no_grad()
    def set_scales(self, key_scales, value_scales):
        """Install six TRAIN-derived positive finite RMS scalars atomically."""
        candidates = []
        for supplied, buffer in ((key_scales, self.key_scales), (value_scales, self.value_scales)):
            value = torch.as_tensor(supplied, device=buffer.device, dtype=buffer.dtype).detach()
            if value.shape != (self.layers,) or not bool(torch.isfinite(value).all()) or bool((value <= 0).any()):
                raise ValueError('RMS scales must contain one positive finite scalar per layer')
            candidates.append(value.clone())
        self.key_scales.copy_(candidates[0]); self.value_scales.copy_(candidates[1])

    def _validate(self, keys, values, positions, group_ids, valid):
        if (keys.ndim != 4 or values.shape != keys.shape or keys.shape[1] != self.layers
                or keys.shape[-1] != self.kv_width or not keys.is_floating_point()
                or not values.is_floating_point()):
            raise ValueError('Expected matching floating K/V[B,layers,T,kv_width]')
        b, _, t, _ = keys.shape
        if b == 0 or t == 0 or positions.shape != (3, b, t) or group_ids.shape != (b, t) or valid.shape != (b, t):
            raise ValueError('Invalid position, group or validity layout')
        if (valid.dtype != torch.bool or group_ids.dtype not in (torch.int32, torch.int64)
                or bool((group_ids < 0).any()) or bool((group_ids >= self.groups).any())
                or not bool(valid.any(-1).all())):
            raise ValueError('Expected valid token(s) and integer group IDs in range')
        if any(tensor.device != self.time_queries.device for tensor in (keys, values, positions, group_ids, valid)):
            raise ValueError('Reader and source tensors must share a device')
        if any(not bool(torch.isfinite(tensor).all()) for tensor in (keys, values, positions)):
            raise ValueError('Source K/V and positions must be finite')

    def forward(self, keys, values, positions, group_ids, valid):
        self._validate(keys, values, positions, group_ids, valid)
        keys, values, positions = keys.detach(), values.detach(), positions.detach()
        group_ids, valid = group_ids.detach().long(), valid.detach()
        dtype = self.time_queries.dtype
        # [B,T,3,16] -> [B,T,96]; no token pooling or implicit token ordering.
        angles = positions.permute(1, 2, 0).float()[..., None]*self.position_frequencies.float()
        features = torch.cat((angles.sin(), angles.cos()), dim=-1).flatten(-2)
        position_keys = self.position_projection(features.to(dtype))
        counts = torch.zeros(len(keys), self.groups, device=keys.device, dtype=torch.long)
        counts.scatter_add_(1, group_ids, valid.long())
        token_counts = counts.gather(1, group_ids).clamp_min(1).to(dtype)
        prior = -token_counts.log()
        outputs = []
        for layer in range(self.layers):
            projected_keys = self.key_projections[layer](keys[:, layer].to(dtype)/self.key_scales[layer]) + position_keys
            projected_values = self.value_projections[layer](values[:, layer].to(dtype)/self.value_scales[layer])
            bias = prior + self.group_bias[layer][group_ids]
            bias = bias.masked_fill(~valid, float('-inf'))[:, None, None, :]
            query = self.time_queries[layer][None, None].expand(len(keys), 1, -1, -1)
            output = F.scaled_dot_product_attention(query, projected_keys[:, None], projected_values[:, None],
                                                    attn_mask=bias, dropout_p=0.)
            outputs.append(output[:, 0])
        return self.head(torch.cat(outputs, dim=-1))


class FinalFeatureReader(nn.Module):
    """New original-topology head over eight frozen Slow planning features."""
    def __init__(self, width, hidden=256):
        super().__init__()
        self.width = _positive_integer(width, 'width')
        _positive_integer(hidden, 'hidden')
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, 3))

    def forward(self, features):
        if (features.ndim != 3 or features.shape[1:] != (8, self.width) or features.shape[0] == 0
                or not features.is_floating_point() or not bool(torch.isfinite(features).all())):
            raise ValueError('Expected finite frozen final features[B,8,width]')
        if features.device != self.head[0].weight.device:
            raise ValueError('Reader and final features must share a device')
        return self.head(features.detach().to(self.head[0].weight.dtype))
