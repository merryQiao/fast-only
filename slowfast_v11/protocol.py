"""Typed cache payloads and causal admission; no offline labels enter this module."""
from dataclasses import dataclass, replace
import math
import torch


@dataclass(frozen=True)
class LayerKV:
    key: torch.Tensor  # KNorm output, before RoPE: [B, Hkv, T, D]
    value: torch.Tensor

    def __post_init__(self):
        if self.key.ndim != 4 or self.key.shape != self.value.shape or not self.key.numel():
            raise ValueError('Expected nonempty matching KV[B,Hkv,T,D]')
        if self.key.device != self.value.device or self.key.dtype != self.value.dtype:
            raise ValueError('KV device and dtype must agree')
        if not self.key.is_floating_point() or not self.value.is_floating_point():
            raise ValueError('KV payloads must have floating dtype')
        if not torch.isfinite(self.key).all() or not torch.isfinite(self.value).all():
            raise ValueError('KV payloads must be finite, including masked source columns')


@dataclass(frozen=True)
class SourceMetadata:
    source_time: float
    ready_time: float
    episode_id: str
    navigation_id: str
    model_id: str
    source_pose: tuple[float, float, float]

    def __post_init__(self):
        _finite((self.source_time,self.ready_time,*self.source_pose))
        if len(self.source_pose)!=3 or self.ready_time < self.source_time:
            raise ValueError('Invalid source pose or ready time before source time')
        if not all(isinstance(x,str) and x for x in (self.episode_id,self.navigation_id,self.model_id)):
            raise ValueError('Cache identity fields must be nonempty strings')


@dataclass(frozen=True)
class KVSnapshot:
    layers: tuple[LayerKV, ...]
    position_ids: torch.Tensor  # Original Slow temporal/height/width positions.
    valid_mask: torch.Tensor
    metadata: SourceMetadata | None = None

    def __post_init__(self):
        if not isinstance(self.layers,tuple) or not self.layers:
            raise ValueError('Snapshot needs a nonempty immutable tuple of layers')
        b,_,t,_ = self.layers[0].key.shape
        if self.position_ids.shape != (3,b,t) or self.position_ids.dtype != torch.long:
            raise ValueError('Expected integer source positions[3,B,T]')
        if self.valid_mask.shape != (b,t) or self.valid_mask.dtype != torch.bool:
            raise ValueError('Expected boolean source valid_mask[B,T]')
        shape=self.layers[0].key.shape
        if any(z.key.shape!=shape for z in self.layers):
            raise ValueError('All snapshot layers must have equal KV shapes')
        device=self.layers[0].key.device
        if any(z.key.device!=device for z in self.layers) or self.position_ids.device!=device or self.valid_mask.device!=device:
            raise ValueError('A snapshot must be local to one device')

    def to(self, device):
        """Differentiable device transfer for single-process joint training."""
        return replace(self,layers=tuple(LayerKV(z.key.to(device),z.value.to(device)) for z in self.layers),
            position_ids=self.position_ids.to(device),valid_mask=self.valid_mask.to(device))

    def inference_copy(self, device=None):
        """Detached owned storage. Publisher must not mutate tensors while readers exist."""
        def own(t):return t.detach().to(device or t.device).clone()
        return replace(self,layers=tuple(LayerKV(own(z.key),own(z.value)) for z in self.layers),
            position_ids=own(self.position_ids),valid_mask=own(self.valid_mask))


def _finite(values):
    if not all(math.isfinite(float(x)) for x in values):
        raise ValueError('Times and poses must be finite')


def admit_snapshot(metadata, *, image_time, selection_time, episode_id, navigation_id, model_id):
    """Times must already be mapped to the same observation/runtime clock domain."""
    _finite((image_time,selection_time))
    if selection_time < image_time:
        raise ValueError('Selection cannot precede the selected image')
    return (metadata.source_time <= image_time and metadata.ready_time <= selection_time
        and metadata.episode_id==episode_id and metadata.navigation_id==navigation_id
        and metadata.model_id==model_id)


def relative_source_pose(source_pose, current_pose):
    """P_current^-1 P_source in right-handed x-forward/y-left SE(2)."""
    if len(source_pose)!=3 or len(current_pose)!=3:
        raise ValueError('Expected SE(2) poses (x,y,yaw)')
    _finite((*source_pose,*current_pose))
    sx,sy,sa=source_pose;cx,cy,ca=current_pose
    dx,dy=sx-cx,sy-cy;c,s=math.cos(ca),math.sin(ca)
    return (c*dx+s*dy,-s*dx+c*dy,math.atan2(math.sin(sa-ca),math.cos(sa-ca)))


def cache_metadata(metadata, *, image_time, current_pose, enabled=True, device=None, dtype=torch.float32):
    """Raw SI units: [age,dx,dy,sin(dyaw),cos(dyaw),enabled]; no-cache is zeros."""
    _finite((image_time,*current_pose))
    if metadata is None or not enabled:
        return torch.zeros(6,device=device,dtype=dtype)
    age=image_time-metadata.source_time
    if age<0:
        raise ValueError('Cache source is in the future')
    dx,dy,yaw=relative_source_pose(metadata.source_pose,current_pose)
    return torch.tensor([age,dx,dy,math.sin(yaw),math.cos(yaw),1.],device=device,dtype=dtype)
