"""Optional residual at the existing Fast head; source models stay frozen."""
from contextlib import contextmanager

import torch
from torch import nn


class ResidualFusion(nn.Module):
    """Current Fast conditions select a residual from fixed Reader features.

    Returns features, never a second trajectory. Caller owns the original head.
    Off mode bypasses this module entirely through ``fused_head``.
    """
    def __init__(self,fast_width,reader_width=768,hidden=256,gate_bias=-2.):
        super().__init__()
        for x in [fast_width,reader_width,hidden]:
            if not isinstance(x,int) or isinstance(x,bool) or x<=0:raise ValueError('Widths must be positive integers')
        self.fast_width=fast_width;self.reader_width=reader_width
        self.fast_norm=nn.LayerNorm(fast_width)
        self.reader_norm=nn.LayerNorm(reader_width)
        self.combine=nn.Linear(fast_width+reader_width,hidden)
        self.gate=nn.Linear(hidden,1)
        self.residual=nn.Linear(hidden,fast_width)
        nn.init.zeros_(self.residual.weight);nn.init.zeros_(self.residual.bias)
        nn.init.zeros_(self.gate.weight);nn.init.constant_(self.gate.bias,gate_bias)

    def forward(self,current,source):
        if (current.ndim!=3 or source.ndim!=3 or current.shape[:2]!=source.shape[:2]
                or current.shape[1]!=8 or current.shape[-1]!=self.fast_width or source.shape[-1]!=self.reader_width
                or current.device!=source.device or not current.is_floating_point() or not source.is_floating_point()
                or not bool(torch.isfinite(current).all()) or not bool(torch.isfinite(source).all())):
            raise ValueError('Expected finite paired Fast/Reader features[B,8,width]')
        # Do not detach the fused result: the frozen head must pass its gradient.
        current,source=current.detach().float(),source.detach().float()
        hidden=torch.nn.functional.gelu(self.combine(torch.cat((self.fast_norm(current),self.reader_norm(source)),dim=-1)))
        return current+self.gate(hidden).sigmoid()*self.residual(hidden)


@contextmanager
def fused_head(head,fusion,read_source,*,enabled):
    """Scoped single-head integration. Disabled means no Reader or hook call."""
    if not enabled:
        yield
        return
    calls=[]
    def replace(module,args):
        if calls:raise ValueError('Fusion expects one original head invocation')
        calls.append(True)
        return (fusion(args[0],read_source()),*args[1:])
    handle=head.register_forward_pre_hook(replace)
    try:yield
    finally:handle.remove()


@torch.no_grad()
def reader_features(reader,batch):
    """Capture existing six-way outputs without changing frozen Reader math."""
    captured=[]
    handle=reader.head.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()))
    try:prediction=reader(**batch)
    finally:handle.remove()
    if len(captured)!=1:raise ValueError('Expected one Reader head input')
    return captured[0],prediction
