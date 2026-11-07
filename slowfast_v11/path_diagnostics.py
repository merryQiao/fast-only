"""Scoped inference hooks for metadata/path and downstream readout diagnostics."""
from contextlib import contextmanager
import torch
from .attention import LAYERS


@contextmanager
def force_no_cache(model):
    if torch.is_grad_enabled() or model.training:raise RuntimeError('Requires eval/no_grad inference')
    def hook(module,args,output):return model.no_cache[None].expand(output.shape[0],-1).to(output)
    handle=model.metadata_encoder.register_forward_hook(hook)
    try:yield
    finally:handle.remove()


@contextmanager
def trace_fast(model,*,full_context=False):
    if torch.is_grad_enabled() or model.training:raise RuntimeError('Requires eval/no_grad inference')
    handles=[];records={};config=model.backbone.config.text_config
    def save(name,value):
        if isinstance(value,(tuple,list)):value=value[0]
        records[name]=value[:,-8:].detach().float().cpu().clone()
        if full_context:records['context_'+name]=value[:,:-8].detach().cpu().clone()
    for index in LAYERS:
        attn=model.backbone.language_model.layers[index].self_attn;number=index+1
        def gate(module,args,output,n=number):
            b,t,_=output.shape
            value=output.view(b,t,config.num_attention_heads,2*config.head_dim).chunk(2,dim=-1)[1].sigmoid()
            save(f'gate_{n}',value.flatten(-2))
        handles.append(attn.q_proj.register_forward_hook(gate))
        handles.append(attn.o_proj.register_forward_pre_hook(lambda module,args,n=number:save(f'post_gate_{n}',args[0])))
        handles.append(attn.o_proj.register_forward_hook(lambda module,args,out,n=number:save(f'post_projection_{n}',out)))
    for index,layer in enumerate(model.backbone.language_model.layers):
        handles.append(layer.register_forward_hook(lambda module,args,out,n=index+1:save(f'layer_{n}',out)))
    handles.append(model.head.register_forward_pre_hook(lambda module,args:save('head_input',args[0])))
    try:yield records
    finally:
        for handle in handles:handle.remove()


def compare_traces(a,b):
    if a.keys()!=b.keys():raise ValueError('Trace keys differ')
    output={}
    for name in a:
        x=a[name].float();y=b[name].float()
        if x.shape!=y.shape:raise ValueError('Trace shapes differ')
        d=x-y
        output[name]=dict(max_abs_difference=float(d.abs().max()) if d.numel() else 0.,
            rms_difference=float(d.square().mean().sqrt()) if d.numel() else 0.,
            relative_l2=float(d.norm()/x.norm().clamp_min(1e-12)),equal=bool(torch.equal(x,y)))
    return output
