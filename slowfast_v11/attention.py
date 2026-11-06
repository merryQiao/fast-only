"""Native Qwen3.5 hooks: capture pre-RoPE KV, fuse only planning-query rows.

The native attention runs once unchanged. A pre-hook on its output projection
replaces the last planning rows, after attention and sigmoid gating but before
the single native output projection. Context never sees the external prefix.
This preserves native projection/dropout calls and costs extra attention work.
Hooks require a non-reentrant, non-checkpointed complete prefill.
"""
from contextlib import contextmanager
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from .protocol import LayerKV, KVSnapshot

LAYERS = (3,7,11,15,19,23)


def _layout(language):
    layers=tuple(i for i,l in enumerate(language.layers) if l.layer_type=='full_attention')
    if len(language.layers)!=24 or layers!=LAYERS:
        raise ValueError('V1.1 requires full 24-layer backbones and FA layers 4,8,12,16,20,24')
    return layers


def check_compatibility(fast,slow):
    _layout(fast);_layout(slow)
    fields=('num_attention_heads','num_key_value_heads','head_dim','rope_parameters')
    for name in fields:
        if getattr(fast.config,name)!=getattr(slow.config,name):
            raise ValueError(f'Fast/Slow incompatible {name}')
    return {'layers_one_based':[i+1 for i in LAYERS],
        'kv_heads':fast.config.num_key_value_heads,'head_dim':fast.config.head_dim,
        'fast_width':fast.config.hidden_size,'slow_width':slow.config.hidden_size}


def planning_attention(q,k,v,source_k,source_v,fast_valid,source_valid,dropout=0.):
    """Query rows are the final Q positions; inputs already have RoPE applied.

    Bool mask is [B,1,Q,S+T], never the quadratic full-context block mask.
    Invalid external columns are masked in SDPA, never simulated by zero values.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import repeat_kv
    b,h,q_len,d=q.shape;t=k.shape[2]
    if h%k.shape[1] or q_len>t:
        raise ValueError('Invalid GQA or planning query shape')
    causal=torch.arange(t,device=q.device)[None,:]<=torch.arange(t-q_len,t,device=q.device)[:,None]
    allowed=torch.cat((source_valid[:,None,:].expand(b,q_len,-1),fast_valid[:,None,:]&causal),dim=-1)
    keys=repeat_kv(torch.cat((source_k,k),dim=2),h//k.shape[1])
    values=repeat_kv(torch.cat((source_v,v),dim=2),h//k.shape[1])
    return F.scaled_dot_product_attention(q,keys,values,attn_mask=allowed[:,None],dropout_p=dropout)


class KVBridge:
    def __init__(self,language,*,planning_queries=8,position_offset=4096):
        _layout(language)
        self.language=language;self.queries=planning_queries;self.offset=position_offset
        self._call=None;self._handles=[]
        self._handles.append(language.register_forward_pre_hook(self._native_prefill,with_kwargs=True))
        for index in LAYERS:
            attn=language.layers[index].self_attn
            for field,module in [('q_proj',attn.q_proj),('q',attn.q_norm),('k',attn.k_norm),('v',attn.v_proj)]:
                self._handles.append(module.register_forward_hook(self._record(index,field)))
            self._handles.append(attn.o_proj.register_forward_pre_hook(self._fuse_rows(index,attn)))

    def _native_prefill(self,module,args,kwargs):
        call=self._call
        if call is None:return
        if args:
            raise ValueError('Bridge requires keyword prefill arguments')
        if kwargs.get('past_key_values') is not None or kwargs.get('use_cache') is not False:
            raise ValueError('Bridge forbids native cache; pass use_cache=False explicitly')
        if kwargs.get('output_attentions',False):
            raise ValueError('External attention weights are not exposed by this hook prototype')
        positions=kwargs.get('position_ids')
        if positions is None or not torch.equal(positions,call.positions):
            raise ValueError('Native positions must match the fixed bridge session positions')
        mask=kwargs.get('attention_mask')
        if mask is None or not torch.equal(mask.bool(),call.mask):
            raise ValueError('Native attention mask must match the bridge session')
        if call.forwards:
            raise RuntimeError('A bridge session permits exactly one complete native prefill')
        call.forwards+=1

    def _record(self,index,field):
        def hook(module,args,output):
            if self._call is not None:
                self._call.values.setdefault(index,{})[field]=output
        return hook

    def _fuse_rows(self,index,attn):
        def hook(module,args):
            call=self._call
            if call is None:return
            values=call.values[index]
            if call.mode=='capture':
                b,t,_,d=values['k'].shape
                call.layers[index]=LayerKV(values['k'].transpose(1,2),values['v'].view(b,t,-1,d).transpose(1,2))
                del call.values[index]
                return
            if not call.enabled.any():
                del call.values[index]
                return
            from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
            b,t,h,d=values['q'].shape
            q=values['q'].transpose(1,2);k=values['k'].transpose(1,2)
            v=values['v'].view(b,t,-1,d).transpose(1,2)
            q,k=apply_rotary_pos_emb(q,k,*call.fast_rope)
            source=call.snapshot.layers[LAYERS.index(index)]
            _,sk=apply_rotary_pos_emb(source.key,source.key,*call.source_rope)
            external_valid=call.snapshot.valid_mask & call.enabled[:,None]
            fused=planning_attention(q[:,:,-self.queries:],k,v,sk,source.value,
                call.mask,external_valid,attn.attention_dropout if attn.training else 0.)
            gate=values['q_proj'].view(b,t,h,d*2).chunk(2,dim=-1)[1]
            fused=fused.transpose(1,2).reshape(b,self.queries,-1)*gate[:,-self.queries:].reshape(b,self.queries,-1).sigmoid()
            original=args[0]
            rows=torch.where(call.enabled[:,None,None],fused,original[:,-self.queries:])
            replacement=torch.cat((original[:,:-self.queries],rows),dim=1)
            del call.values[index]
            call.fused_layers.append(index)
            return (replacement,*args[1:])
        return hook

    @contextmanager
    def _session(self,mode,positions,mask,snapshot=None,enabled=None):
        if self._call is not None:
            raise RuntimeError('KV bridge does not support concurrent or nested calls')
        if getattr(self.language,'gradient_checkpointing',False):
            raise ValueError('Hook KV bridge does not support gradient checkpointing')
        if positions.shape!=(3,*mask.shape) or mask.shape[1]<self.queries:
            raise ValueError('Invalid positions or token mask')
        if mode=='capture':
            selected=positions.masked_select(mask.bool()[None].expand_as(positions))
            if (selected<0).any() or (selected>=self.offset).any():
                raise ValueError('Source positions exceed the reserved Fast position range')
        if mode=='fuse':
            if enabled.dtype!=torch.bool or enabled.shape!=(mask.shape[0],):
                raise ValueError('Expected enabled boolean[B]')
            if len(snapshot.layers)!=6 or snapshot.valid_mask.shape[0]!=mask.shape[0]:
                raise ValueError('Snapshot requires matching batch and six layers')
            expected=(mask.shape[0],self.language.config.num_key_value_heads,
                      snapshot.valid_mask.shape[1],self.language.config.head_dim)
            if any(z.key.shape!=expected for z in snapshot.layers):
                raise ValueError('Incompatible external KV shape')
            if snapshot.valid_mask.device!=mask.device or enabled.device!=mask.device:
                raise ValueError('External snapshot and gate must be local to Fast')
            if (positions.masked_select(mask.bool()[None].expand_as(positions))<self.offset).any():
                raise ValueError('Fast positions must use fixed reserved offset')
            source_pos=snapshot.position_ids.masked_select(snapshot.valid_mask[None].expand_as(snapshot.position_ids))
            if (source_pos<0).any() or (source_pos>=self.offset).any():
                raise ValueError('Source positions outside reserved prefix')
        call=SimpleNamespace(mode=mode,positions=positions,mask=mask.bool(),values={},layers={},
                             snapshot=snapshot,enabled=enabled,fused_layers=[],forwards=0)
        if mode=='fuse':
            sample=snapshot.layers[0].key
            call.fast_rope=self.language.rotary_emb(sample,positions)
            call.source_rope=self.language.rotary_emb(sample,snapshot.position_ids)
        self._call=call
        try:
            yield call
            if mode=='capture':
                if tuple(call.layers)!=LAYERS:raise RuntimeError('Incomplete Slow prefill')
                call.snapshot=KVSnapshot(tuple(call.layers[i] for i in LAYERS),positions,mask.bool())
            elif enabled.any() and tuple(call.fused_layers)!=LAYERS:
                raise RuntimeError('Incomplete Fast fusion')
        finally:
            self._call=None
            call.values.clear()

    def capture(self,positions,mask):return self._session('capture',positions,mask)

    def fuse(self,snapshot,positions,mask,enabled):
        return self._session('fuse',positions,mask,snapshot,enabled)

    def close(self):
        if self._call is not None:raise RuntimeError('Cannot remove active KV hooks')
        for handle in self._handles:handle.remove()
        self._handles.clear()
