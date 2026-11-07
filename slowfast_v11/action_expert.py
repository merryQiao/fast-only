"""Random fully trainable regression expert reading frozen native Slow K/V.

This QwenDrive-inspired diagnostic preserves our eight SE(2) targets and Fast
reserved-position convention. It is not a diffusion model or paper reproduction.
"""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5RMSNorm, Qwen3_5TextRotaryEmbedding, apply_rotary_pos_emb, repeat_kv)


class ExpertBlock(nn.Module):
    """Dense pre-norm decoder block, with native per-head norm and output gate."""
    def __init__(self, width, ffn_width, query_heads, kv_heads, head_dim):
        super().__init__()
        self.query_heads=query_heads;self.kv_heads=kv_heads;self.head_dim=head_dim
        self.input_norm=Qwen3_5RMSNorm(width,eps=1e-6)
        self.post_attention_norm=Qwen3_5RMSNorm(width,eps=1e-6)
        self.q_proj=nn.Linear(width,2*query_heads*head_dim,bias=False)
        self.k_proj=nn.Linear(width,kv_heads*head_dim,bias=False)
        self.v_proj=nn.Linear(width,kv_heads*head_dim,bias=False)
        self.q_norm=Qwen3_5RMSNorm(head_dim,eps=1e-6)
        self.k_norm=Qwen3_5RMSNorm(head_dim,eps=1e-6)
        self.o_proj=nn.Linear(query_heads*head_dim,width,bias=False)
        self.gate_proj=nn.Linear(width,ffn_width,bias=False)
        self.up_proj=nn.Linear(width,ffn_width,bias=False)
        self.down_proj=nn.Linear(ffn_width,width,bias=False)

    def forward(self, hidden, source_key, source_value, own_rope, allowed):
        b,n,_=hidden.shape;d=self.head_dim
        normalized=self.input_norm(hidden)
        # Match Qwen3.5's head-interleaved query/gate projection exactly.
        query,gate=self.q_proj(normalized).view(b,n,self.query_heads,2*d).chunk(2,dim=-1)
        query=self.q_norm(query).transpose(1,2)
        key=self.k_norm(self.k_proj(normalized).view(b,n,self.kv_heads,d)).transpose(1,2)
        value=self.v_proj(normalized).view(b,n,self.kv_heads,d).transpose(1,2)
        query,key=apply_rotary_pos_emb(query,key,*(tensor.to(query.dtype) for tensor in own_rope))
        # Frozen source is already KNorm'ed, and receives no projection or norm.
        keys=torch.cat((source_key,key),dim=2)
        values=torch.cat((source_value,value),dim=2)
        repetitions=self.query_heads//self.kv_heads
        result=F.scaled_dot_product_attention(query,repeat_kv(keys,repetitions),
            repeat_kv(values,repetitions),attn_mask=allowed,dropout_p=0.)
        result=result.transpose(1,2).reshape(b,n,-1)*gate.reshape(b,n,-1).sigmoid()
        hidden=hidden+self.o_proj(result)
        normalized=self.post_attention_norm(hidden)
        return hidden+self.down_proj(F.silu(self.gate_proj(normalized))*self.up_proj(normalized))


class ActionExpert(nn.Module):
    """24 fully trainable dense blocks, each reading one of six raw source K/Vs.

    Expert blocks 1..4 use Slow layer4, 5..8 layer8, ..., 21..24 layer24.
    K/V inputs are [B,6,T,kv_heads*head_dim], with heads concatenated per token;
    positions [3,B,T], valid [B,T]. group_ids is accepted but never consumed.
    """
    def __init__(self, text_config=None, *, rope=None, width=512, ffn_width=1536,
                 layers=24, query_heads=8, kv_heads=2, head_dim=256, position_offset=4096):
        super().__init__()
        geometry=(width,ffn_width,layers,query_heads,kv_heads,head_dim,position_offset)
        if (any(isinstance(v,bool) or not isinstance(v,int) or v<=0 for v in geometry)
                or layers!=24 or query_heads%kv_heads):
            raise ValueError('Expert requires 24 blocks and positive compatible head geometry')
        if text_config is not None and rope is not None:
            raise ValueError('Provide a native text configuration or a rotary module, not both')
        if text_config is None and rope is None:
            text_config=Qwen3_5TextConfig(hidden_size=width,num_hidden_layers=layers,
                num_attention_heads=query_heads,num_key_value_heads=kv_heads,head_dim=head_dim,
                max_position_embeddings=262144,rope_parameters=dict(rope_type='default',rope_theta=10000000.,
                    partial_rotary_factor=.25,mrope_section=[11,11,10],mrope_interleaved=True))
        config=text_config if text_config is not None else getattr(rope,'config',None)
        if config is None or config.head_dim!=head_dim or config.num_key_value_heads!=kv_heads:
            raise ValueError('Native source RoPE head geometry differs from the expert')
        self.width=width;self.kv_heads=kv_heads;self.head_dim=head_dim
        self.position_offset=position_offset;self.source_layer_mapping=tuple(i//4 for i in range(layers))
        # Own the tiny rotary module; never retain or mutate a Slow module.
        self.rotary_emb=(Qwen3_5TextRotaryEmbedding(copy.deepcopy(config)) if rope is None else copy.deepcopy(rope))
        self.rotary_emb.float()
        self.ego_mlp=nn.Sequential(nn.Linear(4,256),nn.GELU(),nn.Linear(256,width))
        self.command_embedding=nn.Embedding(4,width)
        self.metadata_mlp=nn.Sequential(nn.Linear(6,256),nn.GELU(),nn.Linear(256,width))
        self.planning_queries=nn.Parameter(torch.randn(8,width)*.02)
        self.blocks=nn.ModuleList(ExpertBlock(width,ffn_width,query_heads,kv_heads,head_dim) for _ in range(layers))
        self.head=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,256),nn.GELU(),nn.Linear(256,3))
        self.apply(self._initialize)
        for block in self.blocks:
            for projection in (block.o_proj,block.down_proj):
                nn.init.normal_(projection.weight,std=.02/math.sqrt(2*layers))

    @staticmethod
    def _initialize(module):
        if isinstance(module,(nn.Linear,nn.Embedding)):
            nn.init.normal_(module.weight,std=.02)
            if isinstance(module,nn.Linear) and module.bias is not None:nn.init.zeros_(module.bias)

    def expert_positions(self, positions):
        """Match Fast's ego/command/metadata/eight-query suffix for this source."""
        if (positions.ndim!=3 or positions.shape[0]!=3 or positions.shape[-1]<18
                or positions.dtype!=torch.long):
            raise ValueError('Source requires integer native positions[3,B,T] with the 18-token Slow suffix')
        first=positions.detach()[:,:,-18:][:,:,:1]
        suffix=first+torch.arange(18,device=positions.device)[None,None]
        if not torch.equal(positions[:,:,-18:],suffix):
            raise ValueError('Slow suffix positions must be consecutive on all three axes')
        return first+self.position_offset+torch.arange(11,device=positions.device)[None,None]

    def _validate(self, keys, values, positions, group_ids, valid, ego, command, metadata):
        if (keys.ndim!=4 or keys.shape[1]!=6 or keys.shape[-1]!=self.kv_heads*self.head_dim
                or values.shape!=keys.shape or not keys.is_floating_point() or not values.is_floating_point()):
            raise ValueError('Expected six matching raw source K/V[B,6,T,Hkv*D]')
        b,_,t,_=keys.shape
        if (b==0 or positions.shape!=(3,b,t) or valid.shape!=(b,t) or valid.dtype!=torch.bool
                or ego.shape!=(b,4) or command.shape!=(b,) or command.dtype!=torch.long
                or metadata.shape!=(b,6) or bool((command<0).any()) or bool((command>=4).any())):
            raise ValueError('Invalid source, sensor, command or metadata layout')
        if group_ids is not None and group_ids.shape!=(b,t):raise ValueError('Invalid optional group layout')
        tensors=(keys,values,positions,valid,ego,command,metadata)
        if any(t.device!=self.planning_queries.device for t in tensors):
            raise ValueError('Expert and inputs must share a device')
        if any(not bool(torch.isfinite(t).all()) for t in (keys,values,ego,metadata)):
            raise ValueError('Expert source and numerical inputs must be finite')

    def forward_features(self, keys, values, positions, group_ids=None, valid=None,
                         ego=None, command=None, metadata=None):
        self._validate(keys,values,positions,group_ids,valid,ego,command,metadata)
        keys,values,positions,valid=(t.detach() for t in (keys,values,positions,valid))
        own_positions=self.expert_positions(positions)
        sensors=torch.stack((self.ego_mlp(ego.detach().to(self.planning_queries.dtype)),
                             self.command_embedding(command),
                             self.metadata_mlp(metadata.detach().to(self.planning_queries.dtype))),dim=1)
        hidden=torch.cat((sensors,self.planning_queries[None].expand(len(keys),-1,-1)),dim=1)
        # Native rotary computes phases in float32, then rounds cos/sin to the
        # actual attention dtype, as in the original Qwen/Fast bridge.
        dtype=(torch.get_autocast_dtype(keys.device.type) if torch.is_autocast_enabled(keys.device.type)
               else self.planning_queries.dtype)
        sample=keys[:,0].float()
        own_rope=self.rotary_emb(sample,own_positions)
        source_rope=tuple(t.to(dtype) for t in self.rotary_emb(sample,positions))
        sources=[]
        for layer in range(6):
            key=keys[:,layer].to(dtype).reshape(len(keys),keys.shape[2],self.kv_heads,self.head_dim).transpose(1,2)
            value=values[:,layer].to(dtype).reshape(len(keys),keys.shape[2],self.kv_heads,self.head_dim).transpose(1,2)
            _,key=apply_rotary_pos_emb(key,key,*source_rope)
            sources.append((key,value))
        causal=torch.ones(11,11,device=keys.device,dtype=torch.bool).tril()
        allowed=torch.cat((valid[:,None].expand(-1,11,-1),causal[None].expand(len(keys),-1,-1)),dim=-1)[:,None]
        for block,source_index in zip(self.blocks,self.source_layer_mapping):
            hidden=block(hidden,*sources[source_index],own_rope,allowed)
        return hidden

    def forward(self, keys, values, positions, group_ids=None, valid=None, ego=None, command=None, metadata=None):
        hidden=self.forward_features(keys,values,positions,group_ids,valid,ego,command,metadata)
        return self.head(hidden[:,-8:])
