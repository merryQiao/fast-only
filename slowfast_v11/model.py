"""Additive NAVSIM planners for bounded V1.1 interface experiments.

No streaming state or admission policy lives in a neural forward. The caller
must admit/fix the snapshot before calling Fast. Stage D passes live tensors;
deployment passes inference_copy() only after complete transfer/publication.
"""
from contextlib import nullcontext
from dataclasses import dataclass
import torch
from torch import nn
from fast_navsim.model import NavsimPlanner
from fast_only.model import append_positions
from .attention import KVBridge
from .protocol import KVSnapshot


def assemble(planner,native,ego,command,suffix,position_offset=0):
    """Existing frozen native vision/merger, then continuous suffix tokens."""
    ids,mask=native['input_ids'],native['attention_mask']
    if ids.ndim!=2 or mask.shape!=ids.shape or not mask[:,-1].bool().all() or (mask[:,1:]<mask[:,:-1]).any():
        raise ValueError('Native inputs require left padding')
    b=ids.shape[0]
    if ego.shape!=(b,4) or command.shape!=(b,) or not torch.isfinite(ego).all():
        raise ValueError('Expected finite ego[B,4] and command[B]')
    if command.dtype!=torch.long or (command<0).any() or (command>=4).any():
        raise ValueError('Expected official NAVSIM integer command 0..3')
    grid=native.get('image_grid_thw')
    if grid is None or len(grid)!=4*b:
        raise ValueError('Exactly four chronological images are required')
    backbone=planner.backbone
    embeds=backbone.get_input_embeddings()(ids)
    with torch.no_grad():
        vision=backbone.get_image_features(native['pixel_values'],grid,return_dict=True)
        features=torch.cat(vision.pooler_output,dim=0).to(embeds)
    image_mask,_=backbone.get_placeholder_mask(ids,embeds,image_features=features)
    embeds=embeds.masked_scatter(image_mask,features)
    types=native.get('mm_token_type_ids',(ids==backbone.config.image_token_id).long())
    positions,_=backbone.get_rope_index(ids,mm_token_type_ids=types,image_grid_thw=grid,attention_mask=mask)
    positions=append_positions(positions,mask,suffix.shape[1])+position_offset
    return (torch.cat((embeds,suffix.to(embeds)),dim=1),positions,
            torch.cat((mask,mask.new_ones(b,suffix.shape[1])),dim=-1))


class FastWithSlowKV(NavsimPlanner):
    def __init__(self,backbone,*,position_offset=4096,**kwargs):
        super().__init__(backbone,**kwargs)
        d=backbone.config.text_config.hidden_size
        self.metadata_encoder=nn.Sequential(nn.Linear(6,256),nn.GELU(),nn.Linear(256,d))
        self.no_cache=nn.Parameter(torch.randn(d)*.02)
        self.position_offset=position_offset
        self.bridge=KVBridge(backbone.language_model,position_offset=position_offset)

    def forward(self,native,ego,command,nav=None,*,snapshot=None,metadata=None,enabled=None):
        if nav is not None:raise ValueError('Continuous target is forbidden in NAVSIM')
        b=len(ego)
        if snapshot is None:
            if metadata is not None or (enabled is not None and enabled.any()):
                raise ValueError('Cache metadata/enabled requires a snapshot')
            enabled=torch.zeros(b,device=ego.device,dtype=torch.bool)
            token=self.no_cache[None].expand(b,-1)
        else:
            enabled=torch.ones(b,device=ego.device,dtype=torch.bool) if enabled is None else enabled
            if enabled.dtype!=torch.bool or enabled.shape!=(b,) or enabled.device!=ego.device:
                raise ValueError('Expected local enabled boolean[B]')
            if metadata is None or metadata.shape!=(b,6) or not torch.isfinite(metadata).all():
                raise ValueError('Snapshot requires finite metadata[B,6]')
            if (metadata[enabled,0]<0).any() or (metadata[enabled,5]!=1).any():
                raise ValueError('Enabled metadata requires nonnegative age and valid marker=1')
            encoded=self.metadata_encoder(metadata.float())
            token=torch.where(enabled[:,None],encoded,self.no_cache[None])
        sensors=torch.stack((self.ego_mlp(ego.float()),self.command_embedding(command),token),dim=1)
        suffix=torch.cat((sensors,self.queries[None].expand(b,-1,-1)),dim=1)
        embeds,positions,mask=assemble(self,native,ego,command,suffix,self.position_offset)
        session=self.bridge.fuse(snapshot,positions,mask,enabled) if snapshot is not None else nullcontext()
        with session:
            hidden=self.backbone.language_model(inputs_embeds=embeds,position_ids=positions,
                attention_mask=mask,use_cache=False,return_dict=True).last_hidden_state
        self.last_layout={'sequence_length':embeds.shape[1],'trajectory_shape':[8,3],
            'suffix_order':['ego','command','cache_metadata','planning_queries_x8'],
            'position_offset':self.position_offset,'query_indices':list(range(embeds.shape[1]-8,embeds.shape[1]))}
        return self.head(hidden[:,-8:].float())


@dataclass
class SlowOutput:
    trajectory: torch.Tensor
    snapshot: KVSnapshot


class SlowPrefill(NavsimPlanner):
    """Full 2B prefill and numeric trajectory feedback; Future/motion heads deferred."""
    def __init__(self,backbone,*,position_offset=4096,**kwargs):
        super().__init__(backbone,**kwargs)
        d=backbone.config.text_config.hidden_size
        self.feedback_encoder=nn.Sequential(nn.Linear(5,256),nn.GELU(),nn.Linear(256,d))
        self.feedback_type=nn.Parameter(torch.randn(d)*.02)
        self.no_feedback=nn.Parameter(torch.randn(d)*.02)
        self.register_buffer('feedback_times',torch.arange(1,9,dtype=torch.float32)*.5)
        self.bridge=KVBridge(backbone.language_model,position_offset=position_offset)

    def forward(self,native,ego,command,nav=None,*,feedback=None):
        if nav is not None:raise ValueError('Continuous target is forbidden in NAVSIM')
        b=len(ego)
        if feedback is None:
            # Same time layout, explicit missing-plan marker at each point.
            numeric=torch.zeros(b,8,5,device=ego.device)
            numeric[...,0]=self.feedback_times
            feedback_tokens=self.feedback_encoder(numeric)+self.no_feedback
        else:
            if feedback.shape!=(b,8,3) or not torch.isfinite(feedback).all():
                raise ValueError('Completed feedback must be finite [B,8,3]')
            times=self.feedback_times[None,:,None].expand(b,-1,-1)
            numeric=torch.cat((times,feedback[...,:2],feedback[...,2:3].sin(),feedback[...,2:3].cos()),dim=-1)
            feedback_tokens=self.feedback_encoder(numeric.float())+self.feedback_type
        sensors=torch.stack((self.ego_mlp(ego.float()),self.command_embedding(command)),dim=1)
        suffix=torch.cat((sensors,feedback_tokens,self.queries[None].expand(b,-1,-1)),dim=1)
        embeds,positions,mask=assemble(self,native,ego,command,suffix)
        with self.bridge.capture(positions,mask) as call:
            hidden=self.backbone.language_model(inputs_embeds=embeds,position_ids=positions,
                attention_mask=mask,use_cache=False,return_dict=True).last_hidden_state
        self.last_layout={'sequence_length':embeds.shape[1],'trajectory_shape':[8,3],
            'feedback_missing':feedback is None,'feedback_times_s':self.feedback_times.tolist(),
            'suffix_order':['ego','command','feedback_x8','planning_queries_x8']}
        return SlowOutput(self.head(hidden[:,-8:].float()),call.snapshot)
