"""Native Qwen vision/LoRA planner with eight NAVSIM SE(2) queries."""
import torch
from torch import nn
from fast_only.model import FastPlanner, append_positions


def _check_poses(prediction,target):
    if prediction.shape!=target.shape or prediction.ndim!=3 or prediction.shape[1:]!=(8,3):
        raise ValueError('Expected matching finite [B,8,3] NAVSIM poses')
    if not len(prediction) or not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError('Expected nonempty finite NAVSIM poses')


def pose_loss(prediction,target):
    """Official baseline: unweighted mean L1 across batch, poses, x/y/heading.

    XY are metres and heading is radians. Do not wrap only the training loss:
    this matches the official MLP's functional.l1_loss contract exactly.
    """
    _check_poses(prediction,target)
    return torch.nn.functional.l1_loss(prediction.float(),target.float())


def trajectory_metrics(prediction,target):
    prediction,target=prediction.detach().float().cpu(),target.detach().float().cpu()
    _check_poses(prediction,target)
    delta=prediction-target
    xy=torch.linalg.vector_norm(delta[...,:2],dim=-1)
    yaw=torch.atan2(delta[...,2].sin(),delta[...,2].cos()).abs()
    result={'ADE':xy.mean().item(),'FDE':xy[:,-1].mean().item(),
            'loss_l1_pose':delta.abs().mean().item(),
            'loss_l1_xy':delta[...,:2].abs().sum(-1).mean().item(),
            'heading_mae_rad':yaw.mean().item(),'heading_final_mae_rad':yaw[:,-1].mean().item(),
            'samples':len(prediction)}
    result.update({f'L2@{(i+1)*.5:g}s':v for i,v in enumerate(xy.mean(0).tolist())})
    return result


class NavsimPlanner(FastPlanner):
    """Reuse the verified native backbone; replace only dataset-specific tokens/head."""
    def __init__(self,backbone,**kwargs):
        super().__init__(backbone,ego_dim=4,num_commands=4,**kwargs)
        # NAVSIM has command navigation, no continuous target. Remove the
        # inherited B2D projection entirely, rather than feeding zero values.
        del self.nav_mlp
        d=backbone.config.text_config.hidden_size
        self.queries=nn.Parameter(torch.randn(8,d)*.02)
        self.head[-1]=nn.Linear(256,3)

    def forward(self,native,ego,command,nav=None):
        if nav is not None:
            raise ValueError('NAVSIM uses route command only; continuous target is forbidden')
        ids,mask=native['input_ids'],native['attention_mask']
        if not mask[:,-1].bool().all() or (mask[:,1:]<mask[:,:-1]).any():
            raise ValueError('Native sequences require left padding')
        if ego.shape!=(len(ids),4) or command.shape!=(len(ids),):
            raise ValueError('Expected ego[B,4] and scalar command per sample')
        if command.dtype!=torch.long or (command<0).any() or (command>=4).any():
            raise ValueError('Expected native four-way integer command')
        grid=native.get('image_grid_thw')
        if grid is None or len(grid)!=4*len(ids):
            raise ValueError('Exactly four chronological images are required')
        embeds=self.backbone.get_input_embeddings()(ids)
        with torch.no_grad():
            vision=self.backbone.get_image_features(native['pixel_values'],grid,return_dict=True)
            features=torch.cat(vision.pooler_output,dim=0).to(embeds)
        image_mask,_=self.backbone.get_placeholder_mask(ids,embeds,image_features=features)
        embeds=embeds.masked_scatter(image_mask,features)
        types=native.get('mm_token_type_ids')
        if types is None:types=(ids==self.backbone.config.image_token_id).long()
        positions,_=self.backbone.get_rope_index(ids,mm_token_type_ids=types,image_grid_thw=grid,attention_mask=mask)
        positions=append_positions(positions,mask,10)
        sensors=torch.stack((self.ego_mlp(ego.float()),self.command_embedding(command)),dim=1)
        suffix=torch.cat((sensors,self.queries.unsqueeze(0).expand(len(ids),-1,-1)),dim=1).to(embeds)
        embeds=torch.cat((embeds,suffix),dim=1)
        attention=torch.cat((mask,mask.new_ones(len(ids),10)),dim=-1)
        hidden=self.backbone.language_model(inputs_embeds=embeds,attention_mask=attention,
                position_ids=positions,use_cache=False,return_dict=True).last_hidden_state
        self.last_layout={'sequence_length':embeds.shape[1],'observation_length':ids.shape[1],
            'query_indices':list(range(ids.shape[1]+2,ids.shape[1]+10)),
            'image_grid_thw':grid.detach().cpu().tolist(),
            'image_tokens_per_frame':(grid.prod(-1)//self.backbone.visual.spatial_merge_size**2).tolist(),
            'position_axes':positions.shape[0],'query_positions':positions[...,-8:].detach().cpu().tolist(),
            'continuous_target_token':False,'trajectory_shape':[8,3]}
        return self.head(hidden[:,-8:].float())

    def gradient_report(self):
        def describe(params):
            gradients=[p.grad for p in params if p.grad is not None]
            return {'with_gradient':len(gradients),
                    'finite':bool(gradients) and all(torch.isfinite(g).all().item() for g in gradients),
                    'nonzero':any(torch.count_nonzero(g).item()>0 for g in gradients)}
        result={name:describe(module.parameters()) for name,module in
                [('ego',self.ego_mlp),('command',self.command_embedding),('head',self.head)]}
        result['queries']=describe([self.queries])
        full=[];gdn=[]
        for layer in self.backbone.language_model.layers:
            if layer.layer_type=='full_attention':
                full.extend(p for n,p in layer.named_parameters() if 'self_attn' in n and 'lora_' in n)
            else:gdn.extend(p for n,p in layer.named_parameters() if 'mlp' in n and 'lora_' in n)
        result.update(full_attention_lora=describe(full),gdn_layer_ffn_lora=describe(gdn),
            vision_frozen=all(not p.requires_grad for p in self.backbone.visual.parameters()),
            vision_has_gradient=any(p.grad is not None for p in self.backbone.visual.parameters()))
        return result
