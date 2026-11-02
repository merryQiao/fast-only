"""Native Qwen3.5 vision/mRoPE, sensor embeddings and six planning queries.

The multimodal assembly below follows Qwen3_5Model.forward: native vision
features replace the processor's image placeholders; get_rope_index supplies
their positions. The only extension is nine continuous tokens at the end.
The native language model retains its causal mask and Gated DeltaNet layers.
"""
from __future__ import annotations

import torch
from torch import nn


def append_positions(positions: torch.Tensor, mask: torch.Tensor, count: int) -> torch.Tensor:
    """Continue every native mRoPE axis after the largest nonpadding position."""
    valid = mask.bool().unsqueeze(0).expand_as(positions)
    start = positions.masked_fill(~valid, -1).amax(dim=(0, 2)) + 1
    suffix = start[None, :, None] + torch.arange(count, device=positions.device)[None, None, :]
    return torch.cat((positions, suffix.expand(positions.shape[0], -1, -1)), dim=-1)


def planning_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean over batch and time of |dx|+|dy|, in metres (not mean over xy)."""
    if prediction.shape != target.shape or prediction.shape[-2:] != (6, 2):
        raise ValueError('Expected matching [B,6,2] planning tensors')
    return (prediction.float() - target.float()).abs().sum(-1).mean()


class FastPlanner(nn.Module):
    def __init__(self, backbone, *, lora_rank=16, lora_alpha=32, lora_dropout=.05,
                 ego_dim=3, num_commands=7, lora_target_suffixes=None,
                 lora_parameter_dtype=None):
        super().__init__()
        from peft import LoraConfig, inject_adapter_in_model

        self.backbone = backbone
        self.ego_dim = ego_dim
        backbone.requires_grad_(False)
        # Explicit names exclude vision and all GDN recurrence/decay/gate projections.
        allowed = set(lora_target_suffixes or
                      ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
        self.lora_targets = [name for name, module in backbone.language_model.named_modules()
                             if isinstance(module, nn.Linear) and name.rsplit('.', 1)[-1] in allowed]
        if not self.lora_targets:
            raise ValueError('No verified attention/FFN LoRA targets in language backbone')
        inject_adapter_in_model(LoraConfig(r=lora_rank, lora_alpha=lora_alpha,
                                           lora_dropout=lora_dropout,
                                           target_modules=self.lora_targets, bias='none'),
                                backbone.language_model)
        if lora_parameter_dtype is not None:
            if lora_parameter_dtype != 'float32':
                raise ValueError('Supported trainable LoRA master dtype is float32')
            for name, parameter in backbone.language_model.named_parameters():
                if 'lora_' in name and parameter.requires_grad:
                    parameter.data = parameter.data.float()
        d = backbone.config.text_config.hidden_size
        self.ego_mlp = nn.Sequential(nn.Linear(ego_dim, 256), nn.GELU(), nn.Linear(256, d))
        self.command_embedding = nn.Embedding(num_commands, d)
        self.nav_mlp = nn.Sequential(nn.Linear(2, 256), nn.GELU(), nn.Linear(256, d))
        self.queries = nn.Parameter(torch.randn(6, d) * .02)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 256), nn.GELU(), nn.Linear(256, 2))
        self.last_layout = {}
        self.backbone.visual.eval()

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        from transformers import Qwen3_5Model
        dtype = kwargs.pop('dtype', torch.bfloat16)
        backbone = Qwen3_5Model.from_pretrained(path, dtype=dtype, local_files_only=True,
                                               attn_implementation='sdpa')
        return cls(backbone, **kwargs)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.visual.eval()
        return self

    def forward(self, native, ego, command, nav):
        ids, mask = native['input_ids'], native['attention_mask']
        if not mask[:, -1].bool().all() or (mask[:, 1:] < mask[:, :-1]).any():
            raise ValueError('Native sequences require left padding so appended queries follow observations')
        if ego.shape != (ids.shape[0], self.ego_dim) or nav.shape != (ids.shape[0], 2):
            raise ValueError(f'Expected ego[B,{self.ego_dim}], nav[B,2]')
        grid = native.get('image_grid_thw')
        if grid is None or len(grid) != 4 * ids.shape[0]:
            raise ValueError('Exactly four chronological images per sample are required')
        # Computing embeddings does not train the base token table.
        embeds = self.backbone.get_input_embeddings()(ids)
        with torch.no_grad():
            vision = self.backbone.get_image_features(native['pixel_values'], grid, return_dict=True)
            features = torch.cat(vision.pooler_output, dim=0).to(embeds)
        image_mask, _ = self.backbone.get_placeholder_mask(ids, embeds, image_features=features)
        embeds = embeds.masked_scatter(image_mask, features)
        types = native.get('mm_token_type_ids')
        if types is None:
            # Native processor versions that omit this field: images=1, text=0.
            types = (ids == self.backbone.config.image_token_id).long()
        positions, _ = self.backbone.get_rope_index(ids, mm_token_type_ids=types,
                                                    image_grid_thw=grid, attention_mask=mask)
        positions = append_positions(positions, mask, 9)
        b = ids.shape[0]
        sensors = torch.stack((self.ego_mlp(ego.float()), self.command_embedding(command),
                               self.nav_mlp(nav.float())), dim=1)
        suffix = torch.cat((sensors, self.queries.unsqueeze(0).expand(b, -1, -1)), dim=1).to(embeds)
        embeds = torch.cat((embeds, suffix), dim=1)
        mask = torch.cat((mask, mask.new_ones(b, 9)), dim=-1)
        hidden = self.backbone.language_model(inputs_embeds=embeds, attention_mask=mask,
                                               position_ids=positions, use_cache=False,
                                               return_dict=True).last_hidden_state
        self.last_layout = {
            'sequence_length': embeds.shape[1], 'observation_length': ids.shape[1],
            'query_indices': list(range(ids.shape[1] + 3, ids.shape[1] + 9)),
            'image_grid_thw': grid.detach().cpu().tolist(),
            'image_tokens_per_frame': (grid.prod(-1) // self.backbone.visual.spatial_merge_size**2).tolist(),
            'position_axes': positions.shape[0],
            'query_positions': positions[..., -6:].detach().cpu().tolist(),
        }
        return self.head(hidden[:, -6:].float())

    def gradient_report(self):
        def describe(parameters):
            grads = [p.grad for p in parameters if p.grad is not None]
            return {'with_gradient': len(grads),
                    'finite': bool(grads) and all(torch.isfinite(g).all().item() for g in grads),
                    'nonzero': any(torch.count_nonzero(g).item() > 0 for g in grads)}
        report = {k: describe(module.parameters()) for k, module in (
            ('ego', self.ego_mlp), ('command', self.command_embedding),
            ('nav', self.nav_mlp), ('head', self.head))}
        report['queries'] = describe([self.queries])
        language = self.backbone.language_model
        full = []; gdn_ffn = []
        for layer in language.layers:
            if layer.layer_type == 'full_attention':
                full += [p for n, p in layer.named_parameters() if 'self_attn' in n and 'lora_' in n]
            else:
                gdn_ffn += [p for n, p in layer.named_parameters() if 'mlp' in n and 'lora_' in n]
        report['full_attention_lora'] = describe(full)
        report['gdn_layer_ffn_lora'] = describe(gdn_ffn)
        report['vision_frozen'] = all(not p.requires_grad for p in self.backbone.visual.parameters())
        report['vision_has_gradient'] = any(p.grad is not None for p in self.backbone.visual.parameters())
        return report

    def trainable_state(self):
        return {name: p.detach().cpu() for name, p in self.named_parameters() if p.requires_grad}

    def load_trainable_state(self, state):
        expected = {name for name, p in self.named_parameters() if p.requires_grad}
        if set(state) != expected:
            raise ValueError(f'Checkpoint trainable keys mismatch: missing={expected-set(state)}, extra={set(state)-expected}')
        self.load_state_dict(state, strict=False)
