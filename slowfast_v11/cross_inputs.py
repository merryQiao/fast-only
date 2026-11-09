"""Deterministic cross-recording donors and frozen visual-only replacement.

These helpers neither select on labels nor run a model. The runner authenticates
the canonical VAL256 cohort; donor construction also supports smaller equal-size
validation groups for CPU checks.
"""
from collections.abc import Mapping

import torch

from slowfast_v11.adapter_data import recording_group


def donor_maps(rows, offsets=(1, 4, 7)):
    """Map each VAL token to the same manifest index in a shifted recording.

    Recording identities are sorted, while rows inside each recording retain
    their original manifest order. Only token, route and split are read. Equal
    nonempty group sizes and distinct positive offsets guarantee cross-recording
    bijections with a different donor for every requested offset.
    """
    groups = {}
    tokens = set()
    for row in rows:
        if not isinstance(row, Mapping) or row.get('split') != 'val':
            raise ValueError('Donors require validation rows')
        token, route = row.get('token'), row.get('route')
        if not isinstance(token, str) or not token or token in tokens:
            raise ValueError('Donor tokens must be nonempty and unique')
        if not isinstance(route, str) or not route:
            raise ValueError('Donors require a recording route')
        tokens.add(token)
        groups.setdefault(recording_group(route), []).append(token)
    names = sorted(groups)
    count = len(names)
    if count < 2 or len({len(group) for group in groups.values()}) != 1:
        raise ValueError('Donors require equal nonempty recording groups')
    offsets = tuple(offsets)
    if (not offsets or any(type(offset) is not int or not 0 < offset < count for offset in offsets)
            or len(set(offsets)) != len(offsets)):
        raise ValueError('Donor offsets must be distinct positive integers below the group count')
    return {offset: {target: groups[names[(index + offset) % count]][position]
                     for index, name in enumerate(names)
                     for position, target in enumerate(groups[name])}
            for offset in offsets}


def _validate_packet(packet):
    if not isinstance(packet, Mapping) or not {'token', 'batch', 'vision'} <= packet.keys():
        raise ValueError('Expected a complete frozen visual packet')
    batch = packet['batch']
    if not isinstance(batch, Mapping) or set(batch) != {'native', 'ego', 'command'}:
        raise ValueError('Cached batch must contain only native, ego and command')
    native = batch['native']
    required = {'input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw'}
    optional = {'token_type_ids', 'mm_token_type_ids'}
    if (not isinstance(native, Mapping) or not required <= native.keys()
            or set(native) - required - optional):
        raise ValueError('Only complete native processor fields may be cached')
    if any(not isinstance(value, torch.Tensor) or value.layout != torch.strided
           for value in native.values()):
        raise ValueError('Native processor fields must be dense tensors')
    if native['pixel_values'].numel():
        raise ValueError('Frozen visual outputs require an empty pixel sentinel')
    ids = native['input_ids']
    if (ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0
            or native['attention_mask'].shape != ids.shape
            or any(native[key].shape != ids.shape for key in optional & native.keys())
            or native['image_grid_thw'].shape != (4, 3)):
        raise ValueError('Expected the native single-window four-image token layout')
    vision = packet['vision']
    if not isinstance(vision, tuple) or len(vision) != 4:
        raise ValueError('Expected a complete chronological four-frame visual tuple')
    if any(not isinstance(frame, torch.Tensor) or frame.layout != torch.strided
           or frame.ndim != 2 or min(frame.shape) == 0 or not frame.is_floating_point()
           or frame.requires_grad for frame in vision):
        raise ValueError('Expected dense, nonempty, frozen floating visual outputs')


def _same_layout(target, donor):
    return (target.shape == donor.shape and target.dtype == donor.dtype
            and target.device == donor.device and target.layout == donor.layout
            and target.stride() == donor.stride())


def swap_vision(target_packet, donor_packet):
    """Return a shallow target packet with only its four visual outputs replaced.

    The exact target batch object, Ego/navigation tensors and all target packet
    metadata survive. Native IDs, masks, grids and optional type fields must
    match in value and layout; feature tensors must match in layout and remain
    finite and frozen. Nothing is resized, repacked, cloned or written.
    """
    _validate_packet(target_packet)
    _validate_packet(donor_packet)
    target_native = target_packet['batch']['native']
    donor_native = donor_packet['batch']['native']
    if target_native.keys() != donor_native.keys():
        raise ValueError('Native processor field sets differ')
    for key in target_native:
        target, donor = target_native[key], donor_native[key]
        if not _same_layout(target, donor) or not torch.equal(target, donor):
            raise ValueError(f'Native processor layout/value differs: {key}')
    for index, (target, donor) in enumerate(zip(target_packet['vision'], donor_packet['vision'])):
        if not _same_layout(target, donor):
            raise ValueError(f'Frozen visual layout differs at frame {index}')
        if not torch.isfinite(target).all().item() or not torch.isfinite(donor).all().item():
            raise ValueError(f'Frozen visual outputs must be finite at frame {index}')
    return dict(target_packet, vision=donor_packet['vision'])
