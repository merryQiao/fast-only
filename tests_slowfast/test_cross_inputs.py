"""GT-free donors and a complete, frozen four-frame-only intervention."""
import copy
import importlib
import importlib.util

import pytest
import torch

from slowfast_v11.adapter_data import recording_group


def api():
    assert importlib.util.find_spec('slowfast_v11.cross_inputs') is not None, 'cross-input helper is missing'
    return importlib.import_module('slowfast_v11.cross_inputs')


def rows(groups=8, windows=3):
    # Interleaved groups and nonlexical tokens expose accidental token sorting.
    return [dict(token=f'g{group}_{windows-index:03}',
                 route=f'recording_{group}_000_999', split='val',
                 target=[[index, group, 0.]], prediction=[[0., 0., 0.]],
                 ego=[float(index)], command=group % 3)
            for index in range(windows) for group in reversed(range(groups))]


def packet(token, sign=1., *, shape=(2, 8), dtype=torch.float32):
    ids = torch.tensor([[7, 11, 12, 12, 13, 8, 11, 12, 12, 13,
                         9, 11, 12, 12, 13, 10, 11, 12, 12, 13]])
    return dict(token=token,
                batch=dict(native=dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                                       image_grid_thw=torch.tensor([[1, 2, 4]] * 4),
                                       pixel_values=torch.empty(0),
                                       token_type_ids=torch.zeros_like(ids),
                                       mm_token_type_ids=(ids == 12).long()),
                           ego=torch.tensor([[1., 2., 3., 4.]]) * sign,
                           command=torch.tensor([0 if sign > 0 else 2])),
                vision=tuple(torch.full(shape, sign * (index + 1), dtype=dtype)
                             for index in range(4)),
                original_prediction=torch.full((1, 8, 3), sign),
                image_audit=[dict(time=t, source=token) for t in (-1.5, -1., -.5, 0.)],
                metadata={'source': token})


def snapshot(value):
    """Record values and tensor versions, including NaN bytes, without writes."""
    if isinstance(value, torch.Tensor):
        bits = None if value.device.type == 'meta' else value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        return (id(value), value._version, value.shape, value.dtype,
                value.device, value.layout, value.stride(), value.requires_grad, bits)
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return (type(value), [snapshot(item) for item in value])
    return copy.deepcopy(value)


def test_three_maps_are_cross_recording_bijections_with_distinct_sources():
    manifest = rows(windows=32)
    before = copy.deepcopy(manifest)
    maps = api().donor_maps(manifest)
    assert list(maps) == [1, 4, 7]
    tokens = {row['token'] for row in manifest}
    groups = {row['token']: recording_group(row['route']) for row in manifest}
    for mapping in maps.values():
        assert set(mapping) == set(mapping.values()) == tokens
        assert len(mapping) == 256
        assert all(groups[target] != groups[donor] for target, donor in mapping.items())
    assert all(len({maps[offset][token] for offset in maps}) == 3 for token in tokens)
    assert manifest == before


def test_offsets_follow_manifest_index_and_reproduce_old_offset_four_fixture():
    manifest = rows()
    groups = {route: [row for row in manifest if row['route'] == route]
              for route in sorted({row['route'] for row in manifest})}
    routes = list(groups)
    # Exact historical tools/diagnose_direct_full.py rule on this VAL fixture.
    old = {row['token']: groups[routes[(index + 4) % 8]][position]['token']
           for index, route in enumerate(routes) for position, row in enumerate(groups[route])}
    maps = api().donor_maps(manifest)
    assert maps[4] == old
    assert maps[1]['g0_003'] == 'g1_003'
    assert maps[7]['g0_001'] == 'g7_001'


def test_recording_identity_combines_segments_and_retains_their_manifest_order():
    manifest = rows(groups=3, windows=2)
    for index, row in enumerate(manifest):
        row['route'] = f"recording_{row['token'][1]}_{index:03}_{index+10:03}"
    maps = api().donor_maps(manifest, offsets=(1, 2))
    assert maps[1]['g0_002'] == 'g1_002'
    assert maps[2]['g0_001'] == 'g2_001'


def test_gt_predictions_ego_commands_and_extraneous_fields_cannot_select_donors():
    manifest = rows()
    expected = api().donor_maps(manifest)
    changed = [dict(row, target=None, trajectory_gt={'nonsense': 10}, prediction='opposite',
                    ego=[-999.], command=99, speed_mps=float('nan')) for row in manifest]
    assert api().donor_maps(changed) == expected
    identity_only = [{key: row[key] for key in ('token', 'route', 'split')} for row in manifest]
    assert api().donor_maps(identity_only) == expected


@pytest.mark.parametrize('offsets', [(), (1, 1), (0,), (-1,), (8,), (True,), (1.0,), ('1',)])
def test_invalid_offsets_fail_instead_of_reusing_a_donor(offsets):
    with pytest.raises(ValueError):
        api().donor_maps(rows(), offsets=offsets)


@pytest.mark.parametrize('problem', ['empty', 'one_group', 'unequal', 'duplicate',
                                    'train', 'missing_split', 'empty_token', 'missing_route'])
def test_donor_cohort_must_be_unique_validation_and_complete_equal_groups(problem):
    manifest = rows()
    if problem == 'empty':
        manifest = []
    elif problem == 'one_group':
        manifest = rows(groups=1)
    elif problem == 'unequal':
        manifest.pop()
    elif problem == 'duplicate':
        manifest[1]['token'] = manifest[0]['token']
    elif problem == 'train':
        manifest[0]['split'] = 'train'
    elif problem == 'missing_split':
        del manifest[0]['split']
    elif problem == 'empty_token':
        manifest[0]['token'] = ''
    elif problem == 'missing_route':
        del manifest[0]['route']
    before = copy.deepcopy(manifest)
    with pytest.raises(ValueError):
        api().donor_maps(manifest)
    assert manifest == before


def test_swap_replaces_all_four_chronological_features_and_keeps_exact_target_batch():
    target, donor = packet('target'), packet('opposite', -1.)
    before = snapshot((target, donor))
    result = api().swap_vision(target, donor)
    assert result is not target and result is not donor
    assert result['batch'] is target['batch']
    assert result['vision'] is donor['vision']
    assert [frame[0, 0].item() for frame in result['vision']] == [-1., -2., -3., -4.]
    assert result['token'] == 'target'
    for key in target.keys() - {'vision'}:
        assert result[key] is target[key]
    assert torch.equal(result['batch']['ego'], torch.tensor([[1., 2., 3., 4.]]))
    assert torch.equal(result['batch']['command'], torch.tensor([0]))
    assert not torch.equal(result['batch']['ego'], donor['batch']['ego'])
    assert not torch.equal(result['batch']['command'], donor['batch']['command'])
    assert snapshot((target, donor)) == before


def test_swap_accepts_real_four_by_2040_by_1024_bfloat16_cache_shape():
    target = packet('target', shape=(2040, 1024), dtype=torch.bfloat16)
    donor = packet('donor', -1., shape=(2040, 1024), dtype=torch.bfloat16)
    for item in (target, donor):
        item['batch']['native']['image_grid_thw'] = torch.tensor([[1, 60, 136]] * 4)
    result = api().swap_vision(target, donor)
    assert result['vision'] is donor['vision']
    assert all(frame.shape == (2040, 1024) for frame in result['vision'])
    assert result['batch'] is target['batch']


def test_optional_processor_token_type_fields_can_be_absent_in_both_packets():
    target, donor = packet('target'), packet('donor', -1.)
    for item in (target, donor):
        del item['batch']['native']['token_type_ids']
        del item['batch']['native']['mm_token_type_ids']
    assert api().swap_vision(target, donor)['vision'] is donor['vision']


@pytest.mark.parametrize('side', ['target', 'donor'])
@pytest.mark.parametrize('field', ['input_ids', 'attention_mask', 'image_grid_thw',
                                  'token_type_ids', 'mm_token_type_ids'])
def test_native_layout_values_must_be_exact_and_errors_leave_both_inputs_unchanged(side, field):
    target, donor = packet('target'), packet('donor', -1.)
    item = target if side == 'target' else donor
    item['batch']['native'][field].view(-1)[0] += 1
    before = snapshot((target, donor))
    with pytest.raises(ValueError):
        api().swap_vision(target, donor)
    assert snapshot((target, donor)) == before


@pytest.mark.parametrize('problem', ['missing_optional', 'extra_optional', 'unsupported_native',
                                    'missing_required', 'native_dtype', 'native_shape',
                                    'native_device', 'nonempty_pixels', 'extra_batch',
                                    'vision_shape', 'vision_dtype', 'vision_device', 'vision_stride',
                                    'three_frames', 'five_frames', 'list_frames', 'trainable',
                                    'nan', 'infinity', 'integer_vision', 'grid_frame_count'])
@pytest.mark.parametrize('side', ['target', 'donor'])
def test_incompatible_or_unfrozen_packets_fail_without_repacking_or_mutation(problem, side):
    target, donor = packet('target'), packet('donor', -1.)
    item = target if side == 'target' else donor
    native = item['batch']['native']
    if problem == 'missing_optional':
        del native['token_type_ids']
    elif problem == 'extra_optional':
        del (donor if side == 'target' else target)['batch']['native']['mm_token_type_ids']
    elif problem == 'unsupported_native':
        native['inputs_embeds'] = torch.ones(1, 20, 8)
    elif problem == 'missing_required':
        del native['attention_mask']
    elif problem == 'native_dtype':
        native['input_ids'] = native['input_ids'].to(torch.int32)
    elif problem == 'native_shape':
        native['input_ids'] = native['input_ids'].flatten()
    elif problem == 'native_device':
        native['input_ids'] = native['input_ids'].to('meta')
    elif problem == 'nonempty_pixels':
        native['pixel_values'] = torch.ones(1)
    elif problem == 'extra_batch':
        item['batch']['trajectory_gt'] = torch.ones(1, 8, 3)
    elif problem == 'vision_shape':
        item['vision'] = (*item['vision'][:3], torch.ones(3, 8))
    elif problem == 'vision_dtype':
        item['vision'] = tuple(frame.to(torch.bfloat16) for frame in item['vision'])
    elif problem == 'vision_device':
        item['vision'] = tuple(frame.to('meta') for frame in item['vision'])
    elif problem == 'vision_stride':
        item['vision'] = tuple(torch.ones(8, 2).t() for _ in range(4))
    elif problem == 'three_frames':
        item['vision'] = item['vision'][:3]
    elif problem == 'five_frames':
        item['vision'] = (*item['vision'], item['vision'][0])
    elif problem == 'list_frames':
        item['vision'] = list(item['vision'])
    elif problem == 'trainable':
        item['vision'][0].requires_grad_()
    elif problem == 'nan':
        item['vision'][0][0, 0] = float('nan')
    elif problem == 'infinity':
        item['vision'][0][0, 0] = float('inf')
    elif problem == 'integer_vision':
        item['vision'] = tuple(frame.long() for frame in item['vision'])
    elif problem == 'grid_frame_count':
        native['image_grid_thw'] = native['image_grid_thw'][:3]
    before = snapshot((target, donor))
    with pytest.raises(ValueError):
        api().swap_vision(target, donor)
    assert snapshot((target, donor)) == before
