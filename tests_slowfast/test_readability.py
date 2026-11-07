"""CPU semantic checks for frozen-source diagnostic readouts."""
import pytest
import torch


def inputs(batch=2, tokens=13, width=8):
    generator = torch.Generator().manual_seed(9)
    keys = torch.randn(batch, 6, tokens, width, generator=generator, requires_grad=True)
    values = torch.randn(batch, 6, tokens, width, generator=generator, requires_grad=True)
    positions = torch.arange(3*batch*tokens).reshape(3, batch, tokens)
    groups = torch.arange(tokens)[None].expand(batch, -1) % 9
    valid = torch.ones(batch, tokens, dtype=torch.bool)
    return keys, values, positions, groups, valid


def reader():
    from slowfast_v11.readability import KVReader
    torch.manual_seed(42)
    return KVReader(kv_width=8, hidden=16)


def test_all_six_layers_receive_gradients_while_sources_stay_frozen():
    model = reader(); data = inputs()
    before = [x.detach().clone() for x in data]
    prediction = model(*data)
    assert prediction.shape == (2, 8, 3) and torch.isfinite(prediction).all()
    prediction.square().mean().backward()
    for projections in (model.key_projections, model.value_projections):
        assert len(projections) == 6
        for projection in projections:
            gradient = projection.weight.grad
            assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert model.position_projection.weight.grad.abs().sum() > 0
    assert model.group_bias.grad.abs().sum() > 0
    assert data[0].grad is None and data[1].grad is None
    assert all(torch.equal(x, old) for x, old in zip(data, before))


def test_joint_token_permutation_preserves_readout_but_content_swap_changes_it():
    model = reader().eval(); keys, values, positions, groups, valid = inputs()
    permutation = torch.tensor([12, 0, 4, 8, 3, 11, 2, 7, 6, 1, 10, 5, 9])
    with torch.no_grad():
        original = model(keys, values, positions, groups, valid)
        permuted = model(keys[:, :, permutation], values[:, :, permutation], positions[:, :, permutation],
                         groups[:, permutation], valid[:, permutation])
        swapped = model(keys.flip(0), values.flip(0), positions, groups, valid)
    torch.testing.assert_close(original, permuted, atol=2e-6, rtol=2e-6)
    assert not torch.allclose(original, swapped, atol=1e-5, rtol=1e-5)


def test_positions_and_named_group_bias_can_change_attention_semantics():
    model = reader().eval(); keys, values, positions, groups, valid = inputs()
    with torch.no_grad():
        original = model(keys, values, positions, groups, valid)
        moved = model(keys, values, positions.flip(-1), groups, valid)
        model.group_bias[:, 0] = 5.
        favored = model(keys, values, positions, groups, valid)
        renamed = model(keys, values, positions, (groups+1) % 9, valid)
    assert not torch.allclose(original, moved, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(favored, renamed, atol=1e-6, rtol=1e-6)


def test_group_count_prior_does_not_reward_duplicate_identical_tokens():
    model = reader().eval(); keys, values, positions, groups, valid = inputs(tokens=2)
    # Duplicating every token of one group leaves its attention partition mass
    # unchanged under the inverse valid-group-count prior.
    indices = torch.tensor([0, 0, 0, 1])
    with torch.no_grad():
        base = model(keys, values, positions, groups, valid)
        duplicated = model(keys[:, :, indices], values[:, :, indices], positions[:, :, indices],
                           groups[:, indices], valid[:, indices])
    torch.testing.assert_close(base, duplicated, atol=2e-6, rtol=2e-6)


def test_scales_are_registered_invertible_per_layer_and_restore_exactly():
    from slowfast_v11.readability import KVReader
    model = reader(); data = inputs()
    baseline = model(*data).detach()
    scales = torch.arange(1, 7).float()
    model.set_scales(scales, scales*2)
    assert 'key_scales' in dict(model.named_buffers()) and 'value_scales' in dict(model.named_buffers())
    assert torch.equal(model.key_scales, scales)
    scaled = list(data)
    scaled[0] = data[0]*scales[None, :, None, None]
    scaled[1] = data[1]*(scales*2)[None, :, None, None]
    torch.testing.assert_close(model(*scaled), baseline, atol=2e-6, rtol=2e-6)
    restored = KVReader(kv_width=8, hidden=16)
    restored.load_state_dict(model.state_dict())
    assert torch.equal(restored.key_scales, scales)
    assert torch.equal(restored(*scaled), model(*scaled))
    for invalid in [torch.zeros(6), torch.ones(5), torch.tensor([1., 1., 1., 1., 1., float('nan')])]:
        with pytest.raises(ValueError): model.set_scales(invalid, scales)
    assert torch.equal(model.key_scales, scales)


def test_invalid_layout_empty_valid_and_masked_content():
    model = reader(); keys, values, positions, groups, valid = inputs()
    with pytest.raises(ValueError): model(keys, values, positions, groups, valid.float())
    with pytest.raises(ValueError): model(keys, values, positions, groups+9, valid)
    with pytest.raises(ValueError): model(keys[:, :5], values[:, :5], positions, groups, valid)
    with pytest.raises(ValueError): model(keys, values, positions[:2], groups, valid)
    empty = valid.clone(); empty[1] = False
    with pytest.raises(ValueError): model(keys, values, positions, groups, empty)
    valid[:, -2:] = False
    changed_k, changed_v = keys.detach().clone(), values.detach().clone()
    changed_k[:, :, -2:] += 1000.; changed_v[:, :, -2:] -= 1000.
    with torch.no_grad():
        torch.testing.assert_close(model(keys, values, positions, groups, valid),
                                   model(changed_k, changed_v, positions, groups, valid), atol=0., rtol=0.)


def test_setting_scales_owns_values_even_when_arguments_alias_existing_buffers():
    model = reader()
    keys, values = torch.arange(1, 7).float(), torch.arange(7, 13).float()
    model.set_scales(keys, values)
    model.set_scales(model.value_scales, model.key_scales)
    assert torch.equal(model.key_scales, values)
    assert torch.equal(model.value_scales, keys)


def test_frozen_final_feature_reader_has_no_inherited_head():
    from slowfast_v11.readability import FinalFeatureReader
    torch.manual_seed(42)
    model = FinalFeatureReader(width=12, hidden=16)
    features = torch.randn(2, 8, 12, requires_grad=True)
    before = features.detach().clone()
    result = model(features)
    assert result.shape == (2, 8, 3) and torch.isfinite(result).all()
    result.square().mean().backward()
    assert features.grad is None and torch.equal(features, before)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    with pytest.raises(ValueError): model(features[:, :7])


def test_final_feature_reader_matches_original_normalization_topology():
    from slowfast_v11.readability import FinalFeatureReader
    model = FinalFeatureReader(width=12)
    assert [type(module) for module in model.head] == [
        torch.nn.LayerNorm, torch.nn.Linear, torch.nn.GELU, torch.nn.Linear]
    assert model.head[1].out_features == 256 and model.head[-1].out_features == 3
    features = torch.randn(2, 8, 12)
    # The original LayerNorm makes a common per-token feature offset irrelevant.
    torch.testing.assert_close(model(features), model(features+5.), atol=2e-6, rtol=2e-6)


def test_both_readers_fit_small_source_dependent_semantic_signal():
    from slowfast_v11.readability import KVReader, FinalFeatureReader
    torch.manual_seed(42)
    latent = torch.linspace(-1., 1., 12)
    target = torch.stack((latent, -.5*latent, .2*latent), -1)[:, None].expand(-1, 8, -1)
    keys = torch.randn(12, 6, 7, 8)
    values = latent[:, None, None, None].expand(12, 6, 7, 8).clone()
    positions = torch.arange(7)[None, None].expand(3, 12, -1)
    groups = (torch.arange(7) % 3)[None].expand(12, -1)
    valid = torch.ones(12, 7, dtype=torch.bool)
    # A nonuniform signal with a fixed independent channel pattern remains
    # identifiable after the original head's per-token LayerNorm.
    direction = torch.linspace(-1., 1., 12)
    features = (latent[:, None]*direction[None]+direction.square()[None])[:, None].expand(-1, 8, -1).clone()
    for model, data in [(KVReader(kv_width=8, hidden=12), (keys, values, positions, groups, valid)),
                        (FinalFeatureReader(width=12, hidden=16), (features,))]:
        opt = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=0.)
        initial = torch.nn.functional.l1_loss(model(*data), target).item()
        for _ in range(80):
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.l1_loss(model(*data), target)
            loss.backward(); opt.step()
        final = torch.nn.functional.l1_loss(model(*data), target).item()
        assert final < .06 and final < initial*.2
