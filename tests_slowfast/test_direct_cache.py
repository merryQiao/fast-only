"""Frozen-cache authentication must reject changed inputs and role leakage."""
import pytest
import torch

from tools.cache_direct_full import validate_rows, validate_packet


def test_train_and_val_roles_and_recording_groups_are_independent():
    train=[dict(token='a',split='train',cohort='train',route='log_a_000_100')]
    val=[dict(token='b',split='val',cohort='new_val',route='log_b_000_100')]
    validate_rows(train,val,counts=(1,1),groups=(1,1))
    with pytest.raises(ValueError):
        validate_rows(train,[dict(val[0],route='log_a_101_200')],counts=(1,1),groups=(1,1))
    with pytest.raises(ValueError):
        validate_rows(train,[dict(val[0],token='a')],counts=(1,1),groups=(1,1))
    with pytest.raises(ValueError):
        validate_rows(train,[dict(val[0],split='train')],counts=(1,1),groups=(1,1))


def test_packets_cannot_cache_language_embeddings_or_future_targets():
    packet=dict(token='a',batch=dict(native=dict(pixel_values=torch.empty(0)),ego=torch.ones(1,4),command=torch.ones(1,dtype=torch.long)),vision=(torch.ones(2,4),))
    validate_packet(packet)
    packet['batch']['native']['inputs_embeds']=torch.ones(2,4)
    with pytest.raises(ValueError):validate_packet(packet)
    del packet['batch']['native']['inputs_embeds']
    packet['batch']['trajectory_gt']=torch.ones(8,3)
    with pytest.raises(ValueError):validate_packet(packet)
