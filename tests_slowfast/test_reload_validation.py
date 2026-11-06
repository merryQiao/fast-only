"""Pure validation for independent-process held-out prediction comparison."""
import copy
import math

import pytest

from tools.reload_slowfast_validation import compare_predictions


def record(token='a', mode='none', source=None, age=0.):
    return dict(stage='after_consumer',split='val',token=token,mode=mode,
                source_token=source,age_s=age,prediction=[[float(i),.25,0.] for i in range(8)])


def test_exact_comparison_matches_every_condition_regardless_of_record_order():
    refs=[record(),record(mode='fresh',source='a'),record(mode='shuffled',source='b'),
          record(mode='stale',source='old',age=.5)]
    report=compare_predictions(refs,list(reversed(copy.deepcopy(refs))))
    assert report['passed'] is True
    assert report['count']==4 and report['max_abs_difference']==0.
    assert report['atol']==0.
    assert report['by_mode']=={m:{'count':1,'max_abs_difference':0.,'passed':True}
                              for m in ('none','fresh','shuffled','stale')}


def test_default_requires_exact_predictions_and_explicit_atol_is_absolute_only():
    ref=[record()];actual=copy.deepcopy(ref);actual[0]['prediction'][7][2]=.001
    assert compare_predictions(ref,actual)['passed'] is False
    assert compare_predictions(ref,actual,atol=.002)['passed'] is True
    assert compare_predictions(ref,actual,atol=.001)['max_abs_difference']==.001
    actual[0]['prediction'][7][0]=1000.
    assert compare_predictions(ref,actual,atol=.002)['passed'] is False


@pytest.mark.parametrize('atol', [-1., math.nan, math.inf, True, '0.1'])
def test_invalid_tolerance_is_rejected(atol):
    with pytest.raises(ValueError,match='tolerance'):compare_predictions([record()],[record()],atol=atol)


@pytest.mark.parametrize('side', ['reference','actual'])
@pytest.mark.parametrize('mutation', ['short','wide','nan','bool','duplicate','wrong_split','wrong_stage'])
def test_malformed_or_ambiguous_records_never_compare_successfully(side,mutation):
    refs=[record()];actual=[record()];records=refs if side=='reference' else actual
    if mutation=='short':records[0]['prediction'].pop()
    if mutation=='wide':records[0]['prediction'][0].append(0.)
    if mutation=='nan':records[0]['prediction'][0][0]=math.nan
    if mutation=='bool':records[0]['prediction'][0][0]=True
    if mutation=='duplicate':records.append(copy.deepcopy(records[0]))
    if mutation=='wrong_split':records[0]['split']='train'
    if mutation=='wrong_stage':records[0]['stage']='before_consumer'
    with pytest.raises(ValueError):compare_predictions(refs,actual)


@pytest.mark.parametrize('mutation', ['missing','extra','different_source','different_age','empty'])
def test_condition_identity_and_complete_coverage_are_required(mutation):
    refs=[record(),record(mode='fresh',source='a')];actual=copy.deepcopy(refs)
    if mutation=='missing':actual.pop()
    if mutation=='extra':actual.append(record('b'))
    if mutation=='different_source':actual[1]=record(mode='shuffled',source='b')
    if mutation=='different_age':actual[1]=record(mode='stale',source='a',age=.5)
    if mutation=='empty':refs=actual=[]
    with pytest.raises(ValueError):compare_predictions(refs,actual)


@pytest.mark.parametrize('item', [record(mode='bogus'),record(mode='fresh',source='b'),
    record(mode='none',source='a'),record(mode='shuffled',source='a'),
    record(mode='stale',source='old',age=0.),record(age=math.nan)])
def test_invalid_condition_semantics_are_rejected(item):
    with pytest.raises(ValueError):compare_predictions([item],[item])


@pytest.mark.parametrize('saved_identities', [None, {'val_manifest':'changed'}])
def test_reload_rejects_missing_or_changed_data_before_restoring_weights(tmp_path,monkeypatch,saved_identities):
    import sys
    from types import SimpleNamespace
    import torch
    from tools.reload_slowfast_validation import reload_and_compare

    checkpoint=dict(schema='slowfast_validation_v1',stage='fast_consumer',step=120,
        config={'stages':{'fast_consumer':120}},reference_predictions=[record()],
        fast={},slow={},feedback_producer={},feedback_producer_digest='digest',assets={})
    if saved_identities is not None:checkpoint['data_identities']=saved_identities
    source=tmp_path/'final.pt';torch.save(checkpoint,source)
    class FakeExperiment:
        def __init__(self,config):
            self.assets={};self.data_identities={'val_manifest':'original'}
            self.fast=self.slow=None
    def forbidden_restore(*args):
        raise AssertionError('Data identities must be verified before restoring weights')
    monkeypatch.setitem(sys.modules,'tools.validate_slowfast_v11',SimpleNamespace(Experiment=FakeExperiment))
    monkeypatch.setitem(sys.modules,'slowfast_v11.validation',SimpleNamespace(
        state_digest=lambda state:'digest',canonical_state=lambda model:{},restore_weights=forbidden_restore))
    with pytest.raises(ValueError,match='[Dd]ata'):
        reload_and_compare(source,tmp_path/'new.json')
    assert not (tmp_path/'new.json').exists()
