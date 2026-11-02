import copy
import pytest
from fast_navsim.config import read_config
from fast_navsim.train import execution_limit,verify_cohort,verify_build_policy,prepare_output


def test_no_preflight_request_can_exceed_its_configured_execution_cap():
    cfg=read_config('configs/navsim_preflight.yaml')
    assert execution_limit(cfg,None)==4 and execution_limit(cfg,2)==2
    with pytest.raises(ValueError):execution_limit(cfg,8000)
    with pytest.raises(ValueError):execution_limit(cfg,0)


def test_formal_training_refuses_bounded_mini_cohort():
    cfg=read_config('configs/navsim.yaml')
    with pytest.raises(ValueError,match='bounded'):execution_limit(cfg,None)


def test_cohort_validation_uses_log_and_tokens_without_any_test_selection():
    a=[{'split':'train','route':'log-a','token':'a'}]
    b=[{'split':'val','route':'log-b','token':'b'}]
    assert verify_cohort(a,b) is None
    with pytest.raises(ValueError):verify_cohort(a,[dict(b[0],route='log-a')])
    with pytest.raises(ValueError):verify_cohort(a,[dict(b[0],token='a')])
    with pytest.raises(ValueError):verify_cohort(a,[dict(b[0],split='test')])


def test_data_root_and_split_seed_are_consumed_from_configuration(tmp_path):
    import json
    cfg=read_config('configs/navsim_preflight.yaml')
    cfg['data']['root']=str(tmp_path/'raw')
    cfg['data']['train_manifest']=str(tmp_path/'train.jsonl')
    report={'split_seed':42,'data_root':str(tmp_path/'raw')}
    (tmp_path/'build_report.json').write_text(json.dumps(report))
    rows=[{'images':[str(tmp_path/'raw/a.jpg')]*4}]
    verify_build_policy(cfg,rows)
    cfg['split_seed']=13
    with pytest.raises(ValueError,match='seed'):verify_build_policy(cfg,rows)


def test_failed_p0_evidence_without_checkpoint_cannot_be_overwritten(tmp_path):
    (tmp_path/'config.json').write_text('previous failed probe')
    with pytest.raises(FileExistsError):prepare_output(tmp_path,resume=False)
    assert (tmp_path/'config.json').read_text()=='previous failed probe'
