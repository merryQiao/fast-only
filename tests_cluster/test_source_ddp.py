"""Actual two-rank source dropout/no_sync regression via a stable spawn module."""
from tools.smoke_source_ddp import run_smoke


def test_dynamic_source_dropout_no_sync_tail_matches_global_mean(tmp_path):
    result=run_smoke(tmp_path)
    assert result['source_on_off_no_sync_matches_global_reference']
    assert result['all_source_off_update_matches_reference']
