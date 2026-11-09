from collections import Counter
import importlib

import pytest


def api():
    return importlib.import_module('tools.diagnose_cross_inputs')


def test_joint_factorial_cells_have_explicit_same_and_different_donors():
    cells=api().input_cells('full_correct_kv')
    assert len(cells)==len(set((i,k) for i,k,c in cells))==16
    assert Counter(c for i,k,c in cells)=={'A':1,'B':3,'C':3,'D_same':3,'D_independent':6}
    assert all((i==k and i!=0) for i,k,c in cells if c=='D_same')
    assert all(i!=0 and k!=0 and i!=k for i,k,c in cells if c=='D_independent')


def test_fast_only_uses_one_original_and_three_image_sources_without_kv():
    cells=api().input_cells('full_fast_only')
    assert cells==[(0,None,'original'),(1,None,'swapped'),(4,None,'swapped'),(7,None,'swapped')]
    with pytest.raises(ValueError):api().input_cells('reader')


def coverage_fixture(arm='full_correct_kv'):
    tokens=['a','b']
    plan=dict(tokens=tokens,donors={str(i):{'a':f'a-{i}','b':f'b-{i}'} for i in [1,4,7]})
    rows=[]
    for token in tokens:
        for image,kv,cell in api().input_cells(arm):
            rows.append(dict(token=token,arm=arm,image_offset=image,kv_offset=kv,cell=cell,
                image_source_token=token if image==0 else plan['donors'][str(image)][token],
                kv_source_token=None if kv is None else token if kv==0 else plan['donors'][str(kv)][token]))
    return plan,rows


def test_coverage_rejects_missing_duplicate_or_mislabeled_factorial_records():
    plan,rows=coverage_fixture()
    assert api().validate_coverage(rows,plan,'full_correct_kv')==32
    for broken in [rows[:-1],rows+[rows[0]],rows[:1]+[dict(rows[1],cell='A')]+rows[2:],
                   [dict(rows[0],image_source_token='wrong')]+rows[1:]]:
        with pytest.raises(ValueError):api().validate_coverage(broken,plan,'full_correct_kv')


def test_cross_coverage_keeps_null_kv_and_exact_target_token_set():
    plan,rows=coverage_fixture('full_fast_only')
    assert api().validate_coverage(rows,plan,'full_fast_only')==8
    rows[0]['kv_source_token']='a'
    with pytest.raises(ValueError):api().validate_coverage(rows,plan,'full_fast_only')
