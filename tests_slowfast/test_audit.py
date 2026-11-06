from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import pytest
import torch
import yaml
from tools.slowfast_v11_audit import validate_config


def config():
    return yaml.safe_load((Path(__file__).parents[1]/'configs/slowfast_v11_p0.yaml').read_text())


def test_audit_is_bounded_and_same_device_requires_explicit_override():
    c=config();validate_config(c)
    c['slow']['device']='cuda:0'
    with pytest.raises(ValueError,match='single'):validate_config(c)
    c['audit']['allow_single_gpu']=True;validate_config(c)
    for section,key,value in [('audit','samples',2),('audit','optimizer_updates',8000),
                              ('data','frequency_hz',20),('interface','planning_queries',6),
                              ('data','camera','CAM_B0'),('interface','metadata_units','normalized'),
                              ('data','future_times_s',[1,2,3,4])]:
        changed=deepcopy(c);changed[section][key]=value
        with pytest.raises(ValueError):validate_config(changed)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Real CUDA startup integration check')
def test_fresh_process_initializes_cuda_before_memory_audit():
    code='''
import torch
from tools.slowfast_v11_audit import prepare_memory_measurement
assert not torch.cuda.is_initialized()
prepare_memory_measurement(['cuda:0'])
assert torch.cuda.is_initialized()
assert torch.cuda.max_memory_allocated('cuda:0') == 0
'''
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=240)
    assert result.returncode==0,result.stdout+result.stderr
