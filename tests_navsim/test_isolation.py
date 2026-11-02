import copy
import json
import torch
from PIL import Image
from fast_navsim.data import build_window,compute_ego_stats,NavsimDataset
from fast_navsim.collate import NavsimCollator
from tests_navsim.test_data import frames


def test_actual_native_inputs_are_byte_identical_after_replacing_future_raw_frames(tmp_path):
    raw=frames(n=12)
    for i in range(12):
        path=tmp_path/f'log-a/{i}.jpg';path.parent.mkdir(exist_ok=True)
        Image.new('RGB',(65,35),(30+i,90,120)).save(path)
    original=build_window(raw,3,'train',tmp_path)
    stats=compute_ego_stats([original])
    altered=copy.deepcopy(raw)
    for i in range(4,12):
        altered[i]['ego2global_translation']=[1000-i,200+i,0]
        altered[i]['ego_dynamic_state']=[999]*4
        altered[i]['driving_command']=[1,0,0,0]
        Image.new('RGB',(65,35),(240,10,15)).save(tmp_path/f'log-a/{i}.jpg')
    changed=build_window(altered,3,'train',tmp_path)
    def dataset(row,name):
        path=tmp_path/(name+'.jsonl');path.write_text(json.dumps(row)+'\n')
        return NavsimDataset(path,stats)
    collate=NavsimCollator('models/Qwen3.5-0.8B',brightness=.1)
    a=collate([dataset(original,'before')[0]],seed=42)
    b=collate([dataset(changed,'after')[0]],seed=42)
    for key in a['native']:assert torch.equal(a['native'][key],b['native'][key]),key
    for key in ['ego','command']:assert torch.equal(a[key],b[key])
    assert a['nav'] is None and b['nav'] is None
    assert not torch.equal(a['trajectory_gt'],b['trajectory_gt'])
