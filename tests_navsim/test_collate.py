import torch
import pytest
from PIL import Image
from fast_navsim.collate import NavsimCollator, temporal_content


def sample():
    return {'images':[Image.new('RGB',(65,35),(70,90,120)) for _ in range(4)],
            'ego':torch.ones(4),'command':torch.tensor(1),'trajectory_gt':torch.zeros(8,3)}


def test_temporal_prompt_uses_real_half_second_history():
    contents=temporal_content(sample()['images'])
    text=' '.join(x['text'] for x in contents if x['type']=='text')
    for value in ['-1.5s','-1.0s','-0.5s','+0.0s']:assert value in text
    assert 'eight' in text and 'heading' in text
    with pytest.raises(ValueError):temporal_content([Image.new('RGB',(32,32))])


def test_native_processor_original_pixels_and_future_labels_do_not_change_input():
    collator=NavsimCollator('models/Qwen3.5-0.8B',brightness=.1)
    first=sample();second=sample();second['trajectory_gt'].fill_(123)
    a,b=collator([first],seed=12),collator([second],seed=12)
    assert a['native']['image_grid_thw'].tolist()==[[1,4,6]]*4
    for key in a['native']:
        assert torch.equal(a['native'][key],b['native'][key]),key
    for key in ['ego','command']:assert torch.equal(a[key],b[key])
    assert not torch.equal(a['trajectory_gt'],b['trajectory_gt'])
    assert a['nav'] is None
    assert a['image_audit'][0]['resized'] is False
    assert a['image_audit'][0]['source_size_wh']==[65,35]
