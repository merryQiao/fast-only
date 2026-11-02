import copy
import json
import pytest
import torch
from fast_navsim.runtime import dataset_fingerprints,save_checkpoint,restore_checkpoint,verify_checkpoint


class Toy(torch.nn.Module):
    def __init__(self):super().__init__();self.linear=torch.nn.Linear(2,2)
    def trainable_state(self):return {n:p.detach().cpu().clone() for n,p in self.named_parameters()}
    def load_trainable_state(self,state):self.load_state_dict(state)


def fixture_config(tmp_path):
    paths={}
    for key in ['train_manifest','val_manifest','stats']:
        p=tmp_path/(key+'.json');p.write_text('{}');paths[key]=str(p)
    audit=tmp_path/'audit.json';audit.write_text(json.dumps({'sources':[]}));paths['source_audit']=str(audit)
    model=tmp_path/'model';model.mkdir();(model/'weights.safetensors').write_bytes(b'weights')
    # Manifest paths are empty JSON here to exercise asset fingerprints alone.
    return {'data':paths,'model':{'path':str(model)},'train_seed':42}


def test_checkpoint_saves_actual_rng_sampler_optimizer_and_rejects_drift(tmp_path):
    cfg=fixture_config(tmp_path);identity=dataset_fingerprints(cfg,include_images=False)
    model=Toy();opt=torch.optim.AdamW(model.parameters(),lr=.01)
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda n:1/(n+1))
    loss=model.linear(torch.ones(1,2)).sum();loss.backward();opt.step();scheduler.step()
    cursor={'next_batch':8,'seed':42};path=tmp_path/'state.pt'
    save_checkpoint(path,model,opt,scheduler,cfg,step=1,sampler_state=cursor,identity=identity,best_metric=3.)
    expected=torch.rand(3)
    old=copy.deepcopy(model.state_dict())
    for p in model.parameters():p.data.zero_()
    state=restore_checkpoint(path,model,opt,scheduler)
    assert state['sampler_state']==cursor and state['step']==1
    assert torch.equal(torch.rand(3),expected)
    assert all(torch.equal(model.state_dict()[k],v) for k,v in old.items())
    verify_checkpoint(state,include_images=False)
    altered=copy.deepcopy(state);altered['versions']['torch']='different'
    with pytest.raises(ValueError,match='versions'):verify_checkpoint(altered,include_images=False)
    (tmp_path/'stats.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='fingerprint'):verify_checkpoint(state,include_images=False)
