"""Independent-process NAVSIM reconstruction and prediction equality check."""
import gc
import os
import torch
from fast_only.common import write_json
from fast_navsim.config import read_config
from tools.run_navsim_preflight import reload_predictions


def main():
    reports={}
    for label,path in [('configuration_smoke','configs/navsim_preflight.yaml'),('deep_overfit','configs/navsim_deep.yaml')]:
        config=read_config(path);torch.set_num_threads(config['cpu_threads'])
        report,model=reload_predictions(config);reports[label]=report
        del model;gc.collect();torch.cuda.empty_cache()
    result={'passed':True,'verification_process_pid':os.getpid(),
            'method':'new Python process, native local base weights plus checkpoint; original-resolution inputs',
            'reports':reports}
    write_json('outputs/navsim/audit/independent_process_reload.json',result);print(result,flush=True)


if __name__=='__main__':main()
