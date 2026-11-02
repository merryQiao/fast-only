"""Read-only environment and local-weight audit, without downloading anything."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import subprocess
import sys


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='models/Qwen3.5-0.8B')
    parser.add_argument('--output', default='outputs/fast_only/audit/environment.json')
    args = parser.parse_args()
    packages = {}
    for package in ['torch', 'transformers', 'peft', 'accelerate', 'flash-linear-attention',
                    'causal-conv1d', 'triton', 'tilelang', 'apache-tvm-ffi',
                    'numpy', 'Pillow', 'matplotlib', 'pytest']:
        try: packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: packages[package] = None
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv'],
                         capture_output=True, text=True, check=True).stdout
    model = Path(args.model)
    files = {str(p.name): {'bytes': p.stat().st_size, 'sha256': sha256(p)}
             for p in sorted(model.iterdir()) if p.is_file()}
    report = {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
              'packages': packages, 'gpu': gpu, 'model_path': str(model.resolve()), 'model_files': files,
              'runtime_environment': {key: os.environ.get(key) for key in
                                      ['CUDA_HOME', 'FLA_TILELANG', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'TOKENIZERS_PARALLELISM']},
              'scope': 'Few-window smoke only; no full Bench2Drive training set or prior experiment artifacts changed.',
              'existing_local_model': str((Path.cwd().parent/'BARD-VL-2-pingce/model/Qwen3.5-0.8B').resolve()),
              'environment_changes': 'Project venv uses existing qwen35-t3d torch/transformers; PEFT, accelerate, pytest and TileLang with its dependencies installed locally.'}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
