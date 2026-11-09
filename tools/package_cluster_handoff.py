"""Package committed source, authenticated frozen task state, and tiny fixtures.

The two pretrained base model directories and full dataset stay external.
Run only after committing source; assets are copied without modifying originals.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def package(output, include_fixtures=True):
    root = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip())
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True)
    if dirty.strip():
        raise ValueError('Commit all source first; handoff must identify the actual committed implementation')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve prior handoff; choose a new output')
    frozen = root/'outputs/slowfast_cluster/source'
    receipt = json.loads((frozen/'source_bundle.json').read_text())
    if sha256(frozen/'source_bundle.pt') != receipt['bundle_sha256']:
        raise ValueError('Frozen source file differs from its receipt')
    files = {str(p.relative_to(root)): p for p in frozen.glob('source_bundle.*') if p.suffix in ('.pt', '.json')}
    if include_fixtures:
        bundle = root/'outputs/slowfast_cluster/smoke_data'
        for path in bundle.iterdir():
            if path.is_file():
                files[str(path.relative_to(root))] = path
        sensor_root = (root/'data').resolve()
        for split in ('train', 'val'):
            for line in (bundle/f'{split}.jsonl').read_text().splitlines():
                for relative in json.loads(line)['images']:
                    candidate = (sensor_root/relative).resolve()
                    if not candidate.is_relative_to(sensor_root) or Path(relative).is_absolute():
                        raise ValueError('Fixture image escapes data root')
                    files['data/'+relative] = candidate
        fragment = root/'outputs/slowfast_cluster/async_fragment'
        for path in fragment.rglob('*'):
            if path.is_file():
                files[str(path.relative_to(root))] = path
    inventory = {name: dict(sha256=sha256(path), bytes=path.stat().st_size)
                 for name, path in sorted(files.items())}
    manifest = dict(schema='navsim_cluster_handoff_v1', source_commit=commit,
        source_prefix='source/', files=inventory, includes_fixtures=include_fixtures,
        external=['Qwen3.5-0.8B base assets', 'Qwen3.5-2B base assets', 'full NAVSIM V1 data'],
        instructions='source/docs/NAVSIM_V1_CLUSTER_HANDOFF.md')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(output.name+'.tmp')
    try:
        with tempfile.TemporaryDirectory() as scratch:
            source_archive = Path(scratch)/'source.tar'
            subprocess.run(['git', 'archive', '--format=tar', '--output', str(source_archive), commit],
                           cwd=root, check=True)
            with tarfile.open(temporary_output, 'w:gz', compresslevel=3) as target:
                with tarfile.open(source_archive, 'r:') as source:
                    for member in source:
                        payload = source.extractfile(member) if member.isfile() else None
                        member.name = 'source/'+member.name
                        target.addfile(member, payload)
                for name, path in sorted(files.items()):
                    target.add(path, arcname='source/'+name, recursive=False)
                content = (json.dumps(manifest, indent=2, allow_nan=False)+'\n').encode()
                item = tarfile.TarInfo('HANDOFF.json')
                item.size = len(content)
                target.addfile(item, io.BytesIO(content))
        temporary_output.replace(output)
    finally:
        temporary_output.unlink(missing_ok=True)
    manifest.update(archive_sha256=sha256(output), archive_bytes=output.stat().st_size)
    output.with_name(output.name+'.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k: manifest[k] for k in ('source_commit', 'archive_sha256', 'archive_bytes')}))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--without-fixtures', action='store_true')
    args = parser.parse_args()
    package(args.output, not args.without_fixtures)


if __name__ == '__main__':
    main()
