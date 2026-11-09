"""Extract a fixed real 2 Hz fragment from an authenticated saved local prefix.

No network APIs are imported. Original JPEG bytes are retained; annotations and
future poses never enter the fragment. This is a runtime smoke fixture only.
"""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import pickle
import shutil
import tarfile
import tempfile

from PIL import Image

from fast_navsim.data import _pose
from slowfast_v11.diagnostic_data import _command, _timestamp


LOG = '2021.05.12.22.28.35_veh-35_00620_01164'
PREFIX_SHA256 = 'cba41c71237474f19fd6651e8b97f34eae7356223653523543705e808834832b'
METADATA_SHA256 = 'f61642cd7258d446715e53727f8a7474a22ba282ba0b5d630b006688571db62a'
ARCHIVE_ROOT = 'openscene-v1.1/sensor_blobs/mini/'


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('Extraction requires safe relative paths')
    return path


def extract_selected(prefix, destination, requested):
    """Read only explicitly named regular members; never call tar.extract()."""
    destination = Path(destination).resolve()
    for member, relative in requested.items():
        _safe_relative(member); _safe_relative(relative)
    receipt = {}
    with tarfile.open(prefix, mode='r|gz') as archive:
        for member in archive:
            if member.name not in requested:
                continue
            if not member.isfile() or member.size <= 0 or member.size > 5_000_000:
                raise ValueError('Expected bounded regular JPEG member')
            relative = requested[member.name]
            path = destination / relative
            if not path.resolve().is_relative_to(destination):
                raise ValueError('Extraction requires safe relative destinations')
            if path.exists():
                raise FileExistsError(f'Preserve existing image: {path}')
            data = archive.extractfile(member).read()
            if len(data) != member.size:
                raise ValueError('Truncated selected JPEG')
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                if image.format != 'JPEG' or image.size != (1920, 1080):
                    raise ValueError('Smoke fragment requires original 1920x1080 JPEGs')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            receipt[relative] = dict(archive_member=member.name, bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(), size_wh=[1920, 1080], format='JPEG')
            if len(receipt) == len(requested):
                break  # Saved archive is intentionally a prefix, not a full tgz.
    if len(receipt) != len(requested):
        raise ValueError('Saved local prefix does not contain every selected JPEG')
    return receipt


def prepare_fragment(output, *,
        prefix='data/navsim_direct_full/camera_prefixes/camera_27_prefix.gz',
        prefix_receipt='outputs/slowfast_direct_full/data/camera_27_durable_prefix.json',
        metadata_inventory='outputs/slowfast_direct_full/data/metadata_retained.json'):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve prior fragment; choose a new output directory')
    prefix = Path(prefix).resolve()
    receipt = json.loads(Path(prefix_receipt).read_text())
    if receipt['prefix_bytes'] != prefix.stat().st_size or receipt['prefix_sha256'] != PREFIX_SHA256:
        raise ValueError('Local archive durable receipt identity differs')
    if digest_file(prefix) != PREFIX_SHA256:
        raise ValueError('Local camera prefix SHA256 differs')
    record = next(row for row in json.loads(Path(metadata_inventory).read_text())['files']
                  if Path(row['path']).stem == LOG)
    if record['sha256'] != METADATA_SHA256:
        raise ValueError('Metadata inventory identity differs')
    blob = Path(record['path']).read_bytes()
    if hashlib.sha256(blob).hexdigest() != METADATA_SHA256:
        raise ValueError('Local metadata SHA256 differs')
    frames = pickle.loads(blob)  # Only after exact retained provenance authentication.
    observations, requested = [], {}
    for index in range(247, 311):
        history = frames[index-3:index+1]
        if len(history) != 4 or any(f['log_name'] != LOG for f in history):
            raise ValueError('Selected fragment crosses logs or lacks history')
        times = [_timestamp(frame) for frame in history]
        if any(abs(b-a-500_000)>1000 for a, b in zip(times, times[1:])):
            raise ValueError('Selected fragment has a timestamp gap')
        command = _command(history[-1])
        if any(_command(frame) != command for frame in history):
            raise ValueError('Selected history crosses navigation commands')
        images = []
        for frame in history:
            relative = str(_safe_relative(frame['cams']['CAM_F0']['data_path']))
            if not relative.startswith(LOG + '/CAM_F0/'):
                raise ValueError('Unexpected selected camera path')
            member, target = ARCHIVE_ROOT + relative, 'images/' + relative
            requested[member] = target
            images.append(target)
        current = history[-1]
        ego = [float(v) for v in current['ego_dynamic_state']]
        if len(ego) != 4 or not all(math.isfinite(v) for v in ego):
            raise ValueError('Invalid current raw ego4')
        observations.append(dict(image_paths=images, ego=ego, command=command,
            ego_pose=list(_pose(current)), timestamp_us=times[-1], token=current['token'],
            episode_id=LOG, navversion=0, frame_index=index))
    if len(requested) != 67 or len({r['command'] for r in observations}) != 1:
        raise ValueError('Pinned continuous fragment identity differs')
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.async-fragment-', dir=output.parent))
    try:
        images = extract_selected(prefix, stage, requested)
        source = dict(schema='async_fragment_source_v1', new_download_bytes=0,
            upstream_split='mini', scope='continuous wall-clock runtime smoke; not NAVSIM train/test scoring',
            prefix_path=str(prefix), prefix_bytes=prefix.stat().st_size,
            prefix_sha256=PREFIX_SHA256, prefix_receipt=str(Path(prefix_receipt).resolve()),
            full_archive_verified=False, metadata_path=record['path'], metadata_sha256=METADATA_SHA256,
            episode_id=LOG, first_window_index=247, last_window_index=310,
            windows=64, source_images=67, source_frequency_hz=2,
            source_span_s=(observations[-1]['timestamp_us']-observations[0]['timestamp_us'])/1e6,
            annotations_included=False, original_jpeg_bytes_preserved=True, images=images)
        fragment = dict(schema='async_input_fragment_v1', source_receipt='source_receipt.json',
                        observations=observations)
        (stage/'fragment.json').write_text(json.dumps(fragment, indent=2)+'\n')
        (stage/'source_receipt.json').write_text(json.dumps(source, indent=2)+'\n')
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage)
        raise
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='outputs/slowfast_cluster/async_fragment')
    parser.add_argument('--prefix', default='data/navsim_direct_full/camera_prefixes/camera_27_prefix.gz')
    parser.add_argument('--prefix-receipt', default='outputs/slowfast_direct_full/data/camera_27_durable_prefix.json')
    parser.add_argument('--metadata-inventory', default='outputs/slowfast_direct_full/data/metadata_retained.json')
    args = parser.parse_args()
    result = prepare_fragment(args.output, prefix=args.prefix, prefix_receipt=args.prefix_receipt,
                              metadata_inventory=args.metadata_inventory)
    print(json.dumps({k: result[k] for k in ['windows', 'source_images', 'source_span_s', 'new_download_bytes']}))


if __name__ == '__main__':
    main()
