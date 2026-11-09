"""Export original input-only runtime rows from an authenticated prepared bundle.

No sensor image bytes are opened, copied, or downloaded. The output references
the supplied sensor root through a directory symlink so existing runtime loaders
can keep using safe relative paths. Rerun this exporter after relocating data.
"""
import argparse
import json
import math
from pathlib import Path
import shutil
import tempfile

from fast_cluster.data import PreparedData


INTERVAL_US = 500_000
TOLERANCE_US = 1000


def _project(row, manifest_index):
    """Read current/history fields explicitly; never inspect targets or labels."""
    images = row.get('images')
    if not isinstance(images, list) or len(images) != 4 or len(set(images)) != 4:
        raise ValueError('runtime export requires four distinct chronological image paths')
    for name in images:
        if not isinstance(name, str) or not name:
            raise ValueError('invalid source image path')
        path = Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('image_path_escape')
    history = row.get('history_timestamps_us')
    if (not isinstance(history, list) or len(history) != 4
            or any(type(value) is not int for value in history)
            or type(row.get('timestamp_us')) is not int or history[-1] != row['timestamp_us']
            or any(abs(b-a-INTERVAL_US)>TOLERANCE_US for a, b in zip(history, history[1:]))):
        raise ValueError('invalid causal source history timestamps')
    if type(row.get('frame')) is not int or row['frame'] < 3:
        raise ValueError('invalid current source frame index')
    for key, count in [('ego', 4), ('ego_pose', 3)]:
        values = row.get(key)
        if (not isinstance(values, list) or len(values) != count
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in values)):
            raise ValueError(f'invalid current source {key}')
    if type(row.get('command')) is not int or row['command'] not in range(4):
        raise ValueError('invalid current navigation command')
    if any(not isinstance(row.get(key), str) or not row[key] for key in ['token', 'route']):
        raise ValueError('invalid source identity')
    return dict(token=row['token'], route=row['route'], frame=row['frame'],
        timestamp_us=row['timestamp_us'], history=list(history), images=list(images),
        ego=list(row['ego']), ego_pose=list(row['ego_pose']), command=row['command'],
        manifest_index=manifest_index)


def _consecutive(previous, current):
    return (previous['route'] == current['route'] and current['frame'] == previous['frame']+1
        and abs(current['timestamp_us']-previous['timestamp_us']-INTERVAL_US) <= TOLERANCE_US
        and previous['history'][1:] == current['history'][:3]
        and previous['images'][1:] == current['images'][:3])


def _select(dataset, windows, episode_id):
    # Store only a compact sort index for the complete prepared manifest. Full
    # annotated rows are parsed lazily and immediately projected to causal data.
    index = []
    for number in range(len(dataset)):
        row = dataset.row(number)
        route, timestamp, token = row.get('route'), row.get('timestamp_us'), row.get('token')
        if not isinstance(route, str) or type(timestamp) is not int or not isinstance(token, str):
            raise ValueError('invalid source manifest identity/time')
        if episode_id is None or route == episode_id:
            index.append((route, timestamp, token, number))
    run, longest = [], 0
    for _, _, _, number in sorted(index):
        current = _project(dataset.row(number), number)
        if run and not _consecutive(run[-1], current):
            run = []
        run.append(current)
        longest = max(longest, len(run))
        if len(run) == windows:
            return run
    raise ValueError(f'no_contiguous_sequence: requested={windows}, longest={longest}; '
        'scene/eligibility filters may remove consecutive anchors; gaps are never filled or interpolated')


def export_fragment(bundle, sensor_root, split, windows, output, *, episode_id=None, allow_smoke=False):
    """Select the first (log,time,token) contiguous sequence of the requested size."""
    if split not in ('train', 'val'):
        raise ValueError('Runtime fragment split must be train or val')
    if type(windows) is not int or windows < 2:
        raise ValueError('Runtime fragment requires at least two consecutive windows')
    if episode_id is not None and (not isinstance(episode_id, str) or not episode_id):
        raise ValueError('episode_id must name a nonempty source log')
    output, sensor_root = Path(output).resolve(), Path(sensor_root).resolve()
    if output.exists():
        raise FileExistsError('Preserve existing fragment; select a new output directory')
    if not sensor_root.is_dir():
        raise ValueError('sensor_root must be an existing original sensor directory')
    prepared = PreparedData(bundle, sensor_root, allow_smoke=allow_smoke)
    dataset = prepared.dataset(split, image_frames=4)
    selected = _select(dataset, windows, episode_id)
    original_paths = sorted({name for row in selected for name in row['images']})
    resolved = {}
    for name in original_paths:
        path = (sensor_root/name).resolve()
        if not path.is_relative_to(sensor_root):
            raise ValueError('image_path_escape')
        if not path.is_file():
            raise ValueError(f'missing original source image: {name}')
        resolved[name] = str(path)
    # Authenticate path membership through the already SHA-verified inventory;
    # this does not reopen or rehash any sensor image itself.
    needed = set(original_paths)
    with (prepared.bundle_dir/'images.jsonl').open() as stream:
        for line in stream:
            if line.strip():
                needed.discard(json.loads(line)['path'])
            if not needed:
                break
    if needed:
        raise ValueError('selected image path is missing from authenticated image inventory')
    observations, navversion, previous_command = [], -1, None
    for row in selected:
        if row['command'] != previous_command:
            navversion += 1
            previous_command = row['command']
        observations.append(dict(image_paths=['sensors/'+name for name in row['images']],
            ego=row['ego'], command=row['command'], ego_pose=row['ego_pose'],
            timestamp_us=row['timestamp_us'], token=row['token'], episode_id=row['route'],
            navversion=navversion, frame_index=row['frame']))
    manifest = prepared.identity['manifests'][split]
    receipt = dict(schema='prepared_runtime_fragment_source_v1', runtime_only=True,
        quality_evaluated=False, source_kind=prepared.identity['kind'],
        official_source_coverage=prepared.identity['kind']=='navsim_v1_full',
        bundle=str(prepared.bundle_dir), bundle_identity_sha256=prepared.identity['identity_sha256'],
        navsim_revision=prepared.identity['navsim_revision'],
        manifest=dict(manifest, path=str(dataset.manifest)), split=split,
        sensor_root=str(sensor_root), image_references='original files via output/sensors directory symlink',
        sensor_bytes_read=0, images_copied=0, downloads=0, interpolation=False,
        sensor_validation='path existence/containment and authenticated inventory membership; image bytes not rechecked',
        selection='first sequence sorted by log, current timestamp, token',
        continuity='same log; adjacent frame index; 500000 +/- 1000 us; exact overlapping image/history timestamp triples',
        episode_id=selected[0]['route'], windows=len(selected), source_frequency_hz=2,
        source_span_s=(selected[-1]['timestamp_us']-selected[0]['timestamp_us'])/1e6,
        manifest_indices=[row['manifest_index'] for row in selected],
        selected_tokens=[row['token'] for row in selected], original_resolved_image_paths=resolved,
        source_rows=[dict(token=row['token'], original_image_paths=row['images'],
                          history_timestamps_us=row['history']) for row in selected])
    fragment = dict(schema='async_input_fragment_v1', source_receipt='source_receipt.json', observations=observations)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.runtime-fragment-', dir=output.parent))
    try:
        (stage/'sensors').symlink_to(sensor_root, target_is_directory=True)
        (stage/'fragment.json').write_text(json.dumps(fragment, indent=2, allow_nan=False)+'\n')
        (stage/'source_receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage)
        raise
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--sensor-root', required=True, type=Path)
    parser.add_argument('--split', required=True, choices=['train', 'val'])
    parser.add_argument('--windows', type=int, default=64)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--episode-id', help='Restrict selection to this exact source log')
    parser.add_argument('--allow-smoke', action='store_true', help='Explicitly permit an authenticated smoke bundle')
    args = parser.parse_args(argv)
    receipt = export_fragment(args.bundle, args.sensor_root, args.split, args.windows,
        args.output, episode_id=args.episode_id, allow_smoke=args.allow_smoke)
    print(json.dumps({key: receipt[key] for key in ['episode_id', 'windows', 'source_span_s',
        'bundle_identity_sha256', 'runtime_only', 'quality_evaluated']}))
    return receipt


if __name__ == '__main__':
    main()
