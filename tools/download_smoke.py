"""Download a bounded, official Bench2Drive v0.0.4 front-camera subset.

The compressed route is streamed (never saved); only requested annotations and
front JPEGs are retained. The network budget includes metadata/source requests.
No automatic retries can silently multiply transfer volume.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
from urllib.parse import quote

import requests

REPO = 'rethinklab/Bench2Drive-V0.0.4'
SOURCE_URL = 'https://raw.githubusercontent.com/Thinklab-SJTU/Bench2Drive/0.0.4/tools/data_collect.py'


class LimitedReader:
    def __init__(self, stream, byte_budget):
        self.stream = stream
        self.byte_budget = int(byte_budget)
        self.bytes_read = 0
        self._hasher = hashlib.sha256()

    def read(self, size=-1):
        remaining = self.byte_budget - self.bytes_read
        if remaining <= 0:
            raise RuntimeError('Network byte budget exhausted')
        data = self.stream.read(min(size if size >= 0 else remaining, remaining))
        self.bytes_read += len(data)
        self._hasher.update(data)
        return data


    @property
    def sha256(self):
        return self._hasher.hexdigest()


def extract_subset(stream, output_root, start_frame=0, num_frames=70):
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    extracted, retained = [], 0
    with tarfile.open(fileobj=stream, mode='r|gz') as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError(f'Unsafe archive path: {member.name}')
            match = re.search(r'(?:^|/)(anno/(\d+)\.json(?:\.gz)?|camera/rgb_front/(\d+)\.(?:jpg|jpeg|png))$', str(path))
            if not member.isfile() or match is None:
                continue
            frame = int(match.group(2) or match.group(3))
            if not start_frame <= frame < start_frame + num_frames:
                continue
            if member.size > 32 * 1024 * 1024:
                raise ValueError('Unexpectedly large selected archive member')
            target = root.joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            content = source.read()
            target.write_bytes(content)
            retained += len(content)
            extracted.append(str(target.relative_to(root)))
    return {'extracted_files': len(extracted), 'retained_bytes': retained, 'files': extracted}


def fetch_small(url, remaining_budget):
    with requests.get(url, stream=True, timeout=(20, 120), headers={'Accept-Encoding': 'identity'}) as response:
        response.raise_for_status()
        reader = LimitedReader(response.raw, remaining_budget)
        parts = []
        while True:
            chunk = reader.read(64 * 1024)
            if not chunk:
                break
            parts.append(chunk)
        return b''.join(parts), reader.bytes_read, response.headers.get('Link', '')


def download(output_root, num_frames=70, start_frame=0, byte_budget=256*1024*1024, route=None, repo=REPO, archive=None):
    if route and archive:
        raise ValueError('Use --route or --archive, not both')
    route = archive or route
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'download_provenance.json').exists():
        raise FileExistsError('Use a new output-root to preserve existing provenance')
    total = 0
    entries = []
    # Resolve the mutable dataset branch once. Both the listing and archive
    # request then use the same immutable revision, even if main advances.
    info_url = f'https://huggingface.co/api/datasets/{repo}'
    info, size, _ = fetch_small(info_url, byte_budget-total)
    total += size
    revision = json.loads(info).get('sha')
    if not isinstance(revision, str) or re.fullmatch(r'[0-9a-fA-F]{40}', revision) is None:
        raise ValueError('Dataset API did not return a valid pinned commit revision')
    next_url = f'https://huggingface.co/api/datasets/{repo}/tree/{revision}?recursive=true&expand=false&limit=1000'
    while next_url:
        body, size, link = fetch_small(next_url, byte_budget-total)
        total += size
        entries.extend(json.loads(body))
        next_match = re.search(r'<([^>]+)>; rel="next"', link)
        next_url = next_match.group(1) if next_match else None
    archives = sorted((x for x in entries if x['path'].endswith('.tar.gz')), key=lambda x: (x['size'], x['path']))
    selected = next((x for x in archives if x['path'] == route), None) if route else archives[0]
    if selected is None:
        raise ValueError(f'Unknown route archive: {route}')
    if selected['size'] + total > byte_budget:
        raise RuntimeError('Selected route exceeds remaining network byte budget')
    source, size, _ = fetch_small(SOURCE_URL, byte_budget-total)
    total += size
    (root / 'official_data_collect.py.txt').write_bytes(source)
    url = f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{quote(selected["path"])}'
    reader = None
    try:
        with requests.get(url, stream=True, timeout=(20, 120), headers={'Accept-Encoding': 'identity'}) as response:
            response.raise_for_status()
            reader = LimitedReader(response.raw, byte_budget-total)
            result = extract_subset(reader, root, start_frame, num_frames)
            # Consume gzip footer/trailing bytes too, so this is the complete
            # advertised archive hash, not a prefix hash after tar end markers.
            while reader.bytes_read < selected['size']:
                if not reader.read(min(1024 * 1024, selected['size'] - reader.bytes_read)):
                    break
            if reader.bytes_read != selected['size']:
                raise ValueError('Archive transfer length differs from pinned listing')
            expected_hash = selected.get('lfs', {}).get('oid')
            if expected_hash and reader.sha256 != expected_hash:
                raise ValueError('Archive SHA256 differs from pinned dataset LFS hash')
            total += reader.bytes_read
    except Exception:
        (root / 'download_failure.json').write_text(json.dumps({'network_bytes': total + (reader.bytes_read if reader else 0), 'byte_budget': byte_budget, 'url': url}, indent=2))
        raise
    report = dict(result, repo=repo, url=url, archive=selected['path'], archive_size=selected['size'],
                  dataset_revision=revision, dataset_revision_pinned_at_download=True,
                  dataset_revision_source=info_url,
                  network_bytes=total, archive_network_bytes=reader.bytes_read, byte_budget=byte_budget,
                  archive_sha256=reader.sha256, archive_hash_scope='complete_advertised_compressed_archive',
                  archive_expected_sha256=selected.get('lfs', {}).get('oid'),
                  archive_hash_verified_against_listing=bool(selected.get('lfs', {}).get('oid')),
                  file_sha256={p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in result['files']},
                  listed_archives=len(archives), start_frame=start_frame, num_frames=num_frames,
                  source_url=SOURCE_URL, source_sha256=hashlib.sha256(source).hexdigest(),
                  annotation_timing='no timestamp in official save(); frame_rate=10.0 is declared; fallback requires explicit builder opt-in',
                  full_training_set_downloaded=False)
    (root / 'download_provenance.json').write_text(json.dumps(report, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root', default='data/smoke')
    p.add_argument('--num-frames', type=int, default=70)
    p.add_argument('--start-frame', type=int, default=0)
    p.add_argument('--byte-budget', type=int, default=256*1024*1024)
    p.add_argument('--route')
    p.add_argument('--archive', help='Exact pinned dataset archive path; alias for --route')
    args = p.parse_args()
    if args.num_frames < 34 or args.start_frame < 0:
        p.error('Need at least 34 consecutive frames and nonnegative start')
    report = download(**vars(args))
    print(json.dumps({k: v for k, v in report.items() if k != 'files'}, indent=2))


if __name__ == '__main__':
    main()
