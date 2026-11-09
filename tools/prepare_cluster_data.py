"""Prepare full official NAVSIM V1 data or an explicit no-download mini smoke.

python -m tools.prepare_cluster_data smoke --sensor-root data \
    --output outputs/slowfast_cluster/smoke_data
python -m tools.prepare_cluster_data official --metadata-root /data/trainval_logs \
    --sensor-root /data/trainval_sensor_blobs/trainval --output /data/prepared_v1
"""
import argparse
import json
from pathlib import Path

from fast_cluster.data import prepare_official_data, prepare_smoke_data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    official = commands.add_parser("official", help="local full official TRAIN/VAL; no downloads")
    official.add_argument("--metadata-root", required=True, type=Path)
    official.add_argument("--sensor-root", required=True, type=Path)
    official.add_argument("--output", required=True, type=Path)
    official.add_argument("--devkit-root", type=Path, help="optional local official V1.1 devkit, checked against vendored hashes")
    official.add_argument("--max-rejection-fraction", type=float, default=0.0,
        help="explicit project-validity rejection allowance; default zero, missing official coverage always fails")
    official.add_argument("--source-provenance", type=Path,
        default=Path(__file__).resolve().parents[1] / "fast_cluster/source_provenance.json",
        help="frozen-source training registry; exclude its recording groups from selection VAL, preserve all official VAL for reporting (default: fast_cluster/source_provenance.json)")
    smoke = commands.add_parser("smoke", help="exactly 8 TRAIN / 4 VAL inherited mini rows")
    smoke.add_argument("--train-manifest", type=Path, default=Path("outputs/slowfast_direct_full/data/train.jsonl"))
    smoke.add_argument("--val-manifest", type=Path, default=Path("outputs/slowfast_direct_full/data/val.jsonl"))
    smoke.add_argument("--sensor-root", type=Path, default=Path("data"))
    smoke.add_argument("--output", type=Path, default=Path("outputs/slowfast_cluster/smoke_data"))
    args = parser.parse_args(argv)
    if args.mode == "official":
        receipt = prepare_official_data(args.metadata_root, args.sensor_root, args.output,
            args.devkit_root, args.max_rejection_fraction, source_provenance=args.source_provenance)
    else:
        receipt = prepare_smoke_data(args.train_manifest, args.val_manifest, args.sensor_root, args.output)
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return receipt


if __name__ == "__main__":
    main()
