"""Bounded real async inference; colocated smoke requires an explicit override."""
import argparse
import json

from fast_cluster.async_runtime import run_async
from fast_cluster.config import read_config


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/navsim_v1_split_history_smoke.yaml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--fragment', default='outputs/slowfast_cluster/async_fragment/fragment.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--fast-device', default='cuda:0')
    parser.add_argument('--slow-device', default='cuda:1')
    parser.add_argument('--allow-colocated-smoke', action='store_true')
    parser.add_argument('--max-plans', '--ticks', dest='max_plans', type=int, choices=range(12, 25), default=16)
    parser.add_argument('--refresh-period-s', type=float, default=1.)
    parser.add_argument('--wall-timeout-s', type=float, default=180.)
    parser.add_argument('--startup-timeout-s', type=float, default=300.)
    parser.add_argument('--transport-root', help='Same-host CPU mmap files; prefer node-local NVMe (default TMPDIR or output)')
    return parser


def main():
    args = build_parser().parse_args()
    result = run_async(read_config(args.config), args.fragment, args.checkpoint, args.output,
        fast_device=args.fast_device, slow_device=args.slow_device,
        allow_colocated_smoke=args.allow_colocated_smoke, max_plans=args.max_plans,
        refresh_period_s=args.refresh_period_s, wall_timeout_s=args.wall_timeout_s,
        startup_timeout_s=args.startup_timeout_s, transport_root=args.transport_root)
    print(json.dumps({key: result[key] for key in ['planning_count', 'source_count',
        'source_frequency_hz', 'planning_frequency_hz', 'consumed_versions',
        'dropped_source_windows', 'busy_refresh_skips', 'shutdown_clean', 'multi_gpu_validated']}))


if __name__ == '__main__':
    main()
