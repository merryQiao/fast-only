"""Inspect actual Bench2Drive annotation schema, timing and geometry."""
import argparse
import json
from pathlib import Path

from fast_only.data import collect_windows, discover_routes, read_annotation
from fast_only.navigation import load_source_audit, NAVIGATION_METADATA


def inspect(root, allow_frame_index_timing=False, timing_policy='measured_timestamp', source_audit=None):
    routes = discover_routes(root)
    evidence = load_source_audit(source_audit) if isinstance(source_audit, (str, Path)) else source_audit
    _, report = collect_windows(root, allow_frame_index_timing=allow_frame_index_timing,
                                timing_policy=timing_policy, source_audit=evidence)
    schemas = []
    for route in routes:
        paths = sorted((route / 'anno').glob('*.json*'))
        if not paths:
            continue
        raw = read_annotation(paths[0])
        schemas.append({'route': route.name, 'first_file': str(paths[0]), 'keys': sorted(raw),
                        'field_types': {k: type(v).__name__ for k, v in raw.items()},
                        'front_images': len(list((route / 'camera/rgb_front').glob('*'))),
                        'annotations': len(paths),
                        'sample_observations': {k: raw.get(k) for k in ('x', 'y', 'theta', 'speed', 'command_near', 'x_command_near', 'y_command_near')},
                        'timestamp_present': any(k in raw for k in ('timestamp', 'timestamp_seconds', 'elapsed_seconds'))})
    return dict(report, schemas=schemas,
                schema_difference='Actual v0.0.4 adds desired_speed/disappear/teleport fields; never used as model inputs.',
                source_audit=evidence or {
                    'navigation': dict(NAVIGATION_METADATA, verified=True),
                    'timing': {'policy': timing_policy, 'absolute_timestamps_available': False,
                        'limitation': 'No actual timestamps in released annotations. Nominal10Hz requires source audit opt-in.'},
                    'position': 'Current sensor GPS/compass for input, official ego bounding-box pose for supervision only.',
                    'forbidden_inputs': 'No control, expert assessments, should_brake, other actors, privileged input pose, future state or future commands.'})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', default='data/smoke')
    p.add_argument('--output', default='outputs/fast_only/data/inspection.json')
    p.add_argument('--allow-frame-index-timing', action='store_true')
    p.add_argument('--timing-policy', default='measured_timestamp', choices=['measured_timestamp','nominal_simulation_time_10hz'])
    p.add_argument('--source-audit')
    a = p.parse_args()
    report = inspect(a.data_root, a.allow_frame_index_timing, a.timing_policy, a.source_audit)
    path = Path(a.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
