"""Auditable navigation contract for official Bench2Drive annotations.

The provided route waypoint is legal at observation time. It is neither an
expert future pose nor a synthesized fixed-distance lookahead. The exact
private expert's planner thresholds are unavailable in the public release.
"""
import hashlib
import json
from pathlib import Path

B2D_REVISION = '7ec25d1c9f7522d923ce5f3420986cef1cb2d956'
ZOO_REVISION = '498c1f799dd90faf840dedb3f0d3234ec2e567db'
NAVIGATION_METADATA = {
    'target_fields': ['x_command_near', 'y_command_near'],
    'command_field': 'command_near',
    'coordinate_space': 'CARLA_world_xy',
    'origin': 'ego_center_from_gnss_extrinsic',
    'gnss_mount_xy_m': [-1.4, 0.],
    'privileged_ego_pose_used': False,
    'origin_calibration': 'current_GNSS_xy_minus_rotated_known_GNSS_mount; residual_GNSS_noise_retained',
    'orientation': 'current_sensor_compass_rad_minus_pi_over_two',
    'output_axes': 'x_forward_y_left',
    'future_expert_positions_used': False,
    'collector_revision': B2D_REVISION,
    'online_reference_revision': ZOO_REVISION,
    'field_contract': f'https://github.com/Thinklab-SJTU/Bench2Drive/blob/{B2D_REVISION}/docs/anno.md#L128-L137',
    'online_reference': f'https://github.com/Thinklab-SJTU/Bench2DriveZoo/blob/{ZOO_REVISION}/team_code/vad_b2d_agent.py#L292',
    'lookahead': 'official_current_near_waypoint; exact_private_collector_thresholds_unpublished',
}


def load_source_audit(path):
    """Read evidence and verify retained source bytes before trusting its claims."""
    path = Path(path)
    audit = json.loads(path.read_text())
    sources = audit.get('sources', [])
    if not sources:
        raise ValueError('source_audit contains no retained official sources')
    for item in sources:
        source = Path(item['path'])
        if not source.is_absolute():
            source = path.parent / source
        if hashlib.sha256(source.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError(f'source_audit hash mismatch: {source}')
    audit['audit_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return audit


def validate_nominal_clock(source_audit):
    """Allow a declared nominal clock without claiming measured timestamps."""
    timing = (source_audit or {}).get('timing', {})
    if (not (source_audit or {}).get('audit_sha256')
            or timing.get('nominal_hz') != 10
            or timing.get('declared_collection_hz_verified') is not True):
        raise ValueError('nominal_simulation_time_10hz requires verified source_audit')
    return {
        'hz': 10, 'absolute_timestamp_available': False,
        'clock': 'nominal_declared_collection_frame_index',
        'source_audit_sha256': source_audit['audit_sha256'],
        'save_stride_verified': timing.get('save_stride_verified', False),
        'limitation': timing.get('save_stride_limitation'),
    }
