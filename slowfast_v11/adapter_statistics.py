"""Paired log/seed inference conditional on the frozen source models.

Seeds represent fusion training alone, not independent backbone or Reader
training. Windows are averaged within each log before any resampling.
"""

from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np


def _labels(values, expected, name):
    try:
        values = list(values)
    except TypeError as error:
        raise ValueError(f"{name} must be distinct seed/log labels") from error
    values = [value.item() if isinstance(value, np.generic) else value for value in values]
    if (not values or (expected is not None and len(values) != expected)
            or any(isinstance(value, bool) or not isinstance(value, (int, str)) or value == "" for value in values)
            or len(set(values)) != len(values)):
        raise ValueError(f"{name} must be nonempty, distinct and match the matrix dimension")
    return values


def paired_gain_matrix(rows, *, seeds, logs):
    """Build paired per-log means from window ADE rows without pseudoreplication.

    Each row has ``seed, log, token, ade_correct, ade_reference``. All declared
    seed/log cells must exist, and each log has identical token sets across
    seeds. Window counts may differ between logs; every resulting log receives
    equal weight. Unrecognized labels and duplicate windows are errors.
    """
    seeds = _labels(seeds, None, "seeds")
    logs = _labels(logs, None, "logs")
    cells = {(seed, log): {} for seed in seeds for log in logs}
    required = {"seed", "log", "token", "ade_correct", "ade_reference"}
    for row in rows:
        if not isinstance(row, Mapping) or not required.issubset(row):
            raise ValueError("Each paired window row requires seed/log/token and both ADE values")
        seed, log, token = row["seed"], row["log"], row["token"]
        if isinstance(seed, bool) or not isinstance(seed, (Integral, str)) or not isinstance(log, (Integral, str)):
            raise ValueError("Unknown paired seed/log label")
        if (seed, log) not in cells:
            raise ValueError("Unknown paired seed/log label")
        if not isinstance(token, str) or not token:
            raise ValueError("Window token must be a nonempty string")
        cell = cells[seed, log]
        if token in cell:
            raise ValueError("Duplicate paired seed/log/token window")
        values = [row["ade_correct"], row["ade_reference"]]
        if any(isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value < 0
               for value in values):
            raise ValueError("Paired ADE values must be finite, nonnegative numbers")
        cell[token] = float(values[1]) - float(values[0])
    for log in logs:
        expected = set(cells[seeds[0], log])
        if not expected or any(set(cells[seed, log]) != expected for seed in seeds):
            raise ValueError("Incomplete paired matrix or mismatched token sets across seeds")
    return np.array([[np.mean([cells[seed, log][token] for token in sorted(cells[seed, log])])
                      for log in logs] for seed in seeds], dtype=np.float64)


def paired_adapter_statistics(gains, *, seeds=None, logs=None, replicates=20_000,
                              analysis_seed=20261107, threshold_m=.02):
    """Bootstrap a [fusion seed, log] ADE(reference)-ADE(correct) matrix.

    Each replicate independently samples seed indices and log indices with
    replacement, then averages their Cartesian submatrix. Both model arms use
    the same sampled units because pairing precedes resampling.
    """
    try:
        gains = np.asarray(gains)
    except (TypeError, ValueError) as error:
        raise ValueError("Expected a complete finite [seed,log] gain matrix") from error
    if (gains.ndim != 2 or 0 in gains.shape or gains.dtype.kind not in "fiu"
            or not np.isfinite(gains).all()):
        raise ValueError("Expected a complete finite [seed,log] gain matrix")
    gains = gains.astype(np.float64)
    seed_count, log_count = gains.shape
    seeds = _labels(range(seed_count) if seeds is None else seeds, seed_count, "seeds")
    logs = _labels(range(log_count) if logs is None else logs, log_count, "logs")
    if isinstance(replicates, bool) or not isinstance(replicates, Integral) or replicates <= 0:
        raise ValueError("replicates must be a positive integer")
    if isinstance(analysis_seed, bool) or not isinstance(analysis_seed, Integral) or analysis_seed < 0:
        raise ValueError("analysis_seed must be a nonnegative integer")
    if (isinstance(threshold_m, bool) or not isinstance(threshold_m, Real)
            or not np.isfinite(threshold_m) or threshold_m < 0):
        raise ValueError("threshold_m must be finite and nonnegative")
    rng = np.random.default_rng(analysis_seed)
    seed_indices = rng.integers(seed_count, size=(replicates, seed_count))
    log_indices = rng.integers(log_count, size=(replicates, log_count))
    bootstrap = gains[seed_indices[:, :, None], log_indices[:, None, :]].mean(axis=(1, 2))
    interval = np.percentile(bootstrap, [2.5, 97.5]).tolist()
    per_seed = gains.mean(axis=1)
    per_log = gains.mean(axis=0)
    mean = float(gains.mean())
    return {
        "paired_gain_matrix_m": gains.tolist(),
        "mean_gain_m": mean,
        "median_log_gain_m": float(np.median(per_log)),
        "per_seed_gain_m": per_seed.tolist(),
        "per_log_gain_m": per_log.tolist(),
        "positive_log_count": int((per_log > 0).sum()),
        "ci95_m": interval,
        "seed_count": seed_count,
        "log_count": log_count,
        "seeds": seeds,
        "logs": logs,
        "replicates": int(replicates),
        "analysis_seed": int(analysis_seed),
        "threshold_m": float(threshold_m),
        "passes_primary_gate": bool(mean >= threshold_m and interval[0] > 0),
        "seed_scope": "fusion training only; conditional on frozen Fast, Slow and Reader",
    }
