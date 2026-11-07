"""Paired inference treats logs and fusion seeds as the sampling units."""
import numpy as np
import pytest


def test_constant_paired_matrix_has_constant_two_way_interval():
    from slowfast_v11.adapter_statistics import paired_adapter_statistics

    result = paired_adapter_statistics(np.full((3, 8), .03))
    assert result["mean_gain_m"] == pytest.approx(.03)
    assert result["ci95_m"] == pytest.approx([.03, .03])
    assert result["per_seed_gain_m"] == pytest.approx([.03] * 3)
    assert result["per_log_gain_m"] == pytest.approx([.03] * 8)
    assert result["positive_log_count"] == 8
    assert result["seed_count"] == 3 and result["log_count"] == 8
    assert result["replicates"] == 20_000
    assert result["analysis_seed"] == 20261107
    assert result["passes_primary_gate"] is True


def rows_for(gains, counts=(2, 1)):
    return [
        {"seed": seed, "log": log, "token": f"{log}-{window}",
         "ade_reference": 2. + gains[seed_index][log_index], "ade_correct": 2.}
        for seed_index, seed in enumerate([42, 43, 44])
        for log_index, log in enumerate(["a", "b"])
        for window in range(counts[log_index])
    ]


def test_builder_pairs_windows_then_weights_logs_equally():
    from slowfast_v11.adapter_statistics import paired_adapter_statistics, paired_gain_matrix

    rows = rows_for([[0., 1.]] * 3, counts=(5, 1))
    gains = paired_gain_matrix(rows, seeds=[42, 43, 44], logs=["a", "b"])
    assert gains.shape == (3, 2)
    assert np.array_equal(gains, np.array([[0., 1.]] * 3))
    result = paired_adapter_statistics(gains, seeds=[42, 43, 44], logs=["a", "b"])
    assert result["mean_gain_m"] == .5
    assert result["mean_gain_m"] != np.mean([row["ade_reference"] - row["ade_correct"] for row in rows])
    assert result["seeds"] == [42, 43, 44] and result["logs"] == ["a", "b"]


def test_identical_pairs_have_zero_effect_regardless_of_window_count():
    from slowfast_v11.adapter_statistics import paired_adapter_statistics, paired_gain_matrix

    rows = rows_for([[0., 0.]] * 3, counts=(17, 1))
    result = paired_adapter_statistics(paired_gain_matrix(rows, seeds=[42, 43, 44], logs=["a", "b"]))
    assert result["mean_gain_m"] == 0
    assert result["ci95_m"] == [0., 0.]
    assert result["passes_primary_gate"] is False


def test_bootstrap_independently_resamples_cartesian_seed_and_log_indices():
    from slowfast_v11.adapter_statistics import paired_adapter_statistics

    gains = np.array([[.01, -.03, .2, -.1], [.07, .01, -.02, .3], [-.04, .1, .02, .08]])
    result = paired_adapter_statistics(gains)
    repeated = paired_adapter_statistics(gains)
    assert result == repeated
    rng = np.random.default_rng(20261107)
    seed_indices = rng.integers(3, size=(20_000, 3))
    log_indices = rng.integers(4, size=(20_000, 4))
    expected = gains[seed_indices[:, :, None], log_indices[:, None, :]].mean(axis=(1, 2))
    assert result["ci95_m"] == pytest.approx(np.percentile(expected, [2.5, 97.5]))
    assert result["per_seed_gain_m"] == pytest.approx(gains.mean(1))
    assert result["per_log_gain_m"] == pytest.approx(gains.mean(0))
    assert "fusion training only" in result["seed_scope"]


def test_bootstrap_does_not_modify_global_numpy_rng():
    from slowfast_v11.adapter_statistics import paired_adapter_statistics

    np.random.seed(5)
    before = np.random.get_state()
    paired_adapter_statistics(np.ones((3, 8)))
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]


@pytest.mark.parametrize("value, passes", [(.02, True), (.019, False), (0., False), (-.02, False)])
def test_primary_practical_threshold_boundary(value, passes):
    from slowfast_v11.adapter_statistics import paired_adapter_statistics

    assert paired_adapter_statistics(np.full((3, 8), value))["passes_primary_gate"] is passes


def test_positive_mean_with_zero_lower_bound_fails_primary_gate():
    from slowfast_v11.adapter_statistics import paired_adapter_statistics

    result = paired_adapter_statistics(np.tile([0., .1], (3, 1)))
    assert result["mean_gain_m"] >= .02
    assert result["ci95_m"][0] == 0
    assert result["passes_primary_gate"] is False


@pytest.mark.parametrize("gains", [[], [1., 2.], [[1., 2.], [1.]], [[float("nan")]],
                                    [[float("inf")]], np.empty((0, 2)), np.empty((2, 0)), [[True]]])
def test_invalid_gain_matrices_are_rejected(gains):
    from slowfast_v11.adapter_statistics import paired_adapter_statistics

    with pytest.raises(ValueError):
        paired_adapter_statistics(gains)


@pytest.mark.parametrize("kwargs", [
    {"seeds": [42, 42, 44]}, {"logs": ["a", "a"]}, {"seeds": [42]}, {"logs": ["a"]},
    {"replicates": 0}, {"replicates": True}, {"replicates": 2.5},
    {"analysis_seed": -1}, {"threshold_m": float("nan")}, {"threshold_m": -1.},
])
def test_bad_analysis_configuration_is_rejected(kwargs):
    from slowfast_v11.adapter_statistics import paired_adapter_statistics

    with pytest.raises(ValueError):
        paired_adapter_statistics(np.zeros((3, 2)), **kwargs)


@pytest.mark.parametrize("problem", ["duplicate", "incomplete", "tokens", "nonfinite", "unknown", "missing", "negative"])
def test_builder_rejects_unpaired_or_invalid_window_rows(problem):
    from slowfast_v11.adapter_statistics import paired_gain_matrix

    rows = rows_for([[.1, .2]] * 3)
    if problem == "duplicate":
        rows.append(dict(rows[0]))
    elif problem == "incomplete":
        rows = [row for row in rows if (row["seed"], row["log"]) != (44, "b")]
    elif problem == "tokens":
        rows[0]["token"] = "unmatched"
    elif problem == "nonfinite":
        rows[0]["ade_correct"] = float("nan")
    elif problem == "unknown":
        rows[0]["log"] = "c"
    elif problem == "missing":
        del rows[0]["ade_reference"]
    else:
        rows[0]["ade_correct"] = -.1
    with pytest.raises(ValueError):
        paired_gain_matrix(rows, seeds=[42, 43, 44], logs=["a", "b"])
