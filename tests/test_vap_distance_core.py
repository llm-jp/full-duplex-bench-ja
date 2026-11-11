"""Core of VAP-distance on synthetic probabilities: window extraction, KL
direction, 20 Hz alignment and the boundary weights."""

import numpy as np
import pytest

from fdbja.metrics import vap_distance as vd

FR = 20


def _onehotish(n, cls, mass=0.9, k=4):
    p = np.full((n, k), (1 - mass) / (k - 1))
    p[:, cls] = mass
    return p


# --- KL direction ------------------------------------------------------------

def test_kl_is_forward_human_to_model():
    p = np.array([[0.9, 0.1]])
    q = np.array([[0.5, 0.5]])
    fwd = 0.9 * np.log(0.9 / 0.5) + 0.1 * np.log(0.1 / 0.5)
    rev = 0.5 * np.log(0.5 / 0.9) + 0.5 * np.log(0.5 / 0.1)
    assert vd.kl_series(p, q)[0] == pytest.approx(fwd)
    assert vd.kl_series(q, p)[0] == pytest.approx(rev)
    assert fwd != pytest.approx(rev)


def test_kl_zero_for_identical_and_nonnegative():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(256), size=50)
    q = rng.dirichlet(np.ones(256), size=50)
    assert np.allclose(vd.kl_series(p, p), 0.0, atol=1e-9)
    assert (vd.kl_series(p, q) >= -1e-12).all()


def test_kl_mode_covering_penalises_missing_human_mass():
    # the model puts no mass where the human does -> large (clipped, finite) divergence
    p = np.array([[1.0, 0.0]])
    q = np.array([[0.0, 1.0]])
    d = vd.kl_series(p, q)[0]
    assert np.isfinite(d) and d > 10
    # the reverse case (human zero where the model has mass) is cheap in this direction
    p2 = np.array([[0.5, 0.5]])
    q2 = np.array([[0.5 - 1e-9, 0.5 + 1e-9]])
    assert vd.kl_series(p2, q2)[0] == pytest.approx(0.0, abs=1e-6)


# --- 20 Hz alignment -----------------------------------------------------------

def test_kl_series_truncates_to_the_shorter_stream():
    p = _onehotish(40, 0)
    q = _onehotish(37, 1)
    assert len(vd.kl_series(p, q)) == 37
    assert len(vd.kl_series(q, p)) == 37


def test_frames_line_up_with_window_seconds_at_20hz():
    # frame i covers audio up to (i + 1) / 20 s; the model differs only in 1.0-2.0 s
    n = 80
    times = (np.arange(n) + 1) / FR
    p = _onehotish(n, 0)
    q = p.copy()
    diff = (times > 1.0) & (times <= 2.0)
    q[diff] = _onehotish(int(diff.sum()), 1)
    dkl = vd.kl_series(p, q)
    inside = vd.window_stats(times, dkl, 1.05, 2.0)
    before = vd.window_stats(times, dkl, 0.0, 1.0)
    after = vd.window_stats(times, dkl, 2.05, 4.0)
    assert inside["n_frames"] == 20
    assert inside["mean_dkl"] > 1.0
    assert before["mean_dkl"] == pytest.approx(0.0, abs=1e-9)
    assert after["mean_dkl"] == pytest.approx(0.0, abs=1e-9)


# --- windows -------------------------------------------------------------------

def test_window_stats_inclusive_bounds_and_empty():
    times = np.arange(10) / 10
    dkl = np.arange(10, dtype=float)
    s = vd.window_stats(times, dkl, 0.2, 0.4)
    assert s == {"mean_dkl": 3.0, "max_dkl": 4.0, "n_frames": 3}
    assert vd.window_stats(times, dkl, 5.0, 6.0) is None


def test_event_windows_per_task():
    assert vd.event_window("pause", {"start": 1.0, "end": 1.6}) == (1.0, 1.6)
    assert vd.event_window("smooth_tt", {"a_end": 3.0}) == (3.0, 5.0)
    assert vd.event_window("smooth_tt", {"a_end": 3.0}, tt_horizon=1.0) == (3.0, 4.0)
    assert vd.event_window("bc", {"a_utt_start": 0.5, "a_utt_end": 4.0}) == (0.5, 4.0)
    with pytest.raises(ValueError):
        vd.event_window("user_interrupt", {})


def test_boundary_windows():
    assert vd.boundary_window("pause", {"start": 10.0, "end": 10.6}) == (5.0, 12.1, 10.0)
    lo, hi, t0 = vd.boundary_window("smooth_tt", {"a_end": 10.0})
    assert (lo, hi, t0) == (4.5, 13.5, 10.0)
    assert vd.boundary_window("smooth_tt", {"a_end": 10.0}, pre=1.0, post=0.0) == (8.5, 12.0, 10.0)
    with pytest.raises(ValueError):
        vd.boundary_window("bc", {})


# --- boundary weights ---------------------------------------------------------

def test_boundary_weights_shape():
    t = np.array([8.0, 10.0, 12.0, 14.0])
    w = vd.boundary_weights(t, 10.0, 2.0)
    assert w[1] == 1.0
    assert w[0] == pytest.approx(np.exp(-1.0)) and w[2] == pytest.approx(np.exp(-1.0))
    assert w[3] == pytest.approx(np.exp(-2.0))
    assert np.all(vd.boundary_weights(t, 10.0, np.inf) == 1.0)
    with pytest.raises(ValueError):
        vd.boundary_weights(t, 10.0, 0.0)


def test_weighted_mean_emphasises_frames_near_the_boundary():
    times = np.arange(0, 10, 1 / FR)
    dkl = np.where(np.abs(times - 5.0) < 0.5, 1.0, 0.0)   # error only near t0
    uniform = vd.window_stats(times, dkl, 0.0, 9.95)["mean_dkl"]
    weighted = vd.weighted_window_stats(times, dkl, 0.0, 9.95, 5.0, 2.0)["mean_dkl"]
    assert weighted > 2 * uniform
    # tau = inf reproduces the equal-weight mean
    eq = vd.weighted_window_stats(times, dkl, 0.0, 9.95, 5.0, np.inf)["mean_dkl"]
    assert eq == pytest.approx(uniform)
    # the same error far from t0 counts less than near it
    far = np.where(np.abs(times - 1.0) < 0.5, 1.0, 0.0)
    assert vd.weighted_window_stats(times, far, 0.0, 9.95, 5.0, 2.0)["mean_dkl"] < weighted
    assert vd.weighted_window_stats(times, dkl, 20.0, 30.0, 25.0) is None


def test_score_from_dkl():
    assert vd.score_from_dkl(0.0, 3.0) == 1.0
    assert vd.score_from_dkl(1.0, np.log(2)) == pytest.approx(0.5)
