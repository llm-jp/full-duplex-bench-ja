"""alpha calibration of the exp(-alpha * D_KL) score (``metrics.vap_distance.alpha``)."""

import math
import warnings

import numpy as np
import pytest

from fdbja.config import Config
from fdbja.metrics import vap_distance as vd
from fdbja.runner import _aggregate


def test_calibrate_alpha_maps_median_to_half():
    a = vd.calibrate_alpha(np.array([0.5, 1.0, 2.0]))
    assert a == pytest.approx(math.log(2))
    assert vd.score_from_dkl(1.0, a) == pytest.approx(0.5)


@pytest.mark.parametrize("vals", [[], [0.0, 0.0, 0.1], [np.nan, np.inf], [-1.0, 0.0]])
def test_calibrate_alpha_degenerate_returns_none(vals):
    with warnings.catch_warnings():
        warnings.simplefilter("error")       # no divide-by-zero / empty-median warnings
        assert vd.calibrate_alpha(np.array(vals)) is None


def test_calibrate_alpha_ignores_non_finite():
    assert vd.calibrate_alpha(np.array([1.0, np.nan, 1.0])) == pytest.approx(math.log(2))


def _rows():
    rows = []
    # Pause, desired TO = 0. The silence baseline is "desired" everywhere with D_KL ~ 0.
    for i in range(4):
        rows.append({"task": "pause", "model": "sys", "clip": f"c{i}", "TO": 0,
                     "mean_dkl": 2.0})
        rows.append({"task": "pause", "model": "silence_baseline", "clip": f"c{i}",
                     "TO": 0, "mean_dkl": 0.0})
        rows.append({"task": "pause", "model": "talk", "clip": f"c{i}", "TO": 1,
                     "mean_dkl": 3.0})
    return rows


def _config(**vap):
    return Config({"tasks": ["pause"],
                   "models": [{"name": "sys", "adapter": "offline"},
                              {"name": "talk", "adapter": "offline"},
                              {"name": "silence_baseline", "adapter": "silence"}],
                   "metrics": {"vap_distance": {"enabled": True, **vap}},
                   "output": {"bootstrap": {"n": 50}}})


def test_default_calibration_includes_baselines():
    by = {r["model"]: r for r in _aggregate(_rows(), _config())["summary"]}
    # desired D_KL = [2.0]*4 + [0.0]*4 -> median 1.0 -> alpha = log 2
    assert by["sys"]["alpha"] == round(math.log(2), 4)
    assert "alpha_calibration" not in _aggregate(_rows(), _config())


def test_calibration_can_exclude_baselines():
    res = _aggregate(_rows(), _config(calibration_exclude_baselines=True))
    by = {r["model"]: r for r in res["summary"]}
    assert by["sys"]["alpha"] == round(math.log(2) / 2.0, 4)
    assert by["sys"]["score"] == 0.5
    assert res["alpha_calibration"].startswith("desired events of non-baseline")


def test_zero_median_gives_no_score_instead_of_inf():
    rows = [r for r in _rows() if r["model"] != "sys"]
    with pytest.warns(RuntimeWarning, match="alpha calibration skipped"):
        res = _aggregate(rows, _config())
    for r in res["summary"]:
        assert "alpha" not in r and "score" not in r
        assert "event_mean_dkl" in r


@pytest.mark.parametrize("bad", [0, -1.0])
def test_non_positive_configured_alpha_raises(bad):
    with pytest.raises(ValueError, match="alpha must be positive"):
        _aggregate(_rows(), _config(alpha=bad))


def test_configured_alpha_is_used():
    by = {r["model"]: r for r in _aggregate(_rows(), _config(alpha=1.0))["summary"]}
    assert by["sys"]["alpha"] == 1.0
    assert by["sys"]["score"] == round(math.exp(-2.0), 4)
