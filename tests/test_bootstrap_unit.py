"""``output.bootstrap.unit``: event (default, published) vs clip (cluster) bootstrap."""

import numpy as np
import pytest

from fdbja.config import Config
from fdbja.runner import _aggregate
from fdbja.summary import CI_NOTE, bootstrap_ci, bootstrap_unit, scoring_conditions


def test_cluster_bootstrap_is_wider_for_correlated_events():
    # 20 clips with 5 identical events each: the event bootstrap sees 100
    # independent values, the clip bootstrap 20
    clips = np.repeat(np.arange(20), 5)
    vals = (clips % 2).astype(float)
    ev = bootstrap_ci(vals, 4000, seed=0)
    cl = bootstrap_ci(vals, 4000, seed=0, clusters=clips)
    assert (cl[1] - cl[0]) > 1.5 * (ev[1] - ev[0])
    assert cl[0] <= 0.5 <= cl[1]


def test_cluster_bootstrap_with_singleton_clusters_matches_the_spread():
    x = np.random.default_rng(1).random(200)
    ev = bootstrap_ci(x, 4000, seed=3)
    cl = bootstrap_ci(x, 4000, seed=3, clusters=range(200))
    assert cl == pytest.approx(ev, abs=0.02)


def test_cluster_bootstrap_edge_cases():
    assert bootstrap_ci([], 100, clusters=[]) is None
    assert bootstrap_ci([None, None], 100, clusters=["a", "b"]) is None
    assert bootstrap_ci([1.0, 1.0, None], 100, clusters=["a", "b", "c"]) == [1.0, 1.0]
    with pytest.raises(ValueError, match="one cluster label"):
        bootstrap_ci([1.0], 100, clusters=["a", "b"])
    with pytest.raises(ValueError, match="mean only"):
        bootstrap_ci([1.0], 100, clusters=["a"], stat=np.median)


def test_unit_setting():
    assert bootstrap_unit(Config({})) == "event"
    assert bootstrap_unit(Config({"output": {"bootstrap": {"unit": "clip"}}})) == "clip"
    with pytest.raises(ValueError, match="event\\|clip"):
        bootstrap_unit(Config({"output": {"bootstrap": {"unit": "session"}}}))


def _rows():
    return [{"task": "pause", "model": "m", "clip": f"c{i // 4}", "TO": (i // 4) % 2,
             "mean_dkl": None} for i in range(40)]


def _config(unit=None):
    boot = {"n": 2000}
    if unit:
        boot["unit"] = unit
    return Config({"tasks": ["pause"], "models": [{"name": "m", "adapter": "offline"}],
                   "output": {"bootstrap": boot}})


def test_default_unit_keeps_the_event_intervals():
    a = _aggregate(_rows(), _config())
    b = _aggregate(_rows(), _config("event"))
    assert a["summary"][0]["TOR_ci"] == b["summary"][0]["TOR_ci"]
    assert a["ci"] == CI_NOTE


def test_clip_unit_in_score_offline():
    ev = _aggregate(_rows(), _config())["summary"][0]["TOR_ci"]
    res = _aggregate(_rows(), _config("clip"))
    cl = res["summary"][0]["TOR_ci"]
    assert res["summary"][0]["TOR"] == 0.5
    assert (cl[1] - cl[0]) > (ev[1] - ev[0])
    assert "over clips" in res["ci"]
    assert "over clips" in scoring_conditions(_config("clip"), {})["ci"]
