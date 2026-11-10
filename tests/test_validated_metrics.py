"""Boundary-weighted VAP-distance and the Pause response metric
(``metrics.vap_distance.weighting: boundary``, ``metrics.pause_response``)."""

import csv
import importlib.util
import os

import numpy as np
import pytest
import soundfile as sf
import yaml

from fdbja.config import Config
from fdbja.metrics import vap_distance as vd
from fdbja.metrics.pause_response import PauseResponseConfig, judge_event, judge_events, voiced_runs

FR = 20
DT = 1 / FR


# --- boundary-weighted VAP-distance -------------------------------------------------

def test_boundary_window_follows_the_judgement_window():
    assert vd.boundary_window("pause", {"start": 10.0, "end": 10.6}) == (5.0, 12.1, 10.0)
    lo, hi, t0 = vd.boundary_window("smooth_tt", {"a_end": 8.0})
    assert (lo, hi, t0) == (2.5, 11.5, 8.0)
    assert vd.boundary_window("pause", {"start": 3.0, "end": 3.5}, pre=2.0, post=2.0) == (1.0, 5.5, 3.0)
    with pytest.raises(ValueError):
        vd.boundary_window("bc", {"a_utt_start": 0.0, "a_utt_end": 1.0})


@pytest.mark.parametrize("tau", [0.5, 1.0, 2.0, 3.0, 50.0])
def test_weights_are_normalised(tau):
    t = np.arange(0, 20, DT)
    # a constant series keeps its value whatever the weights
    st = vd.weighted_window_stats(t, np.full(len(t), 0.7), 3.0, 12.0, 8.0, tau)
    assert st["mean_dkl"] == pytest.approx(0.7)
    assert st["n_frames"] == int(((t >= 3.0) & (t <= 12.0)).sum())
    w = vd.boundary_weights(t, 8.0, tau)
    assert w.max() == pytest.approx(1.0) and np.all(w > 0)


def test_boundary_weighting_emphasises_the_boundary():
    t = np.arange(0, 20, DT)
    near = np.where(np.abs(t - 8.0) < 0.3, 1.0, 0.0)
    far = np.where(np.abs(t - 3.5) < 0.3, 1.0, 0.0)
    a = vd.weighted_window_stats(t, near, 3.0, 12.0, 8.0, 2.0)["mean_dkl"]
    b = vd.weighted_window_stats(t, far, 3.0, 12.0, 8.0, 2.0)["mean_dkl"]
    u_a = vd.weighted_window_stats(t, near, 3.0, 12.0, 8.0, np.inf)["mean_dkl"]
    u_b = vd.weighted_window_stats(t, far, 3.0, 12.0, 8.0, np.inf)["mean_dkl"]
    assert u_a == pytest.approx(u_b)          # equal weights: same mass, same mean
    assert a > u_a > b                        # boundary weights favour frames near t0
    # the shorter tau, the more it concentrates on t0
    assert vd.weighted_window_stats(t, near, 3.0, 12.0, 8.0, 0.5)["mean_dkl"] > a


def test_uniform_is_the_limit_of_large_tau():
    rng = np.random.default_rng(0)
    t = np.arange(0, 20, DT)
    d = rng.gamma(1.0, 1.0, len(t))
    uni = vd.window_stats(t, d, 3.0, 12.0)["mean_dkl"]
    assert vd.weighted_window_stats(t, d, 3.0, 12.0, 8.0, np.inf)["mean_dkl"] == pytest.approx(uni)
    assert vd.weighted_window_stats(t, d, 3.0, 12.0, 8.0, 1e7)["mean_dkl"] == pytest.approx(uni, rel=1e-6)
    assert vd.weighted_window_stats(t, d, 30.0, 40.0, 35.0, 2.0) is None
    with pytest.raises(ValueError):
        vd.boundary_weights(t, 8.0, 0.0)


# --- Pause response ------------------------------------------------------------------

def _grid(seconds=10.0):
    return np.arange(int(round(seconds * FR))) / FR


def _on(t, *spans):
    v = np.zeros(len(t), dtype=bool)
    for s, e in spans:   # frames whose start lies in [s, e)
        v |= (t >= s - 1e-9) & (t < e - 1e-9)
    return v


GAP = (5.0, 5.6)


def test_voiced_runs_end_one_frame_after_the_last_voiced_frame():
    t = _grid(2.0)
    assert voiced_runs(_on(t, (0.5, 0.7), (1.0, 1.05)), t, DT) == [
        pytest.approx((0.5, 0.7)), pytest.approx((1.0, 1.05))]
    assert voiced_runs(np.zeros(len(t), bool), t, DT) == []


def test_response_needs_0_2_s_inside_the_gap():
    t = _grid()
    a = _on(t, (0.0, GAP[0]), (GAP[1], 10.0))
    # exactly 0.2 s inside the gap (four 50 ms frames) counts as a reaction
    assert judge_event(_on(t, (5.2, 5.4)), a, t, *GAP)["no_response"] == 0
    assert judge_event(_on(t, (5.25, 5.4)), a, t, *GAP)["no_response"] == 1
    # only the part inside the gap counts: a run that ends 0.15 s into the gap is no reaction
    assert judge_event(_on(t, (4.0, 5.15)), a, t, *GAP)["no_response"] == 1
    # and a run that is cut at the gap end by 0.2 s still is one
    assert judge_event(_on(t, (5.4, 5.9)), a, t, *GAP)["no_response"] == 0


def test_interrupt_rules_and_their_boundaries():
    t = _grid()
    a = _on(t, (0.0, GAP[0]), (GAP[1], 10.0))
    # starts in the gap and lasts exactly 0.5 s past A's resumption -> interrupt
    j = judge_event(_on(t, (5.2, 6.1)), a, t, *GAP)
    assert (j["no_response"], j["interrupt"], j["pause_response"]) == (0, 1, 1)
    assert judge_event(_on(t, (5.2, 6.05)), a, t, *GAP)["interrupt"] == 0
    # started before the gap: not this rule (the overlap rule decides)
    assert judge_event(_on(t, (4.8, 7.0)), a, t, *GAP)["interrupt"] == 0
    # overlap with A in the 2 s before the gap: exactly 0.5 s is allowed, more is not
    assert judge_event(_on(t, (3.5, 4.0)), a, t, *GAP)["interrupt"] == 0
    j = judge_event(_on(t, (3.5, 4.05)), a, t, *GAP)
    assert j["interrupt"] == 1 and j["pre_overlap"] == pytest.approx(0.55)
    # overlap outside the 2 s window does not count
    assert judge_event(_on(t, (1.0, 2.9)), a, t, *GAP)["interrupt"] == 0
    # speech before the gap while A is silent is not an overlap
    a_hole = _on(t, (0.0, 3.0), (GAP[1], 10.0))
    assert judge_event(_on(t, (3.0, 4.5)), a_hole, t, *GAP)["interrupt"] == 0


def test_empty_and_constant_responses():
    t = _grid()
    a = _on(t, (0.0, GAP[0]), (GAP[1], 10.0))
    silent = judge_event(np.zeros(len(t), bool), a, t, *GAP)
    assert (silent["no_response"], silent["interrupt"], silent["pause_response"]) == (1, 0, 1)
    talk = judge_event(np.ones(len(t), bool), a, t, *GAP)
    assert (talk["no_response"], talk["interrupt"], talk["pause_response"]) == (0, 1, 1)
    assert talk["pre_overlap"] == pytest.approx(2.0)
    brief = judge_event(_on(t, (5.1, 5.45)), a, t, *GAP)
    assert brief["pause_response"] == 0


def test_thresholds_come_from_the_config():
    t = _grid()
    a = _on(t, (0.0, GAP[0]), (GAP[1], 10.0))
    cfg = PauseResponseConfig.from_config({"enabled": True, "min_response_seconds": 0.3,
                                           "pre_overlap_max_seconds": 1.0})
    assert judge_event(_on(t, (5.2, 5.4)), a, t, *GAP, cfg)["no_response"] == 1
    assert judge_event(_on(t, (3.5, 4.05)), a, t, *GAP, cfg)["interrupt"] == 0
    with pytest.raises(ValueError, match="unknown metrics.pause_response"):
        PauseResponseConfig.from_config({"min_response": 0.3})


def test_judge_events_reads_a_from_gt_and_system_from_combined(tmp_path):
    sr = 16000
    n = 10 * sr
    tt = np.arange(n) / sr
    tone = 0.3 * np.sin(2 * np.pi * 220 * tt).astype(np.float32)
    a = np.where((tt < GAP[0]) | (tt >= GAP[1]), tone, 0).astype(np.float32)
    b_human = np.zeros(n, np.float32)
    b_sys = np.where((tt >= 5.1) & (tt < 5.45), tone, 0).astype(np.float32)
    sf.write(tmp_path / "gt.wav", np.stack([a, b_human], 1), sr)
    sf.write(tmp_path / "combined.wav", np.stack([np.zeros(n, np.float32), b_sys], 1), sr)
    ev = [{"start": GAP[0], "end": GAP[1]}, {"start": 8.0, "end": 8.6}]
    res = judge_events(str(tmp_path / "gt.wav"), str(tmp_path / "combined.wav"), ev)
    assert [r["pause_response"] for r in res] == [0, 1]
    assert res[1]["no_response"] == 1


# --- wiring into score-offline --------------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DUMMY = os.path.join(ROOT, "examples", "dummy")


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"_dummy_{name}", os.path.join(DUMMY, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _EnergyVAP:
    """Stand-in VAP backend: class probabilities from the channel energies."""

    def probs(self, left, right):
        hop = 16000 // FR
        n = min(len(left), len(right)) // hop

        def e(x):
            return np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(1))
        ea, eb = e(left), e(right)
        p = np.full((n, 256), 1.0)
        p[:, 0] += 50 * ea
        p[:, 255] += 50 * eb
        return np.arange(n) / FR, p / p.sum(1, keepdims=True)


def _dummy_config(tmp_path, metrics):
    corpus = _load("make_dummy_corpus")
    _load("energy_asr")
    corpus.make_dummy_corpus(str(tmp_path / "zoom1"), n_sessions=4, seed=0)
    with open(os.path.join(DUMMY, "config.yaml"), encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw["corpus"] = {"zoom1_root": str(tmp_path / "zoom1"), "eval_dir": str(tmp_path / "eval")}
    raw["detection"]["clips_per_task"] = 3
    raw["output"]["dir"] = str(tmp_path / "results")
    for k, v in metrics.items():
        raw["metrics"].setdefault(k, {}).update(v)
    return Config(raw)


def _run(config):
    from fdbja.data.reconstruct import reconstruct
    from fdbja.runner import run_infer, run_transcribe, score_offline

    quiet = lambda *a, **k: None  # noqa: E731
    reconstruct(config, log=quiet)
    run_infer(config, log=quiet)
    run_transcribe(config, log=quiet)
    return score_offline(config, log=quiet)


def test_default_output_has_no_proposed_columns(tmp_path):
    from fdbja.summary import write_summary

    res = _run(_dummy_config(tmp_path, {}))
    keys = {k for r in res["summary"] for k in r}
    assert not keys & {"vapd_boundary", "pause_response", "pause_no_response_rate"}
    assert "pause_response_settings" not in res and "vapd_boundary_settings" not in res
    _, cpath = write_summary(res, str(tmp_path / "out"))
    with open(cpath, encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert not any(c.startswith(("vapd_boundary", "pause_")) for c in header)


# the stand-in VAP gives D_KL = 0 on most desired events, so the alpha calibration divides by 0
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_proposed_metrics_in_the_summary(tmp_path, monkeypatch):
    from fdbja import runner
    from fdbja.summary import write_summary

    monkeypatch.setattr(runner, "_build_vap_backend", lambda config: _EnergyVAP())
    config = _dummy_config(tmp_path, {
        "vap_distance": {"enabled": True, "weighting": "boundary", "tau_seconds": 2.0},
        "pause_response": {"enabled": True}})
    res = _run(config)
    by = {(r["task"], r["model"]): r for r in res["summary"]}
    # boundary weighting only on Smooth TT by default; the equal-weight score stays
    for m in ("silence_baseline", "always_talk_baseline"):
        assert "vapd_boundary" in by[("smooth_tt", m)] and "event_mean_dkl" in by[("smooth_tt", m)]
        assert "vapd_boundary" not in by[("pause", m)]
        assert "pause_response" not in by[("smooth_tt", m)]
    sil = by[("pause", "silence_baseline")]
    assert (sil["pause_no_response_rate"], sil["pause_interrupt_rate"], sil["pause_response"]) == (1.0, 0.0, 1.0)
    talk = by[("pause", "always_talk_baseline")]
    assert talk["pause_no_response_rate"] == 0.0 and talk["pause_interrupt_rate"] == 1.0
    assert res["vapd_boundary_settings"] == {"tau_seconds": 2.0, "tasks": ["smooth_tt"]}
    assert res["pause_response_settings"]["min_response_seconds"] == 0.2
    ev = [e for e in res["events"] if e["task"] == "pause"]
    assert ev and all({"no_response", "interrupt", "pause_response"} <= set(e) for e in ev)
    _, cpath = write_summary(res, str(tmp_path / "out"))
    with open(cpath, encoding="utf-8") as f:
        header = next(csv.reader(f))
    for c in ("vapd_boundary", "vapd_boundary_ci_low", "pause_response",
              "pause_no_response_rate", "pause_interrupt_rate"):
        assert c in header


def test_boundary_weighting_settings_are_checked():
    from fdbja.runner import _boundary_settings

    assert _boundary_settings({}) is None
    assert _boundary_settings({"weighting": "uniform"}) is None
    b = _boundary_settings({"weighting": "boundary", "boundary_tasks": ["pause", "smooth_tt"]})
    assert b == {"tau": 2.0, "tasks": ["pause", "smooth_tt"], "pre": 5.0, "post": 1.5}
    with pytest.raises(ValueError):
        _boundary_settings({"weighting": "gaussian"})
    with pytest.raises(ValueError):
        _boundary_settings({"weighting": "boundary", "boundary_tasks": ["bc"]})
    with pytest.raises(ValueError):
        _boundary_settings({"weighting": "boundary", "tau_seconds": 0})
