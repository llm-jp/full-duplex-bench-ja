"""Adapter registry: registered names, ``module:Class`` paths, ``offline``,
and ``infer`` / ``score-offline`` with entries that cannot run here."""

import importlib.util
import os

import numpy as np
import pytest
import yaml

from fdbja.adapters import ADAPTERS, build_adapter, is_offline, register_adapter
from fdbja.adapters.base import DialogueResponse, ModelAdapter
from fdbja.adapters.baselines import SilenceAdapter
from fdbja.config import Config
from fdbja.data.reconstruct import reconstruct
from fdbja.runner import run_infer, run_transcribe, score_offline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
quiet = lambda *a, **k: None  # noqa: E731


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Echo(ModelAdapter):
    def __init__(self, gain=1.0):
        self.gain = gain

    def infer(self, input_audio, sample_rate):
        return DialogueResponse(input_audio, input_audio * self.gain, sample_rate)


def test_register_and_build_by_name():
    register_adapter("_echo_test", _Echo)
    try:
        a = build_adapter({"name": "x", "adapter": "_echo_test", "gain": 0.5,
                           "domain": "out"})
        assert isinstance(a, _Echo) and a.gain == 0.5
    finally:
        ADAPTERS.pop("_echo_test")


def test_build_by_import_path():
    a = build_adapter({"name": "x", "adapter": "fdbja.adapters.baselines:SilenceAdapter"})
    assert isinstance(a, SilenceAdapter)


def test_offline_and_unknown():
    assert build_adapter({"name": "x", "adapter": "offline"}) is None
    assert is_offline({"name": "x"}) and is_offline({"name": "x", "adapter": "offline"})
    with pytest.raises(ValueError, match="module.path:ClassName"):
        build_adapter({"name": "x", "adapter": "no_such_adapter"})
    with pytest.raises(ValueError, match="cannot import"):
        build_adapter({"name": "x", "adapter": "no_such_module_xyz:Foo"})
    with pytest.raises(ValueError):
        register_adapter("offline", _Echo)


def test_infer_skips_offline_and_unknown_then_scores_what_is_there(tmp_path):
    corpus = _load(os.path.join(ROOT, "examples", "dummy", "make_dummy_corpus.py"), "_mdc")
    _load(os.path.join(ROOT, "examples", "dummy", "energy_asr.py"), "_easr")
    corpus.make_dummy_corpus(str(tmp_path / "zoom1"), n_sessions=4, seed=0)
    with open(os.path.join(ROOT, "examples", "dummy", "config.yaml"), encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw["corpus"] = {"zoom1_root": str(tmp_path / "zoom1"), "eval_dir": str(tmp_path / "eval")}
    raw["detection"]["clips_per_task"] = 3
    raw["output"]["dir"] = str(tmp_path / "results")
    raw["output"]["bootstrap"] = {"n": 200}
    raw["models"] = [
        {"name": "silence_baseline", "adapter": "silence", "domain": "out"},
        {"name": "placed", "adapter": "offline", "domain": "out"},
        {"name": "broken", "adapter": "not_installed_here", "domain": "out"},
    ]
    config = Config(raw)
    reconstruct(config, log=quiet)
    msgs = []
    assert run_infer(config, log=msgs.append) == 1          # "broken" skipped, not raised
    assert any("[broken] skipped" in m for m in msgs)
    assert any("offline" in m and "0/3" in m for m in msgs)

    # place outputs for the offline entry by hand (here: copies of the silence run)
    import shutil
    for task in ("pause", "smooth_tt"):
        shutil.copytree(tmp_path / "eval" / f"{task}_silence_baseline_output",
                        tmp_path / "eval" / f"{task}_placed_output")
    run_transcribe(config, log=quiet)
    res = score_offline(config, log=quiet)
    models = {r["model"] for r in res["summary"]}
    assert models == {"silence_baseline", "placed"}
    only = score_offline(config, log=quiet, models=["placed"])
    assert {r["model"] for r in only["summary"]} == {"placed"}


def test_example_adapter_replies_after_silence():
    mod = _load(os.path.join(ROOT, "examples", "adapters", "reply_after_silence.py"), "_ras")
    sr = 16000
    x = np.zeros(sr * 6, dtype=np.float32)
    rng = np.random.default_rng(0)
    x[sr:3 * sr] = rng.normal(0, 0.1, 2 * sr)              # A speaks 1-3 s
    y = mod.ReplyAfterSilence(wait_s=0.6).infer(x, sr).response_audio
    on = np.flatnonzero(np.abs(y) > 0)
    assert len(on) and abs(on[0] / sr - 3.6) < 0.05         # starts after 0.6 s of silence
    assert not y[:int(3.5 * sr)].any()                      # silent while A speaks
    assert "reply_after_silence" in ADAPTERS
    a = build_adapter({"name": "r", "adapter": "reply_after_silence", "wait_s": 1.0})
    assert a.wait_s == 1.0
