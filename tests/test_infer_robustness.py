"""``infer`` over many clips: a failed clip does not end the run, the clip is
retried by the next run, and a clip is never left half written."""

import importlib.util
import os

import numpy as np
import pytest
import soundfile as sf
import yaml

from fdbja import runner
from fdbja.adapters import ADAPTERS, register_adapter
from fdbja.adapters.base import DialogueResponse, ModelAdapter
from fdbja.config import Config
from fdbja.data.reconstruct import reconstruct

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
quiet = lambda *a, **k: None  # noqa: E731


class _Flaky(ModelAdapter):
    """Echoes the input; raises on the calls listed in ``fail`` (1-based)."""
    calls = 0
    fail: set = set()

    def infer(self, input_audio, sample_rate):
        type(self).calls += 1
        if type(self).calls in type(self).fail or "all" in type(self).fail:
            raise ConnectionError("handshake failed: wss://example.test/live?key=SECRET123&alt=1")
        return DialogueResponse(input_audio, input_audio, sample_rate)


@pytest.fixture
def config(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "_mdc", os.path.join(ROOT, "examples", "dummy", "make_dummy_corpus.py"))
    corpus = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(corpus)
    corpus.make_dummy_corpus(str(tmp_path / "zoom1"), n_sessions=4, seed=0)
    with open(os.path.join(ROOT, "examples", "dummy", "config.yaml"), encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw["corpus"] = {"zoom1_root": str(tmp_path / "zoom1"), "eval_dir": str(tmp_path / "eval")}
    raw["detection"]["clips_per_task"] = 3
    raw["models"] = [{"name": "flaky", "adapter": "_flaky_test", "domain": "out"}]
    cfg = Config(raw)
    reconstruct(cfg, log=quiet)
    register_adapter("_flaky_test", _Flaky)
    _Flaky.calls, _Flaky.fail = 0, set()
    yield cfg
    ADAPTERS.pop("_flaky_test")


def _written(cfg):
    out = []
    for task in cfg.tasks:
        d = os.path.join(cfg.eval_dir, f"{task}_flaky_output")
        if os.path.isdir(d):
            out += [f"{task}/{c}" for c in sorted(os.listdir(d))
                    if os.path.exists(os.path.join(d, c, "combined.wav"))]
    return out


def test_failed_clip_is_reported_left_out_and_retried(config):
    n_clips = 3 * len(config.tasks)
    _Flaky.fail = {2}
    msgs = []
    assert runner.run_infer(config, log=msgs.append) == 1
    assert _Flaky.calls == n_clips                      # the run went on after the failure
    assert len(_written(config)) == n_clips - 1
    (line,) = [m for m in msgs if "failed, no output written" in m]
    assert "ConnectionError" in line and "SECRET123" not in line and "key=***" in line
    assert any("1 failed (run infer again to retry them)" in m for m in msgs)
    # the next run asks the model only for the clip that is missing
    _Flaky.calls, _Flaky.fail = 0, set()
    assert runner.run_infer(config, log=quiet) == 0
    assert _Flaky.calls == 1 and len(_written(config)) == n_clips


def test_model_is_abandoned_after_failures_in_a_row(config, monkeypatch):
    monkeypatch.setattr(runner, "MAX_CONSECUTIVE_FAILURES", 4)
    _Flaky.fail = {"all"}
    msgs = []
    assert runner.run_infer(config, log=msgs.append) == 1
    assert _Flaky.calls == 4 and _written(config) == []
    assert any("stopped: 4 clips failed in a row" in m for m in msgs)


def test_interrupted_write_leaves_no_finished_clip(tmp_path, monkeypatch):
    resp = DialogueResponse(np.zeros(1600, dtype=np.float32),
                            np.zeros(1600, dtype=np.float32), 16000)
    path = tmp_path / "clip" / "combined.wav"

    def killed(src, dst):
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "replace", killed)
    with pytest.raises(KeyboardInterrupt):
        runner._write_outputs(str(path), resp, sf)
    assert not path.exists()                    # infer will record this clip again
    monkeypatch.undo()
    runner._write_outputs(str(path), resp, sf)
    assert path.exists() and not (tmp_path / "clip" / "combined.wav.part").exists()
    audio, sr = sf.read(str(path))
    assert audio.shape == (1600, 2) and sr == 16000


def test_error_text_masks_credentials():
    text = runner._error_text(RuntimeError("401 for url ...?api_key=abc123\nAuthorization: Bearer xyz789"))
    assert "abc123" not in text and "xyz789" not in text and "\n" not in text
    assert text.startswith("RuntimeError: 401")
