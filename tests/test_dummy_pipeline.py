"""End-to-end smoke test on the synthetic corpus in ``examples/dummy/``.

Runs reconstruct -> infer (trivial baselines) -> transcribe (energy stand-in
ASR) -> score-offline without any corpus, model server or speech model, so the
wiring of the four stages is checked in CI. VAP-distance stays disabled.
"""

import importlib.util
import os

import yaml

from fdbja.config import Config
from fdbja.data.reconstruct import reconstruct
from fdbja.runner import run_infer, run_transcribe, score_offline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DUMMY = os.path.join(ROOT, "examples", "dummy")


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"_dummy_{name}", os.path.join(DUMMY, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_dummy_pipeline_end_to_end(tmp_path):
    corpus = _load("make_dummy_corpus")
    _load("energy_asr")                     # registers the "energy" backend
    corpus.make_dummy_corpus(str(tmp_path / "zoom1"), n_sessions=4, seed=0)

    with open(os.path.join(DUMMY, "config.yaml"), encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw["corpus"] = {"zoom1_root": str(tmp_path / "zoom1"),
                     "eval_dir": str(tmp_path / "eval")}
    raw["detection"]["clips_per_task"] = 3
    raw["output"]["dir"] = str(tmp_path / "results")
    config = Config(raw)
    quiet = lambda *a, **k: None  # noqa: E731

    reconstruct(config, log=quiet)
    run_infer(config, log=quiet)
    assert run_transcribe(config, log=quiet) == 2 * 2 * 3   # models x tasks x clips
    results = score_offline(config, log=quiet)

    tor = {(r["task"], r["model"]): r["TOR"] for r in results["summary"]}
    assert set(tor) == {(t, m) for t in ("pause", "smooth_tt")
                        for m in ("silence_baseline", "always_talk_baseline")}
    # silence never takes the floor; continuous noise always overlaps a pause
    assert tor[("pause", "silence_baseline")] == 0.0
    assert tor[("smooth_tt", "silence_baseline")] == 0.0
    assert tor[("pause", "always_talk_baseline")] == 1.0
    assert all(r["n"] > 0 for r in results["summary"])
