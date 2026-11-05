"""Building an evaluation set from your own two-channel dialogues:
audio only -> transcribe --channel 0 1 --speaker A B -> reconstruct --layout
generic -> infer -> transcribe -> score-offline (summary with intervals,
timing and scoring conditions)."""

import csv
import importlib.util
import json
import os

import yaml

from fdbja.cli import main as cli_main
from fdbja.config import Config
from fdbja.data.reconstruct import check_speakers, reconstruct
from fdbja.runner import run_infer, run_transcribe, score_offline
from fdbja.summary import write_summary

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DUMMY = os.path.join(ROOT, "examples", "dummy")
quiet = lambda *a, **k: None  # noqa: E731


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"_cc_{name}", os.path.join(DUMMY, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _config(tmp_path, **corpus):
    with open(os.path.join(DUMMY, "config.yaml"), encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw["corpus"] = {"eval_dir": str(tmp_path / "eval"), **corpus}
    raw["detection"]["clips_per_task"] = 3
    raw["output"]["dir"] = str(tmp_path / "results")
    raw["output"]["bootstrap"] = {"n": 500}
    return raw


def test_check_speakers():
    assert check_speakers([{"speaker": "A"}, {"speaker": "B"}]) == []
    assert check_speakers([{"speaker": "B"}])
    assert check_speakers([{"speaker": "A"}, {"speaker": "B"}, {"speaker": "C"}])


def test_own_dialogues_end_to_end(tmp_path):
    corpus = _load("make_dummy_corpus")
    _load("energy_asr")
    ids = corpus.make_dummy_corpus(str(tmp_path / "mine"), n_sessions=4, seed=0,
                                   layout="generic", audio_only=True)
    assert not (tmp_path / "mine" / "text").exists()

    # 1. transcripts with speaker labels (ch0 = A, ch1 = B), via the CLI
    wavs = [str(tmp_path / "mine" / "audio" / f"{i}.wav") for i in ids]
    assert cli_main(["transcribe", "--backend", "energy", "--wav", *wavs,
                     "--channel", "0", "1", "--speaker", "A", "B",
                     "--out", str(tmp_path / "mine" / "text")]) == 0
    with open(tmp_path / "mine" / "text" / f"{ids[0]}.json", encoding="utf-8") as f:
        jchat = json.load(f)
    assert {c["speaker"] for c in jchat} == {"A", "B"}
    assert [c["start"] for c in jchat] == sorted(c["start"] for c in jchat)

    # 2. evaluation set from the generic layout
    raw = _config(tmp_path, text_dir=str(tmp_path / "mine" / "text"),
                  audio_dir=str(tmp_path / "mine" / "audio"), layout="generic")
    config = Config(raw)
    reconstruct(config, log=quiet)
    for task in ("pause", "smooth_tt"):
        assert len(list((tmp_path / "eval" / task).glob("*.wav"))) == 3

    # 3. models -> transcripts -> scores
    run_infer(config, log=quiet)
    run_transcribe(config, log=quiet)
    res = score_offline(config, log=quiet)
    tor = {(r["task"], r["model"]): r for r in res["summary"]}
    assert tor[("pause", "silence_baseline")]["TOR"] == 0.0
    assert tor[("pause", "always_talk_baseline")]["TOR"] == 1.0
    for r in res["summary"]:
        assert r["n_clips"] == 3 and r["n"] >= 3
        lo, hi = r["TOR_ci"]
        assert lo <= r["TOR"] <= hi
        assert "lat_median" in r
    assert res["asr"] == "energy"
    assert res["boundaries"] == "transcript" and res["system_activity"] == "transcript"
    assert res["ci"] == "95% bootstrap intervals over events"
    assert res["vap_checkpoint"] is None

    paths = write_summary(res, str(tmp_path / "results"))
    with open(paths[0], encoding="utf-8") as f:
        js = json.load(f)
    assert "events" not in js and js["summary"] and js["asr"] == "energy"
    with open(paths[1], encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert {"TOR_ci_low", "TOR_ci_high", "n_clips", "asr", "boundaries"} <= set(rows[0])


def test_transcribe_default_speaker_warns_in_reconstruct(tmp_path):
    corpus = _load("make_dummy_corpus")
    _load("energy_asr")
    ids = corpus.make_dummy_corpus(str(tmp_path / "mine"), n_sessions=2, seed=0,
                                   layout="generic", audio_only=True)
    wavs = [str(tmp_path / "mine" / "audio" / f"{i}.wav") for i in ids]
    # without --speaker every character is "B" (the model-response default)
    cli_main(["transcribe", "--backend", "energy", "--wav", *wavs,
              "--out", str(tmp_path / "mine" / "text")])
    msgs = []
    raw = _config(tmp_path, text_dir=str(tmp_path / "mine" / "text"),
                  audio_dir=str(tmp_path / "mine" / "audio"))
    reconstruct(Config(raw), log=msgs.append, layout="generic")
    assert any("--speaker A B" in m for m in msgs)


def test_analyze_profile_matches_tor(tmp_path):
    from fdbja import report
    from fdbja.turn_profile import run as run_profile

    corpus = _load("make_dummy_corpus")
    _load("energy_asr")
    corpus.make_dummy_corpus(str(tmp_path / "zoom1"), n_sessions=4, seed=0)
    raw = _config(tmp_path, zoom1_root=str(tmp_path / "zoom1"))
    config = Config(raw)
    reconstruct(config, log=quiet)
    run_infer(config, log=quiet)
    run_transcribe(config, log=quiet)
    res = score_offline(config, log=quiet)
    path = report.write_json(res, str(tmp_path / "results"))
    out = run_profile(path, str(tmp_path / "profile"), log=quiet)
    # outcomes agree with the TO decisions of score-offline
    for e in out["events"]:
        took = e["outcome"] in ("takeover", "taken", "overlap")
        assert took == bool(e["TO"]), e
        assert e["window_start"] <= e["window_end"]
    prof = {r["model"]: r for r in out["profile"]}
    tor = {(r["task"], r["model"]): r["TOR"] for r in res["summary"]}
    for m, r in prof.items():
        assert abs(r["pause_takeover"] - tor[("pause", m)]) < 1e-3
        assert abs(r["stt_taken"] - tor[("smooth_tt", m)]) < 1e-3
        assert abs(r["pause_takeover"] + r["pause_short"] + r["pause_silent"] - 1) < 1e-3
    assert prof["silence_baseline"]["pause_silent"] == 1.0
    assert prof["always_talk_baseline"]["tendency"].startswith("overlapping")
    for name in ("events.csv", "profile.csv", "profile.md"):
        assert (tmp_path / "profile" / name).exists()
