"""`fdbja check-responses`: the check of what `infer` wrote, on hand-made
reception records (no network, no model)."""

import csv
import json
import os

import numpy as np
import soundfile as sf

from fdbja import response_check as rc
from fdbja.adapters._timeline import RECEIVE_END, ResponseTimeline
from fdbja.adapters.base import DialogueResponse
from fdbja.cli import main
from fdbja.config import Config
from fdbja.runner import _write_outputs

SR = 24000


def _tone(seconds, value=1000):
    return np.full(int(seconds * SR), value, dtype=np.int16)


def _record(tl, rule="playout", seconds=2.0, end=None, reason="deadline"):
    if end is not None:
        tl.note(end, RECEIVE_END, reason=reason)
    return tl.record("legacy" if rule != "playout" else "playout", rule,
                     min_samples=int(seconds * SR))


def _save(path, tl, rule="playout", seconds=2.0, end=None, reason="deadline"):
    """Write a clip directory the way ``infer`` does. Returns the record."""
    record = _record(tl, rule, seconds, end, reason)
    out = tl.render(rule, int(seconds * SR))
    resp = DialogueResponse(
        input_audio=np.zeros(int(seconds * SR), dtype=np.float32),
        response_audio=out.astype(np.float32) / 32768.0, sample_rate=SR, events=record,
        received_audio=tl.received_audio().astype(np.float32) / 32768.0)
    _write_outputs(str(path), resp, sf)
    return record


# --- what a record says ------------------------------------------------------

def test_interruption_counts_and_discarded_audio():
    tl = ResponseTimeline(SR, adapter="x")
    tl.add_audio(0.1, _tone(1.0), item="a")         # 1 s arrives at once
    tl.interrupt(0.4, source="notice")              # heard for 0.3 s
    tl.add_audio(0.45, _tone(0.2), item="a")        # still in flight: not played
    tl.add_audio(0.9, _tone(0.1), item="b")
    facts = rc.replay(_record(tl))
    assert facts["chunks"] == 3 and facts["interruptions"] == 1
    assert facts["chunks_after_interruption"] == 1
    assert abs(facts["received_seconds"] - 1.3) < 1e-6
    assert abs(facts["placed_seconds"] - 0.4) < 1e-3
    assert abs(facts["discarded_seconds"] - 0.9) < 1e-3
    assert abs(facts["first_audio"] - 0.1) < 1e-3
    assert abs(facts["last_audio_end"] - 1.0) < 1e-3
    assert facts["stalls"] == 0             # the pause between two replies is not a stall
    # the same record without the interruption handling keeps everything
    kept = rc.replay({**_record(tl), "rule": "arrival"})
    assert kept["discarded_seconds"] == 0 and kept["interruptions"] == 1


def test_stall_inside_one_response_is_counted():
    tl = ResponseTimeline(SR)
    tl.add_audio(0.1, _tone(0.2), item="a")
    tl.add_audio(0.6, _tone(0.2), item="a")         # the queue ran dry for 0.3 s
    facts = rc.replay(_record(tl))
    assert facts["stalls"] == 1
    assert abs(facts["stall_seconds"] - 0.3) < 1e-3
    assert abs(facts["longest_stall"] - 0.3) < 1e-3


def test_continuous_stream_that_falls_behind_real_time():
    # 80 ms frames without an item id; two of them arrive late
    tl = ResponseTimeline(SR)
    for t in (0.10, 0.18, 0.26, 0.50, 0.58, 0.80):
        tl.add_audio(t, _tone(0.08))
    facts = rc.replay(_record(tl))
    assert facts["stalls"] == 2
    assert abs(facts["stall_seconds"] - (0.16 + 0.14)) < 1e-3
    # joined in order (the construction up to 0.2.0) the lateness is not on the timeline
    assert rc.replay(_record(tl, rule="concat"))["stalls"] == 0


def test_server_event_between_chunks_without_item_starts_a_new_response():
    tl = ResponseTimeline(SR)
    tl.add_audio(0.1, _tone(0.2))
    tl.note(0.35, "server_content.turn_complete")
    tl.add_audio(1.0, _tone(0.2))
    assert rc.replay(_record(tl))["stalls"] == 0


def test_tail_beyond_the_written_length_and_failed_responses():
    tl = ResponseTimeline(SR)
    tl.add_audio(0.5, _tone(1.0), item="a")
    tl.note(0.6, "response.done", status="failed")
    tl.note(0.7, "error", code="x")
    record = tl.record("legacy", "arrival", min_samples=SR, max_samples=SR)
    facts = rc.replay(record)
    assert abs(facts["cut_seconds"] - 0.5) < 1e-3
    assert facts["failed_responses"] == 1 and facts["error_notes"] == 1


# --- one clip -----------------------------------------------------------------

def test_clean_clip_has_no_findings_and_rebuilds(tmp_path):
    tl = ResponseTimeline(SR, adapter="x")
    tl.add_audio(0.1, _tone(0.5), item="a")
    tl.interrupt(0.3, source="notice")
    combined = tmp_path / "c" / "combined.wav"
    _save(combined, tl, end=2.4)
    row = rc.check_clip(str(combined), input_seconds=2.0)
    assert row["findings"] == [] and row["rebuild"] == "same"
    assert row["no_audio"] is False and row["rule"] == "playout"
    assert row["receive_end"] == 2.4 and row["receive_end_reason"] == "deadline"


def test_listening_that_ended_before_the_input(tmp_path):
    tl = ResponseTimeline(SR)
    tl.add_audio(0.1, _tone(0.2))
    combined = tmp_path / "c" / "combined.wav"
    _save(combined, tl, end=1.2, reason="closed")
    assert rc.check_clip(str(combined), input_seconds=2.0)["findings"] == ["ended_early"]
    assert rc.check_clip(str(combined), input_seconds=1.2)["findings"] == []
    # without the input length the end of listening is not judged
    assert rc.check_clip(str(combined))["findings"] == []


def test_files_that_do_not_match_the_record(tmp_path):
    tl = ResponseTimeline(SR)
    tl.add_audio(0.1, _tone(0.5), item="a")
    combined = tmp_path / "c" / "combined.wav"
    _save(combined, tl)
    d = combined.parent
    saved, _ = sf.read(str(d / "output.wav"), dtype="int16")
    sf.write(str(d / "output.wav"), np.roll(saved, 2400), SR, subtype="PCM_16")
    row = rc.check_clip(str(combined))
    assert row["findings"] == ["rebuild_differs"] and row["rebuild"] == "differs"
    # with --records-only the audio is not opened
    assert rc.check_clip(str(combined), records_only=True)["findings"] == []
    received, _ = sf.read(str(d / "received.wav"), dtype="int16")
    sf.write(str(d / "received.wav"), received[:-100], SR, subtype="PCM_16")
    assert "received_mismatch" in rc.check_clip(str(combined))["findings"]


def test_missing_output_missing_record_and_unreadable_record(tmp_path):
    assert rc.check_clip(str(tmp_path / "none" / "combined.wav"))["findings"] == [
        "missing_output"]
    resp = DialogueResponse(np.zeros(SR, dtype=np.float32), np.zeros(SR, dtype=np.float32), SR)
    combined = tmp_path / "c" / "combined.wav"
    _write_outputs(str(combined), resp, sf)
    row = rc.check_clip(str(combined))
    assert row["findings"] == ["no_record"] and row["no_audio"] is True
    (combined.parent / "events.json").write_text('{"format": "something else"}')
    assert rc.check_clip(str(combined))["findings"] == ["bad_record"]


def test_stall_limit(tmp_path):
    tl = ResponseTimeline(SR)
    tl.add_audio(0.1, _tone(0.2), item="a")
    tl.add_audio(0.6, _tone(0.2), item="a")
    combined = tmp_path / "c" / "combined.wav"
    _save(combined, tl)
    assert rc.check_clip(str(combined))["findings"] == ["stall"]
    assert rc.check_clip(str(combined), stall_limit=0.5)["findings"] == []


# --- a whole run --------------------------------------------------------------

def _eval_dir(tmp_path, clips=("c1", "c2", "c3")):
    task_dir = tmp_path / "eval" / "pause"
    task_dir.mkdir(parents=True)
    for clip in clips:
        (task_dir / f"{clip}.json").write_text("[]")
        sf.write(str(task_dir / f"{clip}.wav"), np.zeros((2 * 16000, 2), dtype=np.float32),
                 16000)
    return tmp_path / "eval"


def _config(eval_dir, models):
    return Config({"corpus": {"eval_dir": str(eval_dir)}, "tasks": ["pause"],
                   "models": models})


def _speaking(t=0.1):
    tl = ResponseTimeline(SR, adapter="x")
    tl.add_audio(t, _tone(0.5), item="a")
    return tl


def test_run_reports_per_system_and_writes_the_files(tmp_path):
    eval_dir = _eval_dir(tmp_path)
    out = eval_dir / "pause_good_output"
    for clip in ("c1", "c2", "c3"):
        _save(out / clip / "combined.wav", _speaking(), end=2.5)
    # "mixed": one clip under the old rule, one that stopped early, one missing
    out = eval_dir / "pause_mixed_output"
    _save(out / "c1" / "combined.wav", _speaking(), end=2.5)
    _save(out / "c2" / "combined.wav", _speaking(), rule="arrival", end=1.0, reason="closed")
    # "empty": every clip without any audio; "byhand": placed without a record
    for clip in ("c1", "c2", "c3"):
        _save(eval_dir / "pause_empty_output" / clip / "combined.wav",
              ResponseTimeline(SR, adapter="x"), end=2.5)
        resp = DialogueResponse(np.zeros(SR, dtype=np.float32),
                                np.full(SR, 0.1, dtype=np.float32), SR)
        _write_outputs(str(eval_dir / "pause_byhand_output" / clip / "combined.wav"), resp, sf)
    config = _config(eval_dir, [
        {"name": "good", "adapter": "moshi_ws"}, {"name": "mixed", "adapter": "moshi_ws"},
        {"name": "empty", "adapter": "moshi_ws"}, {"name": "byhand", "adapter": "offline"},
        {"name": "floor", "adapter": "silence"}])
    msgs = []
    summary = rc.run(config, str(tmp_path / "check"), log=msgs.append)
    found = {(f["model"], f["finding"]): f for f in summary["findings"]}
    assert set(found) == {
        ("mixed", "missing_output"), ("mixed", "ended_early"), ("mixed", "mixed_rules"),
        ("empty", "all_no_audio"), ("byhand", "no_record")}
    assert found[("mixed", "missing_output")]["examples"] == ["c3"]
    assert found[("mixed", "ended_early")]["examples"] == ["c2"]
    assert found[("byhand", "no_record")]["clips"] == 3
    assert summary["errors"] == 4 and summary["warnings"] == 1
    assert any("floor: trivial baseline" in m for m in msgs)
    systems = {s["model"]: s for s in summary["systems"]}
    assert "floor" not in systems
    assert systems["good"]["records"] == 3 and systems["good"]["errors"] == 0
    assert systems["good"]["rules"] == "playout" and systems["good"]["first_audio_median"] == 0.1
    assert systems["mixed"]["rules"] == "arrival/playout" and systems["mixed"]["outputs"] == 2
    assert systems["empty"]["no_audio"] == 3 and systems["byhand"]["no_audio"] == 0
    with open(tmp_path / "check" / "clips.csv", encoding="utf-8") as f:
        clips = list(csv.DictReader(f))
    assert len(clips) == 12
    assert {r["rebuild"] for r in clips if r["model"] == "good"} == {"same"}
    report = (tmp_path / "check" / "response_check.md").read_text(encoding="utf-8")
    assert "`ended_early`" in report and "| good | pause | 3 | 3 | 3 | playout |" in report
    assert os.path.exists(tmp_path / "check" / "systems.csv")


def test_systems_recorded_under_different_rules_are_reported(tmp_path):
    eval_dir = _eval_dir(tmp_path, clips=("c1",))
    for name, rule in (("a", "playout"), ("b", "playout"), ("c", "concat")):
        _save(eval_dir / f"pause_{name}_output" / "c1" / "combined.wav", _speaking(),
              rule=rule, end=2.5)
    config = _config(eval_dir, [{"name": n, "adapter": "moshi_ws"} for n in "abc"])
    summary = rc.run(config, str(tmp_path / "check"), log=lambda m: None)
    (finding,) = summary["findings"]
    assert finding["finding"] == "rules_differ" and finding["model"] == "c"
    assert finding["level"] == "warning" and "'concat'" in finding["meaning"]
    # restricted to the systems that share a rule, nothing is reported
    summary = rc.run(config, str(tmp_path / "check2"), models=["a", "b"], log=lambda m: None)
    assert summary["findings"] == []


def test_cli_exit_status(tmp_path, capsys):
    eval_dir = _eval_dir(tmp_path, clips=("c1",))
    _save(eval_dir / "pause_m_output" / "c1" / "combined.wav", _speaking(), end=1.0)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(json.dumps({"corpus": {"eval_dir": str(eval_dir)}, "tasks": ["pause"],
                               "models": [{"name": "m", "adapter": "moshi_ws"}]}))
    args = ["check-responses", "--config", str(cfg), "--out", str(tmp_path / "check")]
    assert main(args) == 0
    assert main(args + ["--strict"]) == 1
    assert "ended_early" in capsys.readouterr().out
    assert main(args + ["--strict", "--records-only"]) == 1


def test_session_settings_are_reported_and_have_to_be_the_same(tmp_path):
    from fdbja.adapters._timeline import SESSION

    def speaking(voice):
        tl = ResponseTimeline(SR, adapter="x")
        tl.note(0.0, SESSION, model="m-1", voice=voice, instructions="abc123")
        tl.add_audio(0.1, _tone(0.5), item="a")
        return tl

    eval_dir = _eval_dir(tmp_path)
    for name, voices in (("same", "aaa"), ("changed", "aab")):
        for clip, voice in zip(("c1", "c2", "c3"), voices):
            _save(eval_dir / f"pause_{name}_output" / clip / "combined.wav",
                  speaking({"a": "alto", "b": "bass"}[voice]), end=2.5)
    config = _config(eval_dir, [{"name": n, "adapter": "x"} for n in ("same", "changed")])
    summary = rc.run(config, str(tmp_path / "check"), log=lambda m: None)
    (finding,) = summary["findings"]
    assert finding["finding"] == "mixed_settings" and finding["level"] == "error"
    assert finding["model"] == "changed" and finding["examples"] == ["pause/c3"]
    systems = {s["model"]: s for s in summary["systems"]}
    assert systems["same"]["settings"] == "instructions=abc123, model=m-1, voice=alto"
    assert " | " in systems["changed"]["settings"]
    report = (tmp_path / "check" / "response_check.md").read_text(encoding="utf-8")
    assert "| same | pause | instructions=abc123, model=m-1, voice=alto |" in report


def test_version_and_day_of_recording_are_reported(tmp_path):
    def speaking(version, at):
        tl = ResponseTimeline(SR, adapter="x")
        tl.add_audio(0.1, _tone(0.5), item="a")
        return tl, version, at

    eval_dir = _eval_dir(tmp_path)
    plan = {"same": [("0.3.0", "2026-12-01T10:00:00Z")] * 3,
            "changed": [("0.3.0", "2026-12-01T23:59:00Z"), ("0.3.0", "2026-12-02T00:01:00Z"),
                        ("0.4.0", "2026-12-20T00:00:00Z")]}
    for name, clips in plan.items():
        for clip, (version, at) in zip(("c1", "c2", "c3"), clips):
            d = eval_dir / f"pause_{name}_output" / clip
            tl, _, _ = speaking(version, at)
            _save(d / "combined.wav", tl, end=2.5)
            record = json.loads((d / "events.json").read_text(encoding="utf-8"))
            record.update(fdbja=version, recorded_at=at)
            (d / "events.json").write_text(json.dumps(record), encoding="utf-8")
    config = _config(eval_dir, [{"name": n, "adapter": "x"} for n in plan])
    summary = rc.run(config, str(tmp_path / "check"), log=lambda m: None)
    (finding,) = summary["findings"]
    assert finding["finding"] == "mixed_versions" and finding["level"] == "warning"
    assert finding["model"] == "changed" and finding["examples"] == ["c3"]
    systems = {s["model"]: s for s in summary["systems"]}
    assert systems["same"]["fdbja"] == "0.3.0" and systems["same"]["recorded"] == "2026-12-01"
    assert systems["changed"]["fdbja"] == "0.3.0/0.4.0"
    assert systems["changed"]["recorded"] == "2026-12-01..2026-12-20"
    report = (tmp_path / "check" / "response_check.md").read_text(encoding="utf-8")
    assert "| same | pause | 0.3.0 | 2026-12-01 |" in report
