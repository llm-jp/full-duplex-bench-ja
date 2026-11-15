"""``fdbja leaderboard``: the ranks of the leaderboard page from a summary."""

import json

import pytest

from fdbja import report
from fdbja.cli import main


def _res(rows, **cond):
    """Rows are (task, model, TOR, domain) plus, for Pause rows scored with the
    reaction penalty, (no-reaction rate, barge-in rate)."""
    out = []
    for t, m, tor, d, *pen in rows:
        r = {"task": t, "model": m, "desired_TO": 1 if t == "smooth_tt" else 0,
             "TOR": tor, "domain": d}
        if pen:
            r.update(pause_response=sum(pen), pause_no_response_rate=pen[0],
                     pause_interrupt_rate=pen[1])
        out.append(r)
    return {"summary": out, **cond}


# a: reacts less in pauses (worse on Pause) but takes more turns (better on Smooth TT)
ROWS = [("pause", "a", 0.2, "out", 0.5, 0.1), ("smooth_tt", "a", 0.9, "out"),
        ("pause", "b", 0.5, "out", 0.2, 0.1), ("smooth_tt", "b", 0.6, "out"),
        ("pause", "baseline_silence", 0.0, "baseline", 1.0, 0.0),
        ("smooth_tt", "baseline_silence", 0.0, "baseline")]
NO_PENALTY = [r[:4] for r in ROWS]


def _ranked(text):
    """(Pause rank, model, Smooth TT rank) of the table rows, in print order."""
    body = text.split("\n\n")[0].splitlines()[2:]
    return [(ln.split()[0], ln.split()[1], ln.split()[-2]) for ln in body]


def test_desired_rate():
    assert report.desired_rate({"TOR": 0.25, "desired_TO": 0}) == 0.75
    assert report.desired_rate({"TOR": 0.25, "desired_TO": 1}) == 0.25
    assert report.desired_rate({"TOR": None, "desired_TO": 1}) is None


def test_rows_use_each_tasks_metric():
    rows = {m["model"]: m for m in report.leaderboard_rows(_res(ROWS))}
    assert rows["a"]["pause"] == pytest.approx(0.6) and rows["a"]["smooth_tt"] == 0.9
    assert rows["a"]["pause_no_response"] == 0.5 and rows["a"]["pause_interrupt"] == 0.1
    assert rows["a"]["pause_tor"] == 0.2          # kept as a reference, not ranked
    assert "overall" not in rows["a"]             # no combined score
    assert rows["baseline_silence"]["baseline"]


def test_ranks_per_task_and_baselines_last():
    text = report.format_leaderboard(_res(ROWS))
    # Pause: smaller penalty first; Smooth TT: larger TOR first; baseline unranked
    assert _ranked(text) == [("1", "b", "2"), ("2", "a", "1"), ("-", "baseline_silence", "-")]
    assert "NOTE" not in text and "no combined rank" in text


def test_equal_values_share_a_rank():
    rows = ROWS[:2] + [("pause", "b", 0.5, "out", 0.1, 0.5), ("smooth_tt", "b", 0.9, "out")]
    assert [(p, s) for p, _, s in _ranked(report.format_leaderboard(_res(rows)))] == [("1", "1")] * 2


def test_summary_without_penalty():
    # scored with metrics.pause_response disabled (or without the output audio)
    text = report.format_leaderboard(_res(NO_PENALTY))
    assert _ranked(text)[:2] == [("-", "a", "1"), ("-", "b", "2")]
    assert "NOTE: the summary has no pause reaction penalty" in text


def test_reference_rows_and_condition_warning():
    ref = _res([("pause", "c", 0.1, "out", 0.1, 0.0), ("smooth_tt", "c", 0.95, "out")],
               boundaries="audio", asr="x")
    text = report.format_leaderboard(_res(ROWS, boundaries="transcript", asr="x"), ref)
    assert _ranked(text)[0] == ("1", "c", "1") and "c [ref]" in text
    assert "boundaries: 'transcript' vs 'audio'" in text and "asr:" not in text
    same = report.format_leaderboard(_res(ROWS, boundaries="audio"), ref)
    assert "WARNING" not in same


def test_condition_with_description():
    # docs/results.json appends a description to the value
    ref = {"boundaries": "audio (human-side VAD; detection.boundaries: audio). Same events.",
           "smooth_tt_roles": False}
    assert report.condition_mismatches({"boundaries": "audio", "smooth_tt_roles": False}, ref) == []
    diff = report.condition_mismatches({"boundaries": "transcript", "smooth_tt_roles": True}, ref)
    assert len(diff) == 2 and diff[0].startswith("boundaries: 'transcript' vs 'audio (")


def test_cli(tmp_path, capsys):
    p = tmp_path / "summary.json"
    p.write_text(json.dumps(_res(ROWS)), encoding="utf-8")
    assert main(["leaderboard", "--results", str(p), "--reference", "docs/results.json"]) == 0
    out = capsys.readouterr().out
    assert "a" in out and "[ref]" in out
