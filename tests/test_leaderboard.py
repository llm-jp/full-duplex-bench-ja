"""``fdbja leaderboard``: the ranking of the leaderboard page from a summary."""

import json

from fdbja import report
from fdbja.cli import main


def _res(rows, **cond):
    return {"summary": [{"task": t, "model": m, "desired_TO": 1 if t == "smooth_tt" else 0,
                         "TOR": tor, "domain": d} for t, m, tor, d in rows], **cond}


ROWS = [("pause", "a", 0.2, "out"), ("smooth_tt", "a", 0.6, "out"),
        ("pause", "b", 0.5, "out"), ("smooth_tt", "b", 0.9, "out"),
        ("pause", "baseline_silence", 0.0, "baseline"), ("smooth_tt", "baseline_silence", 0.0, "baseline")]


def test_desired_rate():
    assert report.desired_rate({"TOR": 0.25, "desired_TO": 0}) == 0.75
    assert report.desired_rate({"TOR": 0.25, "desired_TO": 1}) == 0.25
    assert report.desired_rate({"TOR": None, "desired_TO": 1}) is None


def test_ranking_and_baselines_last():
    rows = {m["model"]: m for m in report.leaderboard_rows(_res(ROWS))}
    assert rows["a"]["overall"] == 0.7 and rows["b"]["overall"] == 0.7
    assert rows["baseline_silence"]["baseline"]
    text = report.format_leaderboard(_res(ROWS))
    lines = [ln for ln in text.splitlines() if ln.strip() and ln.split()[0].isdigit()]
    assert [ln.split()[1] for ln in lines] == ["a", "b"]
    assert "baseline_silence" in text.splitlines()[4]


def test_reference_rows_and_condition_warning():
    ref = _res([("pause", "c", 0.1, "out"), ("smooth_tt", "c", 0.9, "out")],
               boundaries="audio", asr="x")
    text = report.format_leaderboard(_res(ROWS, boundaries="transcript", asr="x"), ref)
    assert "c [ref]" in text
    assert "boundaries: 'transcript' vs 'audio'" in text and "asr:" not in text
    same = report.format_leaderboard(_res(ROWS, boundaries="audio"), ref)
    assert "WARNING" not in same


def test_cli(tmp_path, capsys):
    p = tmp_path / "summary.json"
    p.write_text(json.dumps(_res(ROWS)), encoding="utf-8")
    assert main(["leaderboard", "--results", str(p), "--reference", "docs/results.json"]) == 0
    out = capsys.readouterr().out
    assert "a" in out and "[ref]" in out
