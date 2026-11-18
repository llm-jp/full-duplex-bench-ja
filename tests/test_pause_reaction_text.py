"""What the system says in a Pause reaction: reference columns of analyze-profile."""
import numpy as np

from fdbja.metrics.pause_response import (
    DEFAULT_PAUSE_RESPONSE,
    PauseResponseConfig,
    judge_event,
    reaction_characters,
)
from fdbja.turn_profile import classify_reaction, event_rows, profile_rows, write_markdown

RATE = DEFAULT_PAUSE_RESPONSE.frame_rate


def _activity(seconds, spans):
    t = np.arange(int(seconds * RATE)) / RATE
    va = np.zeros(len(t), dtype=bool)
    for s, e in spans:
        va |= (t >= s - 1e-9) & (t < e - 1e-9)
    return va, t


def _chars(text, start, step=0.1):
    return [{"speaker": "B", "word": ch, "start": round(start + i * step, 3),
             "end": round(start + (i + 1) * step, 3)} for i, ch in enumerate(text)]


def test_reaction_runs_are_the_runs_that_count():
    # gap 4-6 s; 0.1 s blip (too short), a 0.4 s reaction, speech long after the gap
    va_s, t = _activity(12, [(4.1, 4.2), (4.6, 5.0), (9.0, 10.0)])
    va_a, _ = _activity(12, [(0.0, 4.0), (6.0, 12.0)])
    j = judge_event(va_s, va_a, t, 4.0, 6.0)
    assert (j["no_response"], j["interrupt"]) == (0, 0)
    assert j["reaction_runs"] == [[4.6, 5.0]]
    silent = judge_event(np.zeros(len(t), dtype=bool), va_a, t, 4.0, 6.0)
    assert silent["no_response"] == 1 and silent["reaction_runs"] == []


def test_reaction_characters_split():
    runs = [[4.6, 5.0]]
    assert reaction_characters(_chars("うん", 4.6), runs) == (2, 0)
    assert reaction_characters(_chars("そうですね", 4.6), runs) == (5, 0)
    n, r = reaction_characters(_chars("大学", 4.6), runs)
    assert (n, r) == (2, 2)
    # nothing transcribed near the reaction: a breath, noise or missed speech
    assert reaction_characters(_chars("そうですね", 8.0), runs) == (0, 0)
    assert reaction_characters([], runs) == (0, 0)


def test_reaction_characters_margin_uses_start_times():
    runs = [[4.6, 5.0]]
    late = _chars("うん", 5.15)                      # first character starts 0.15 s after the run
    assert reaction_characters(late, runs, margin=0.0) == (0, 0)
    assert reaction_characters(late, runs, margin=0.2) == (1, 1)   # only the first character
    # a character whose end was stretched far past the run still counts by its start
    stretched = [{"speaker": "B", "word": "は", "start": 4.7, "end": 4.8},
                 {"speaker": "B", "word": "い", "start": 4.8, "end": 6.6}]
    assert reaction_characters(stretched, runs, margin=0.0) == (2, 0)


def test_margin_is_a_config_key():
    assert PauseResponseConfig.from_config(
        {"enabled": True, "reaction_text_margin_seconds": 0.3}).reaction_text_margin_seconds == 0.3
    assert DEFAULT_PAUSE_RESPONSE.reaction_text_margin_seconds == 0.2


def _pause(model, i, no_response, interrupt, chars=None, residual=None, segs=()):
    return {"task": "pause", "model": model, "clip": f"c{i}", "event_index": 0, "TO": 0,
            "onset": None, "t_start": 4.0, "t_end": 6.0, "sys_segments": list(segs),
            "no_response": no_response, "interrupt": interrupt,
            "reaction_chars": chars, "reaction_residual_chars": residual}


def test_classify_reaction():
    assert classify_reaction(_pause("m", 0, 1, 0)) == ("none", None)
    assert classify_reaction(_pause("m", 0, 0, 1, 8, 5)) == ("barge_in", None)
    assert classify_reaction(_pause("m", 0, 1, 1)) == ("barge_in", None)
    assert classify_reaction(_pause("m", 0, 0, 0, 2, 0)) == ("brief", "backchannel")
    assert classify_reaction(_pause("m", 0, 0, 0, 4, 2)) == ("brief", "other")
    assert classify_reaction(_pause("m", 0, 0, 0, 0, 0)) == ("brief", "no_text")
    # results of an older score-offline: the rule without character counts, or no rule
    assert classify_reaction(_pause("m", 0, 0, 0)) == ("brief", None)
    assert classify_reaction({"t_start": 4.0, "t_end": 6.0}) == (None, None)


def test_profile_follows_the_reaction_rule(tmp_path):
    # "brief": reacts with a long backchannel-like utterance in every pause. The
    # transcript rule calls it talked through (8 characters), the audio rule does not.
    long_seg = [[4.5, 5.5, 8]]
    events = (
        [_pause("brief", i, 0, 0, 8, 0, long_seg) for i in range(6)]
        + [_pause("brief", i, 0, 0, 3, 3, long_seg) for i in range(6, 8)]
        + [_pause("brief", i, 0, 0, 0, 0, long_seg) for i in range(8, 10)]
        + [_pause("barge", i, 0, 1, 20, 18, long_seg) for i in range(6)]
        + [_pause("barge", i, 1, 0) for i in range(6, 10)]
        + [_pause("quiet", i, 1, 0) for i in range(10)]
    )
    results = {"events": events, "summary": []}
    ev = event_rows(results)
    assert {e["reaction"] for e in ev if e["model"] == "brief"} == {"brief"}
    prof = {r["model"]: r for r in profile_rows(ev, results)}

    b = prof["brief"]
    assert b["pause_takeover"] == 1.0                      # transcript rule
    assert (b["pause_no_reaction"], b["pause_barge_in"], b["pause_brief"]) == (0.0, 0.0, 1.0)
    assert b["pause_brief_n"] == 10
    assert (b["brief_backchannel"], b["brief_other"], b["brief_no_text"]) == (0.6, 0.2, 0.2)
    assert "overlapping" not in b["tendency"] and "unreactive" not in b["tendency"]

    g = prof["barge"]
    assert (g["pause_no_reaction"], g["pause_barge_in"], g["pause_brief"]) == (0.4, 0.6, 0.0)
    assert g["brief_backchannel"] is None and g["tendency"].startswith("overlapping")

    q = prof["quiet"]
    assert q["pause_no_reaction"] == 1.0 and q["tendency"] == "unreactive"

    rows = list(prof.values())
    assert len({tuple(r) for r in rows}) == 1              # same CSV columns for every system
    md = tmp_path / "profile.md"
    write_markdown(str(md), rows, {"takeover_hi": 0.5, "no_response_hi": 0.5, "slow_onset": 1.0,
                                   "barge_in_hi": 0.5, "no_reaction_hi": 0.5})
    text = md.read_text(encoding="utf-8")
    assert "Pause reaction (audio-based rule" in text and "| brief | 0% | 0% | 100% (10) | 60% |" in text


def test_profile_without_the_reaction_rule_keeps_the_transcript_tag():
    old = [{k: v for k, v in _pause("m", i, 0, 0, segs=[[4.5, 5.5, 8]]).items()
            if k not in ("no_response", "interrupt", "reaction_chars", "reaction_residual_chars")}
           for i in range(4)]
    results = {"events": old, "summary": []}
    (row,) = profile_rows(event_rows(results), results)
    assert row["pause_barge_in"] is None and row["pause_brief"] is None
    assert row["tendency"] == "overlapping"                # falls back to talked through
