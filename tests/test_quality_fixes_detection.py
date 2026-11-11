"""Regression tests of the 0.2.0 fixes that are switched off by default:
long-vowel backchannel words and the 0.1 s overlap boundary."""

import pytest

from fdbja.config import Config
from fdbja.data import event_detection as ed
from fdbja.metrics import tor

# --- (a) backchannel words with the long-vowel mark -------------------------

@pytest.mark.parametrize("word", ["えー", "ふーん", "あー", "へー", "おー", "はー", "ほー", "んー"])
def test_long_vowel_backchannels_match_when_enabled(word):
    n, r = ed.backchannel_residual(word, match_long_vowel=True)
    assert r == 0
    assert n == len(word.replace("ー", ""))


def test_long_vowel_default_keeps_published_behaviour():
    # the published definition removes "ー" first, so these do not match
    assert ed.backchannel_residual("えー") == (1, 1)
    assert ed.backchannel_residual("ふーん") == (2, 2)
    # words without the mark are unaffected by the switch
    for text in ("うんうん。はい、そうですね", "ハハハ、そうなんですね大学", "今日は大学に行った"):
        assert ed.backchannel_residual(text) == ed.backchannel_residual(text, True)


def test_long_vowel_in_content_words_is_not_counted():
    # "コーヒー": the marks are not counted either way; nothing is removed
    assert ed.backchannel_residual("コーヒー", True) == (2, 2)
    assert ed.backchannel_residual("えーコーヒー", True) == (3, 2)


def test_smooth_tt_roles_switch_reaches_the_checks():
    ev = {"a_text": "大学の授業がとても面白かったんですよ", "a_duration": 4.5,
          "a_end": 4.5, "b_text": "ふーんえーへー"}
    off = ed.smooth_tt_role_flags(ev, [], ed.SmoothTTRoles(enabled=True))
    on = ed.smooth_tt_role_flags(ev, [], ed.SmoothTTRoles(enabled=True, match_long_vowel=True))
    assert "b_backchannel" not in off      # "ふんえへ" left: 4 characters
    assert "b_backchannel" in on
    roles = ed.SmoothTTRoles.from_config({"enabled": True, "match_long_vowel": True})
    assert roles.match_long_vowel


# --- (b) overlap of exactly 0.1 s -------------------------------------------

def _pause_event(start=1.6, end=2.4):
    return {"start": start, "end": end, "duration": end - start}


def _b(start, end, n=10):
    return {"start": start, "end": end, "num_chars": n}


def test_tor_overlap_exactly_point_one():
    ev = _pause_event(1.6, 2.4)
    # model speech from 1.5 s to 1.7 s: overlap 1.7 - 1.6 = 0.0999... in floating point
    b = [_b(1.5, 1.7, n=10)]
    assert 1.7 - 1.6 < 0.1
    assert tor.eval_pause([ev], b) == [{"TO": 0}]     # published: falls short
    assert tor.eval_pause([ev], b, overlap_tolerance=ed.OVERLAP_TOLERANCE) == [{"TO": 1}]
    # clearly shorter overlaps stay out with the tolerance
    assert tor.eval_pause([ev], [_b(1.5, 1.69)],
                          overlap_tolerance=ed.OVERLAP_TOLERANCE) == [{"TO": 0}]


def _chars(speaker, text, start, step):
    return [{"speaker": speaker, "word": ch, "start": round(start + i * step, 3),
             "end": round(start + (i + 1) * step, 3)} for i, ch in enumerate(text)]


def test_detector_overlap_exactly_point_one():
    # A: 0-1.6 s, gap 1.6-2.2 s, A again 2.2-5.4 s; B speaks 1.5-1.7 s (0.1 s in the gap)
    jchat = (_chars("A", "あいうえおかきく", 0.0, 0.2)
             + _chars("A", "たちつてとなにぬねのはひふへほま", 2.2, 0.2)
             + [{"speaker": "B", "word": "ん", "start": 1.5, "end": 1.7}])
    assert 1.7 - 1.6 < 0.1
    assert len(ed.detect_pauses(jchat)) == 1                       # published: B ignored
    assert ed.detect_pauses(jchat, overlap_tolerance=ed.OVERLAP_TOLERANCE) == []
    cfg = Config({"detection": {"pause": {"overlap_tolerance": 1e-6}}})
    assert ed.detect("pause", jchat, None, cfg.detector_params("pause")) == []


def test_runner_passes_overlap_tolerance_to_tor():
    from fdbja.runner import _tor_events
    ev = _pause_event(1.6, 2.4)
    asr = _chars("B", "あいうえおかきくけこ", 1.5, 0.02)   # 1.5-1.7 s, 10 chars
    _, res, _ = _tor_events("pause", [], asr, events=[ev])
    assert res == [{"TO": 0}]
    _, res, _ = _tor_events("pause", [], asr, events=[ev], overlap_tolerance=1e-6)
    assert res == [{"TO": 1}]
