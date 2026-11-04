"""Smooth TT turn-transition checks (``detection.smooth_tt_roles``)."""

import json
import os

import numpy as np
import pytest
import soundfile as sf

from fdbja.data import event_boundaries as eb
from fdbja.data import event_detection as ed
from fdbja.runner import _detect_events

ON = ed.SmoothTTRoles(enabled=True)


def _chars(speaker, text, start, step=0.25):
    return [{"speaker": speaker, "word": ch,
             "start": round(start + i * step, 3), "end": round(start + (i + 1) * step, 3)}
            for i, ch in enumerate(text)]


A_TALK = "今日は大学に行って授業を受けてきました"          # 19 chars, 4.75 s
A_BC = "うんうんうんはいはいそうそうなるほど"               # 18 chars, all backchannel words
B_TALK = "私も昨日行きましたよ楽しかったです本当に"        # 20 chars, 5.0 s
B_BC = "うんうんそうそうなるほどはいはいへえへえ"           # 20 chars, backchannel chain


def _dialogue(a_text=A_TALK, b_text=B_TALK, b_pre=None):
    a = _chars("A", a_text, 0.0)
    a_end = a[-1]["end"]
    b = _chars("B", b_text, a_end + 0.2)
    pre = _chars("B", b_pre, a_end - 1.6) if b_pre else []
    return a + pre + b


def test_backchannel_residual():
    assert ed.backchannel_residual("うんうん。はい、そうですね") == (11, 0)
    assert ed.backchannel_residual("ハハハ、そうなんですね大学") == (9, 2)
    assert ed.backchannel_residual(A_TALK) == (19, 19)


def test_default_is_unchanged():
    d = _dialogue(A_BC)
    base = ed.detect_smooth_tt(d)
    assert len(base) == 1 and "source_index" not in base[0]
    assert ed.detect_smooth_tt(d, roles=ed.SmoothTTRoles()) == base
    assert ed.detect("smooth_tt", d, None) == base
    assert ed.detect("smooth_tt", d, ed.SmoothTTRoles()) == base


def test_handover_kept_with_source_index():
    evs = ed.detect_smooth_tt(_dialogue(), roles=ON)
    assert len(evs) == 1
    assert evs[0]["source_index"] == 0 and evs[0]["role_flags"] == []


@pytest.mark.parametrize("kw, flag", [
    ({"a_text": A_BC}, "a_backchannel"),
    ({"b_text": B_BC}, "b_backchannel"),
    ({"b_pre": "それでその"}, "b_talking_before"),     # 1.25 s of B in the last 2 s of A
])
def test_text_checks(kw, flag):
    d = _dialogue(**kw)
    (ev,) = ed.detect_smooth_tt(d)
    assert flag in ed.smooth_tt_role_flags(ev, d, ON)
    assert ed.detect_smooth_tt(d, roles=ON) == []


def test_short_b_backchannel_before_is_allowed():
    d = _dialogue(b_pre="うん")                       # 0.5 s, within b_pre_max
    assert len(ed.detect_smooth_tt(d, roles=ON)) == 1


def test_source_index_skips_dropped_events():
    first = _dialogue(A_BC)
    t0 = first[-1]["end"] + 2.0
    second = [{**c, "start": c["start"] + t0, "end": c["end"] + t0} for c in _dialogue()]
    d = first + second
    assert len(ed.detect_smooth_tt(d)) == 2
    (ev,) = ed.detect_smooth_tt(d, roles=ON)
    assert ev["source_index"] == 1


def test_from_config():
    assert ed.SmoothTTRoles.from_config(None) == ed.SmoothTTRoles()
    assert ed.SmoothTTRoles.from_config({"enabled": True, "b_pre_max": 0.3}).b_pre_max == 0.3
    with pytest.raises(ValueError):
        ed.SmoothTTRoles.from_config({"enable": True})


def _va(segments, total=12.0, frame=0.02):
    va = np.zeros(int(round(total / frame)), dtype=bool)
    for s, e in segments:
        va[int(round(s / frame)):int(round(e / frame))] = True
    return va


def test_audio_check_flags_b_voiced_before():
    d = _dialogue()
    evs = ed.detect_smooth_tt(d, roles=ON)
    a_end = evs[0]["a_end"]
    va_a = _va([(0.0, a_end)])
    quiet_b = _va([(a_end + 0.2, a_end + 5.2)])
    busy_b = _va([(a_end - 1.5, a_end - 0.6), (a_end + 0.2, a_end + 5.2)])
    (ok,) = eb.refine_events("smooth_tt", evs, va_a, quiet_b, roles=ON)
    (bad,) = eb.refine_events("smooth_tt", evs, va_a, busy_b, roles=ON)
    assert ok["role_flags"] == [] and bad["role_flags"] == ["b_voiced_before"]
    assert eb.drop_role_flagged([ok, bad]) == [ok]
    # without roles nothing is added
    (plain,) = eb.refine_events("smooth_tt", ed.detect_smooth_tt(d), va_a, busy_b)
    assert "role_flags" not in plain


def _tone(segments, total, sr=16000):
    t = np.arange(int(total * sr)) / sr
    x = np.random.default_rng(0).standard_normal(len(t)) * 1e-4
    for s, e in segments:
        m = (t >= s) & (t < e)
        x[m] += 0.1 * np.sin(2 * np.pi * 220 * t[m])
    return x


def test_runner_detect_events_audio(tmp_path):
    d = _dialogue()
    a_end = d[18]["end"]
    total = 12.0
    wav = os.path.join(tmp_path, "clip.wav")
    a = _tone([(0.0, a_end)], total)
    b = _tone([(a_end - 1.5, a_end - 0.6), (a_end + 0.2, a_end + 5.2)], total)
    sf.write(wav, np.stack([a, b], 1), 16000)
    with open(os.path.join(tmp_path, "clip.json"), "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)
    assert len(_detect_events("smooth_tt", d, wav, "audio")) == 1
    assert len(_detect_events("smooth_tt", d, wav, "transcript", ON)) == 1
    assert _detect_events("smooth_tt", d, wav, "audio", ON) == []      # B voiced before a_end
    # pause is never affected
    assert (_detect_events("pause", d, wav, "audio", ON)
            == _detect_events("pause", d, wav, "audio"))
