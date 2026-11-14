"""Re-anchor event boundaries on the speech audio (audio boundaries).

The detectors in ``event_detection`` find events from the character-level
transcript, so their boundaries are character timestamps of the forced
alignment. On LLM-jp-Zoom1 those timestamps run early against the audio: the
speaker-A silence of a pause actually starts about 0.3 s after the last
character's ``end`` and A's voice comes back about 0.15 s after the next
character's ``start`` (measured 2026-10-22 on the reference evaluation clips). A window
taken from the transcript therefore still contains the end of A's speech.

This module keeps the *events* that the transcript detects (which events exist
does not change) and moves their *boundaries* to the nearest speech/silence
transitions of the human channels, measured with a frame-energy voice activity
detector:

  * frame RMS level in dB over non-overlapping 20 ms frames;
  * per-channel threshold: the ITU-T P.56 (method B) threshold computed on the
    frames, i.e. the level ``c`` for which the mean power of the frames above
    ``c`` (the active speech level) is ``margin_db`` = 15.9 dB above ``c``;
    never closer than ``min_snr_db`` to the noise floor (5th percentile of the
    non-digital-zero frames). No hangover (P.56's 0.2 s would delay every
    offset). ``mode="floor"`` / ``"fixed"`` are kept for sensitivity checks;
  * voiced runs shorter than ``min_voiced`` are dropped (clicks, breaths), then
    silent runs shorter than ``min_hole`` inside speech are filled (closures);
  * pause: the silent run of speaker A that overlaps the transcript gap most,
    searched within ``search`` seconds of the gap. Its start and end become the
    new ``start`` / ``end``;
  * smooth_tt: ``a_end`` = the end of A's voiced run nearest to the transcript
    ``a_end`` (within ``search``) that is followed by at least ``min_silence``
    of silence; ``b_start`` = the start of B's voiced run nearest to the
    transcript ``b_start`` that is preceded by at least ``min_silence``;
  * bc: ``a_utt_start`` / ``a_utt_end`` with the same onset / offset rules on
    A, and each human backchannel onset with the onset rule on B.

A boundary that cannot be found keeps its transcript value and the event is
flagged (``boundary["flags"]``) so that the caller can count or exclude it; this
module never drops an event. The transcript values are kept under
``boundary["asr"]``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DB_FLOOR = -120.0


@dataclass(frozen=True)
class VADConfig:
    frame: float = 0.02
    # threshold: "p56" = active speech level - margin_db (ITU-T P.56 style),
    # "floor" = noise floor + rel_db, "fixed" = fixed_db dBFS
    mode: str = "p56"
    margin_db: float = 15.9
    rel_db: float = 20.0
    fixed_db: float = -40.0
    min_snr_db: float = 6.0
    floor_pct: float = 5.0
    min_voiced: float = 0.10
    min_hole: float = 0.06
    search: float = 0.5
    min_silence: float = 0.3


DEFAULT_VAD = VADConfig()




# ---------------------------------------------------------------------------
# Frame-level voice activity
# ---------------------------------------------------------------------------

def frame_db(x: np.ndarray, sr: int, frame: float = 0.02) -> np.ndarray:
    """RMS level (dBFS) of non-overlapping frames; digital zero -> DB_FLOOR."""
    n = int(round(sr * frame))
    m = len(x) // n
    if m == 0:
        return np.zeros(0)
    rms = np.sqrt((np.asarray(x[: m * n], dtype=np.float64).reshape(m, n) ** 2).mean(1))
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(rms)
    return np.maximum(db, DB_FLOOR)


def active_level_db(db: np.ndarray, margin_db: float = 15.9) -> Tuple[float, float]:
    """(active level, threshold) of frame levels, ITU-T P.56 method B on frames.

    The threshold ``c`` is the one for which the mean power of the frames above
    ``c`` (the active level) lies ``margin_db`` above ``c``. P.56 works on a
    smoothed envelope with a 0.2 s hangover; here the frame RMS is used
    directly and no hangover is applied, because the hangover would move every
    speech offset late by 0.2 s.
    """
    live = np.sort(db[db > DB_FLOOR + 1.0])
    if len(live) == 0:
        return float("-inf"), float("inf")
    pw = 10.0 ** (live / 10.0)
    tail = np.concatenate([np.cumsum(pw[::-1])[::-1], [0.0]])   # sum of pw[i:]
    grid = np.arange(live[0], live[-1], 0.25)                     # thresholds c (dB)
    if len(grid) == 0:
        return float(live[-1]), float(live[0])
    i = np.searchsorted(live, grid, side="right")                 # frames above c
    cnt = len(live) - i
    act = 10.0 * np.log10(tail[i] / np.maximum(cnt, 1))
    ok = np.flatnonzero(act - grid <= margin_db)
    j = int(ok[0]) if len(ok) else len(grid) - 1
    return float(act[j]), float(grid[j])


def threshold_db(db: np.ndarray, cfg: VADConfig = DEFAULT_VAD) -> float:
    live = db[db > DB_FLOOR + 1.0]
    if len(live) == 0:
        return float("inf")          # an all-zero channel is silent everywhere
    floor = float(np.percentile(live, cfg.floor_pct))
    if cfg.mode == "fixed":
        return cfg.fixed_db
    if cfg.mode == "floor":
        return floor + cfg.rel_db
    _, thr = active_level_db(db, cfg.margin_db)
    return max(thr, floor + cfg.min_snr_db)


def _runs(mask: np.ndarray) -> List[Tuple[int, int, bool]]:
    """(start, end_exclusive, value) of constant runs."""
    if len(mask) == 0:
        return []
    edges = np.flatnonzero(np.diff(mask.astype(np.int8))) + 1
    starts = np.concatenate([[0], edges])
    ends = np.concatenate([edges, [len(mask)]])
    return [(int(s), int(e), bool(mask[s])) for s, e in zip(starts, ends)]


def _drop_short(mask: np.ndarray, value: bool, min_frames: int,
                keep_edges: bool = True) -> np.ndarray:
    out = mask.copy()
    n = len(mask)
    for s, e, v in _runs(mask):
        if v != value or e - s >= min_frames:
            continue
        if keep_edges and (s == 0 or e == n):
            continue
        out[s:e] = not value
    return out


def voice_activity(x: np.ndarray, sr: int,
                   cfg: VADConfig = DEFAULT_VAD,
                   min_level_db: Optional[float] = None) -> np.ndarray:
    """Boolean voice activity per ``cfg.frame`` frame of a mono signal.

    The threshold is relative to the channel's own level, so a channel that
    holds only faint noise would have the louder part of that noise marked as
    voice. With ``min_level_db`` (dBFS) frames below that level are never
    voice; ``None`` (the benchmark's event detection) keeps the relative
    threshold alone.
    """
    return voice_activity_levels(x, sr, cfg, min_level_db)[0]


def voice_activity_levels(x: np.ndarray, sr: int,
                          cfg: VADConfig = DEFAULT_VAD,
                          min_level_db: Optional[float] = None
                          ) -> Tuple[np.ndarray, np.ndarray, float]:
    """:func:`voice_activity` together with what it was decided from:
    ``(voice activity, frame levels in dBFS, threshold in dBFS)``. The
    threshold is ``inf`` for an all-zero channel."""
    db = frame_db(x, sr, cfg.frame)
    thr = threshold_db(db, cfg)
    if min_level_db is not None:
        thr = max(thr, float(min_level_db))
    v = db > thr
    v = _drop_short(v, True, int(round(cfg.min_voiced / cfg.frame)), keep_edges=False)
    v = _drop_short(v, False, int(round(cfg.min_hole / cfg.frame)))
    return v, db, thr


# ---------------------------------------------------------------------------
# Boundary search
# ---------------------------------------------------------------------------

def refine_gap(va: np.ndarray, start: float, end: float,
               cfg: VADConfig = DEFAULT_VAD) -> Dict:
    """The silent run of ``va`` that overlaps [start, end] most.

    Returns ``{"start", "end", "flags"}``; a side whose silence extends beyond
    ``cfg.search`` keeps its transcript value and is flagged.
    """
    f = cfg.frame
    lo, hi = start - cfg.search, end + cfg.search
    best = None
    for s, e, v in _runs(va):
        if v:
            continue
        ts, te = s * f, e * f
        if te <= lo or ts >= hi:
            continue
        ov = min(te, end) - max(ts, start)
        key = (ov, te - ts)
        if best is None or key > best[0]:
            best = (key, ts, te)
    if best is None or best[0][0] <= 0:
        return {"start": start, "end": end, "flags": ["no_silence"]}
    _, ts, te = best
    flags = []
    if ts < lo:
        ts, flags = start, flags + ["start_not_found"]
    if te > hi:
        te, flags = end, flags + ["end_not_found"]
    return {"start": round(ts, 3), "end": round(te, 3), "flags": flags}


def refine_offset(va: np.ndarray, t: float,
                  cfg: VADConfig = DEFAULT_VAD) -> Optional[float]:
    """End of the voiced run nearest to ``t`` that is followed by at least
    ``cfg.min_silence`` of silence (or the end of the signal)."""
    f = cfg.frame
    need = cfg.min_silence - 1e-9
    runs = _runs(va)
    cands = []
    for i, (s, e, v) in enumerate(runs):
        if not v:
            continue
        nxt = runs[i + 1] if i + 1 < len(runs) else None
        if nxt is not None and (nxt[1] - nxt[0]) * f < need:
            continue
        if abs(e * f - t) <= cfg.search:
            cands.append(e * f)
    return round(min(cands, key=lambda x: abs(x - t)), 3) if cands else None


def refine_onset(va: np.ndarray, t: float,
                 cfg: VADConfig = DEFAULT_VAD) -> Optional[float]:
    """Start of the voiced run nearest to ``t`` that is preceded by at least
    ``cfg.min_silence`` of silence (or the start of the signal)."""
    f = cfg.frame
    need = cfg.min_silence - 1e-9
    runs = _runs(va)
    cands = []
    for i, (s, e, v) in enumerate(runs):
        if not v:
            continue
        prv = runs[i - 1] if i > 0 else None
        if prv is not None and (prv[1] - prv[0]) * f < need:
            continue
        if abs(s * f - t) <= cfg.search:
            cands.append(s * f)
    return round(min(cands, key=lambda x: abs(x - t)), 3) if cands else None


def voiced_seconds(va: np.ndarray, t0: float, t1: float,
                   cfg: VADConfig = DEFAULT_VAD) -> float:
    """Seconds of voice activity of ``va`` in [t0, t1]."""
    i0 = max(0, int(round(t0 / cfg.frame)))
    i1 = min(len(va), int(round(t1 / cfg.frame)))
    return float(va[i0:i1].sum() * cfg.frame) if i1 > i0 else 0.0


def refine_events(task: str, events: Sequence[Dict], va_a: np.ndarray,
                  va_b: Optional[np.ndarray] = None,
                  cfg: VADConfig = DEFAULT_VAD, roles=None) -> List[Dict]:
    """Copies of ``events`` with boundaries moved onto the audio.

    ``va_a`` / ``va_b`` are the voice activity of the human speaker A and the
    human interlocutor B (``voice_activity`` of the GT clip's channels).

    ``roles`` (``event_detection.SmoothTTRoles``): when enabled, a
    Smooth TT event whose B channel is voiced for more than ``b_pre_max``
    seconds in the ``b_pre_window`` before the (re-anchored) ``a_end`` gets
    ``"b_voiced_before"`` in ``role_flags``. Events are still never dropped
    here; ``drop_role_flagged`` removes them.
    """
    out = []
    for ev in events:
        ev2 = copy.deepcopy(ev)
        flags: List[str] = []
        if task == "pause":
            asr = {"start": ev["start"], "end": ev["end"]}
            r = refine_gap(va_a, ev["start"], ev["end"], cfg)
            ev2["start"], ev2["end"] = r["start"], r["end"]
            ev2["duration"] = round(r["end"] - r["start"], 3)
            flags += r["flags"]
            if va_b is not None:
                i0, i1 = int(round(ev2["start"] / cfg.frame)), int(round(ev2["end"] / cfg.frame))
                if va_b[i0:i1].sum() * cfg.frame >= 0.1:
                    flags.append("b_voiced_in_gap")
        elif task == "smooth_tt":
            asr = {"a_end": ev["a_end"], "b_start": ev["b_start"]}
            a_end = refine_offset(va_a, ev["a_end"], cfg)
            if a_end is None:
                flags.append("a_end_not_found")
            else:
                ev2["a_end"] = a_end
            if va_b is not None:
                b_start = refine_onset(va_b, ev["b_start"], cfg)
                if b_start is None:
                    flags.append("b_start_not_found")
                else:
                    ev2["b_start"] = b_start
            ev2["gap"] = round(ev2["b_start"] - ev2["a_end"], 3)
            ev2["latency"] = ev2["gap"]
            if roles is not None and roles.enabled and va_b is not None:
                rf = list(ev2.get("role_flags", []))
                if voiced_seconds(va_b, ev2["a_end"] - roles.b_pre_window,
                                  ev2["a_end"], cfg) > roles.b_pre_max:
                    rf.append("b_voiced_before")
                ev2["role_flags"] = rf
        elif task == "bc":
            asr = {"a_utt_start": ev["a_utt_start"], "a_utt_end": ev["a_utt_end"]}
            s = refine_onset(va_a, ev["a_utt_start"], cfg)
            e = refine_offset(va_a, ev["a_utt_end"], cfg)
            if s is None:
                flags.append("a_start_not_found")
            else:
                ev2["a_utt_start"] = s
            if e is None:
                flags.append("a_end_not_found")
            else:
                ev2["a_utt_end"] = e
            ev2["a_duration"] = round(ev2["a_utt_end"] - ev2["a_utt_start"], 3)
            if va_b is not None and ev.get("human_bc_times"):
                asr["human_bc_times"] = list(ev["human_bc_times"])
                ev2["human_bc_times"] = [
                    t2 if (t2 := refine_onset(va_b, t, cfg)) is not None else t
                    for t in ev["human_bc_times"]]
        else:
            raise ValueError(f"unknown task: {task}")
        ev2["boundary"] = {"source": "audio", "asr": asr, "flags": flags}
        out.append(ev2)
    return out


def drop_role_flagged(events: Sequence[Dict]) -> List[Dict]:
    """Events without ``role_flags`` (events without the key are kept)."""
    return [ev for ev in events if not ev.get("role_flags")]


def refine_from_wav(task: str, events: Sequence[Dict], wav_path: str,
                    cfg: VADConfig = DEFAULT_VAD, roles=None) -> List[Dict]:
    """``refine_events`` on a stereo GT clip (ch0 = A, ch1 = human B)."""
    import soundfile as sf
    x, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    va_a = voice_activity(x[:, 0], sr, cfg)
    va_b = voice_activity(x[:, 1], sr, cfg) if x.shape[1] > 1 else None
    return refine_events(task, events, va_a, va_b, cfg, roles=roles)
