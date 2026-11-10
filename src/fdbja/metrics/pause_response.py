"""Pause response: penalise both not reacting and interrupting in a pause.

``metrics.pause_response``. For each Pause event (A's intra-turn gap
``[g_s, g_e]``), from the voice activity of the system response:

  * no response = 1 when no voiced run has at least ``min_response_seconds``
    of its length inside the gap;
  * interrupt = 1 when a voiced run starts inside the gap and lasts at least
    ``persist_after_resume_seconds`` past A's resumption ``g_e``, or when the
    system and A are voiced together for more than ``pre_overlap_max_seconds``
    in the ``pre_window_seconds`` before the gap;
  * pause response = no response + interrupt (0-2, lower is better). The
    system value is the mean over events.

Voice activity is the frame-energy detector of ``metrics/turnnat.py`` (50 ms
frames at 20 Hz, threshold 5 % of the channel's 99.5th-percentile frame
energy): speaker A from channel 0 of the GT clip, the system from channel 1 of
``combined.wav``. The definition and thresholds are the ones fixed before the
human-judgement study (2026-11): in that study the recorded human B rarely
spoke in these pauses, yet listeners preferred the systems that reacted
briefly, so the metric counts the reaction itself instead of the distance to
the human reference. Energy-based: backchannels, breaths and noise are not
told apart.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from fdbja.metrics.turnnat import voice_activity

TOL = 1e-9


@dataclass(frozen=True)
class PauseResponseConfig:
    min_response_seconds: float = 0.2          # no response: no in-gap voiced part this long
    persist_after_resume_seconds: float = 0.5  # interrupt: run started in the gap lasts this past g_e
    pre_window_seconds: float = 2.0            # interrupt: window before the gap ...
    pre_overlap_max_seconds: float = 0.5       # ... in which overlap with A must not exceed this
    frame_rate: int = 20
    rel_threshold: float = 0.05

    @classmethod
    def from_config(cls, cfg: Optional[Dict[str, Any]]) -> "PauseResponseConfig":
        """From ``metrics.pause_response`` (``enabled`` is ignored here)."""
        cfg = {k: v for k, v in dict(cfg or {}).items() if k != "enabled"}
        names = {f.name for f in fields(cls)}
        unknown = set(cfg) - names
        if unknown:
            raise ValueError(f"unknown metrics.pause_response keys: {sorted(unknown)} "
                             f"(allowed: {sorted(names)})")
        return cls(**cfg)


DEFAULT_PAUSE_RESPONSE = PauseResponseConfig()


def voiced_runs(va: np.ndarray, times: np.ndarray, dt: float
                ) -> List[Tuple[float, float]]:
    """``(start, end)`` of each voiced run; ``times`` are frame starts and a
    run ends one frame after its last voiced frame."""
    va = np.asarray(va, dtype=bool)
    out: List[Tuple[float, float]] = []
    i, n = 0, len(va)
    while i < n:
        if va[i]:
            j = i
            while j < n and va[j]:
                j += 1
            out.append((float(times[i]), float(times[j - 1]) + dt))
            i = j
        else:
            i += 1
    return out


def judge_event(va_sys: Sequence[bool], va_a: Sequence[bool], times: Sequence[float],
                gap_start: float, gap_end: float,
                cfg: PauseResponseConfig = DEFAULT_PAUSE_RESPONSE) -> Dict[str, Any]:
    """No response / interrupt of one Pause event from frame-level voice
    activity of the system (``va_sys``) and speaker A (``va_a``) on the frame
    grid ``times`` (frame starts, spacing ``1 / cfg.frame_rate``)."""
    dt = 1.0 / cfg.frame_rate
    vs = np.asarray(va_sys, dtype=bool)
    va = np.asarray(va_a, dtype=bool)
    t = np.asarray(times, dtype=np.float64)
    runs = voiced_runs(vs, t, dt)
    in_gap = [max(0.0, min(e, gap_end) - max(s, gap_start)) for s, e in runs]
    responded = any(x >= cfg.min_response_seconds - TOL for x in in_gap)
    persists = any(gap_start <= s <= gap_end
                   and e >= gap_end + cfg.persist_after_resume_seconds - TOL
                   for s, e in runs)
    pre = (t >= gap_start - cfg.pre_window_seconds) & (t < gap_start)
    pre_overlap = round(float((va & vs)[pre].sum()) * dt, 9)
    no_response = 0 if responded else 1
    interrupt = 1 if (persists or pre_overlap > cfg.pre_overlap_max_seconds) else 0
    return {"no_response": no_response, "interrupt": interrupt,
            "pause_response": no_response + interrupt,
            "pre_overlap": round(pre_overlap, 3)}


def activity_from_wavs(gt_wav: str, model_wav: str,
                       cfg: PauseResponseConfig = DEFAULT_PAUSE_RESPONSE
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(va_sys, va_a, times)``: A from channel 0 of the GT clip, the system
    from channel 1 of ``combined.wav`` (padded with silence / cut to A's
    length)."""
    import soundfile as sf

    g, sr_g = sf.read(gt_wav, dtype="float32", always_2d=True)
    m, sr_m = sf.read(model_wav, dtype="float32", always_2d=True)
    va_a = voice_activity(g[:, 0], sr_g, cfg.frame_rate, cfg.rel_threshold)
    va_s0 = voice_activity(m[:, min(1, m.shape[1] - 1)], sr_m, cfg.frame_rate,
                           cfg.rel_threshold)
    va_s = np.zeros(len(va_a), dtype=bool)
    k = min(len(va_a), len(va_s0))
    va_s[:k] = va_s0[:k]
    times = np.arange(len(va_a)) / cfg.frame_rate
    return va_s, va_a, times


def judge_events(gt_wav: str, model_wav: str, events: Sequence[Dict[str, Any]],
                 cfg: PauseResponseConfig = DEFAULT_PAUSE_RESPONSE
                 ) -> List[Dict[str, Any]]:
    """:func:`judge_event` for every Pause event of one clip."""
    va_s, va_a, times = activity_from_wavs(gt_wav, model_wav, cfg)
    return [judge_event(va_s, va_a, times, ev["start"], ev["end"], cfg) for ev in events]
