"""TurnNat-style likelihood scorer (comparison baseline for VAP-distance).

Reference: Zhang, Thebaud, Tinchev, Ravichandran, Moro-Velazquez,
"TurnNat: Automatic Evaluation of Turn-Taking Naturalness in Dyadic Spoken
Dialogue", arXiv:2607.01345 (2026).

TurnNat scores timing *atypicality* as the negative log-likelihood of the
observed future voice activity under a causal turn-taking predictor, pooled
over turn-taking boundary units (TBUs) and aggregated with a mean + tail
statistic. It shares this toolkit's 256-class VAP formulation exactly
(H = 2 s, bins [0, 200], [200, 600], [600, 1200], [1200, 2000] ms), so the same
frozen backend can drive both metrics.

The two metrics differ in what they are *relative to*:

    VAP-distance  D_KL( VAP(A + human_B) || VAP(A + model_B) )   -- paired,
                  needs the human response to the same event.
    TurnNat       -log P( observed VA | context )                -- absolute,
                  needs no reference, but only says "unlike training data".

This module exists so both can be reported on the same Japanese clips. Note
that TurnNat's own experiments found a released VAP checkpoint to be a weaker
scorer than their fine-tuned model, so a difference in favour of VAP-distance
here should be checked against a fine-tuned scorer before being claimed.

Numbers are only comparable to the paper's if the backend's class codebook uses
the same bit order; see ``pattern_to_class``.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

# TurnNat / VAP future horizon, in seconds.
BIN_EDGES = (0.0, 0.2, 0.6, 1.2, 2.0)
TBU_CONTEXT = 2.0     # L: TBU covers [tau - L, tau]
MIN_ACTIVE = 0.2      # discard VAD regions shorter than this (seconds)
EPS = 1e-12


def voice_activity(audio: np.ndarray, sample_rate: int, frame_rate: int = 20,
                   rel_threshold: float = 0.05) -> np.ndarray:
    """Frame-level binary voice activity from waveform energy.

    Threshold is relative to the track's own 99.5th-percentile frame energy, so
    it adapts to per-channel level differences. An all-zero channel yields all
    zeros rather than dividing by zero.
    """
    hop = max(1, int(round(sample_rate / frame_rate)))
    n = len(audio) // hop
    if n == 0:
        return np.zeros(0, dtype=bool)
    frames = np.asarray(audio[:n * hop], dtype=np.float64).reshape(n, hop)
    energy = np.sqrt((frames ** 2).mean(axis=1))
    ref = np.percentile(energy, 99.5)
    if ref <= 0:
        return np.zeros(n, dtype=bool)
    return energy > max(rel_threshold * ref, 1e-6)


def future_pattern(va_a: np.ndarray, va_b: np.ndarray, frame_rate: int = 20,
                   bin_edges: Sequence[float] = BIN_EDGES) -> np.ndarray:
    """Observed 8-bit future voice-activity pattern per frame.

    Bit ``i`` of speaker A is 1 iff A is active anywhere in future bin ``i``;
    bits are returned as ``(T, 8)`` with A's four bins first. Frames whose full
    horizon does not fit in the signal are dropped.
    """
    n = min(len(va_a), len(va_b))
    horizon = int(round(bin_edges[-1] * frame_rate))
    usable = n - horizon
    if usable <= 0:
        return np.zeros((0, 8), dtype=bool)
    edges = [int(round(e * frame_rate)) for e in bin_edges]
    out = np.zeros((usable, 8), dtype=bool)
    idx = np.arange(usable)
    for ch, va in enumerate((va_a[:n], va_b[:n])):
        # any() over each frame's future window, vectorised via a running sum
        cs = np.concatenate([[0], np.cumsum(va.astype(np.int32))])
        for b in range(len(edges) - 1):
            lo, hi = edges[b], edges[b + 1]
            out[:, ch * 4 + b] = (cs[idx + hi] - cs[idx + lo]) > 0
    return out


def pattern_to_class(bits: np.ndarray, msb_first: bool = False) -> np.ndarray:
    """Map ``(T, 8)`` bit patterns to ``(T,)`` class indices in [0, 256).

    ``msb_first`` selects whether bit 0 (speaker A, nearest bin) is the most
    significant bit. The default matches the codebook of the Japanese
    VAP-Realtime checkpoint (``vap_state_dict_jp_20hz_2500msec.pt``), verified
    2026-08-18 by marginalising its output over each bit position and checking
    the marginals against known activity: LSB-first, speaker A's four bins in
    the low nibble. Under this order a frame where A speaks and B is silent
    gives P(A active) = 0.98 / P(B active) = 0.08, and reversing any part of
    the order muddies or inverts that.

    The MIT checkpoint ``vap_state_dict_jp_kyoto_20hz_2500msec.pt``
    (maai-kyoto/vap_jp_kyoto) was verified the same way on 2026-09-26 and uses
    the same order (A speaking / B silent: P(A active) = 0.94, P(B active) =
    0.11; the reverse for B-only frames).

    **Re-verify when changing checkpoints** -- an ordering mismatch silently
    produces valid-looking but wrong likelihoods.
    """
    weights = (1 << np.arange(7, -1, -1)) if msb_first else (1 << np.arange(8))
    return (bits.astype(np.int64) * weights).sum(axis=1)


def nll_series(probs: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Frame-level ``-log P(observed class)``; lengths are truncated to match."""
    n = min(len(probs), len(classes))
    p = np.clip(probs[:n][np.arange(n), classes[:n]], EPS, 1.0)
    return -np.log(p)


def boundary_times(va_a: np.ndarray, va_b: np.ndarray, frame_rate: int = 20,
                   min_active: float = MIN_ACTIVE) -> List[float]:
    """Utterance onsets and offsets of both speakers, in seconds.

    Contiguous active regions shorter than ``min_active`` are discarded, then
    each surviving region contributes two boundaries (its onset and offset) --
    the TBU anchors of the paper.
    """
    times: List[float] = []
    min_frames = max(1, int(round(min_active * frame_rate)))
    for va in (va_a, va_b):
        i = 0
        while i < len(va):
            if not va[i]:
                i += 1
                continue
            j = i
            while j < len(va) and va[j]:
                j += 1
            if j - i >= min_frames:
                times.append(i / frame_rate)
                times.append(j / frame_rate)
            i = j
    return sorted(times)


def tbu_scores(nll: np.ndarray, boundaries: Sequence[float],
               frame_rate: int = 20, context: float = TBU_CONTEXT
               ) -> np.ndarray:
    """Mean NLL inside each TBU ``[tau - context, tau]``.

    Boundaries whose window falls entirely outside the scored frames are
    dropped, so an empty array means "no usable boundary in this clip".
    """
    out = []
    for tau in boundaries:
        lo = max(0, int(round((tau - context) * frame_rate)))
        hi = min(len(nll), int(round(tau * frame_rate)))
        if hi > lo:
            out.append(float(nll[lo:hi].mean()))
    return np.asarray(out, dtype=np.float64)


def aggregate(unit_scores: np.ndarray, lam: float = 0.5,
              top_k: int = 5) -> float:
    """``-(lam * MeanNLL + (1 - lam) * TailNLL)``; higher = more natural.

    ``TailNLL`` is the mean of the ``top_k`` highest-NLL units, which is what
    makes the score sensitive to a single badly-timed boundary in an otherwise
    ordinary dialogue.
    """
    if len(unit_scores) == 0:
        return float("nan")
    k = min(top_k, len(unit_scores))
    tail = float(np.sort(unit_scores)[-k:].mean())
    return -float(lam * unit_scores.mean() + (1.0 - lam) * tail)


def turnnat_score(probs: np.ndarray, times: np.ndarray,
                  audio_a: np.ndarray, audio_b: np.ndarray, sample_rate: int,
                  frame_rate: int = 20, lam: float = 0.5, top_k: int = 5,
                  msb_first: bool = True) -> Tuple[float, dict]:
    """End-to-end score for one clip.

    Args:
        probs: ``(T, 256)`` VAP distributions for the stereo pair (A, B), as
            returned by ``fdbja.vap.base.VAPModel.probs``.
        times: ``(T,)`` frame centres in seconds (used only for reporting).
        audio_a / audio_b: the same two mono channels the probs came from.

    Returns ``(score, detail)``; ``detail`` carries the per-TBU scores so an
    event-level analysis can reuse them.
    """
    va_a = voice_activity(audio_a, sample_rate, frame_rate)
    va_b = voice_activity(audio_b, sample_rate, frame_rate)
    bits = future_pattern(va_a, va_b, frame_rate)
    classes = pattern_to_class(bits, msb_first=msb_first)
    nll = nll_series(probs, classes)
    units = tbu_scores(nll, boundary_times(va_a, va_b, frame_rate), frame_rate)
    return aggregate(units, lam=lam, top_k=top_k), {
        "n_tbu": int(len(units)),
        "mean_nll": float(units.mean()) if len(units) else float("nan"),
        "frame_nll": nll,
        "unit_scores": units,
        "n_frames": int(len(nll)),
        "duration": float(times[-1]) if len(times) else 0.0,
    }
