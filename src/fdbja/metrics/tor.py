"""Binary Takeover (TO) / Takeover Rate (TOR) metric.

Follows the metric definition of the original Full-Duplex-Bench (independent
reimplementation for character-level Japanese transcripts): at each detected event,
TO=1 if the model takes the floor (speaks beyond a backchannel), else TO=0.
TOR is the mean TO over events. Desired direction is task-dependent
(see ``event_detection.DESIRED_TO``).

Inputs are the model's speaker-B utterances (grouped from the model output
ASR via ``group_chars_to_utterances``).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from fdbja.data.event_detection import MIN_OVERLAP, intervals_overlap

Utterance = Dict[str, Any]


def timing_jsd(human_times: Sequence[float], model_times: Sequence[float],
               start: float, end: float, bin_s: float = 0.2) -> float:
    """Jensen-Shannon divergence between the human and model backchannel
    onset histograms over the window [start, end] (bins of ``bin_s`` s).

    Follows the Backchannel task of the original Full-Duplex-Bench: onsets are
    binned at 0.2 s and the two histograms are compared with JSD; an event
    where the model produced no backchannel at all scores 1.0.  We report the
    base-2 JS *divergence* (range [0, 1]); the original scripts call
    ``scipy.spatial.distance.jensenshannon``, which returns the JS *distance*
    (square root, natural log), so the two are monotone but not identical.
    """
    n_bins = max(1, int(math.ceil((end - start) / bin_s)))

    def hist(ts: Sequence[float]) -> np.ndarray:
        h = np.zeros(n_bins)
        for t in ts:
            i = int((t - start) // bin_s)
            h[min(max(i, 0), n_bins - 1)] += 1
        return h

    hh, hm = hist(human_times), hist(model_times)
    if hh.sum() == 0 or hm.sum() == 0:
        return 1.0
    p, q = hh / hh.sum(), hm / hm.sum()
    m = (p + q) / 2

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def eval_pause(events, model_b_utts: List[Utterance],
               bc_max_chars: int = 6, bc_max_duration: float = 1.5,
               overlap_tolerance: float = 0.0):
    """During a pause the model should stay silent (desired TO=0).

    Model speech counts when it overlaps the gap by at least 0.1 s minus
    ``overlap_tolerance`` (default 0 = the published numbers, where an
    overlap of exactly 0.1 s can fall short through floating point; see
    ``event_detection.detect_pauses``)."""
    results = []
    for p in events:
        to = 0
        for b in model_b_utts:
            if intervals_overlap(b["start"], b["end"], p["start"], p["end"]):
                if ((min(b["end"], p["end"]) - max(b["start"], p["start"]))
                        < MIN_OVERLAP - overlap_tolerance):
                    continue
                if (b["num_chars"] > bc_max_chars
                        or (b["end"] - b["start"]) > bc_max_duration):
                    to = 1
                    break
        results.append({"TO": to})
    return results


def eval_smooth_tt(events, model_b_utts: List[Utterance],
                   min_response_chars: int = 3):
    """At a smooth transition the model should take the turn (desired TO=1)."""
    results = []
    for t in events:
        a_end = t["a_end"]
        found = False
        for b in model_b_utts:
            if a_end - 0.5 <= b["start"] <= a_end + 2.0:
                if b["num_chars"] >= min_response_chars:
                    # ``latency`` is clipped at 0 (the published column); ``onset``
                    # keeps the sign, so a response that starts before A has
                    # finished (overlap, down to -0.5 s) stays visible
                    results.append({"TO": 1, "latency": max(0.0, b["start"] - a_end),
                                    "onset": b["start"] - a_end})
                    found = True
                    break
        if not found:
            results.append({"TO": 0, "latency": None, "onset": None})
    return results


def eval_bc(events, model_b_utts: List[Utterance],
            bc_max_chars: int = 6, bc_max_duration: float = 1.5):
    """During a backchannel opportunity the model should only backchannel
    (desired TO=0); also report BC frequency (per second of A's utterance)
    and the timing JSD against the human backchannel onsets (``jsd``; None
    when the event carries no ``human_bc_times``)."""
    results = []
    for bc in events:
        a_start, a_end = bc["a_utt_start"], bc["a_utt_end"]
        a_dur = a_end - a_start
        bc_count = 0
        bc_times: List[float] = []
        took_over = False
        for b in model_b_utts:
            if intervals_overlap(b["start"], b["end"], a_start, a_end):
                if (b["num_chars"] <= bc_max_chars
                        and (b["end"] - b["start"]) <= bc_max_duration):
                    bc_count += 1
                    bc_times.append(b["start"])
                else:
                    took_over = True
        human_times: Optional[Sequence[float]] = bc.get("human_bc_times")
        results.append({
            "TO": 1 if took_over else 0,
            "bc_count": bc_count,
            "freq": bc_count / a_dur if a_dur > 0 else 0,
            "jsd": None if human_times is None
            else timing_jsd(human_times, bc_times, a_start, a_end),
        })
    return results


EVALUATORS = {
    "pause": eval_pause,
    "smooth_tt": eval_smooth_tt,
    "bc": eval_bc,
}
