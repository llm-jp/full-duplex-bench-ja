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

from typing import Any, Dict, List

from fdbja.data.event_detection import intervals_overlap

Utterance = Dict[str, Any]


def eval_pause(events, model_b_utts: List[Utterance],
               bc_max_chars: int = 6, bc_max_duration: float = 1.5):
    """During a pause the model should stay silent (desired TO=0)."""
    results = []
    for p in events:
        to = 0
        for b in model_b_utts:
            if intervals_overlap(b["start"], b["end"], p["start"], p["end"]):
                if (min(b["end"], p["end"]) - max(b["start"], p["start"])) < 0.1:
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
                    results.append({"TO": 1, "latency": max(0.0, b["start"] - a_end)})
                    found = True
                    break
        if not found:
            results.append({"TO": 0, "latency": None})
    return results


def eval_bc(events, model_b_utts: List[Utterance],
            bc_max_chars: int = 6, bc_max_duration: float = 1.5):
    """During a backchannel opportunity the model should only backchannel
    (desired TO=0); also report BC frequency."""
    results = []
    for bc in events:
        a_start, a_end = bc["a_utt_start"], bc["a_utt_end"]
        a_dur = a_end - a_start
        bc_count = 0
        took_over = False
        for b in model_b_utts:
            if intervals_overlap(b["start"], b["end"], a_start, a_end):
                if (b["num_chars"] <= bc_max_chars
                        and (b["end"] - b["start"]) <= bc_max_duration):
                    bc_count += 1
                else:
                    took_over = True
        results.append({
            "TO": 1 if took_over else 0,
            "bc_count": bc_count,
            "freq": bc_count / a_dur if a_dur > 0 else 0,
        })
    return results


EVALUATORS = {
    "pause": eval_pause,
    "smooth_tt": eval_smooth_tt,
    "bc": eval_bc,
}
