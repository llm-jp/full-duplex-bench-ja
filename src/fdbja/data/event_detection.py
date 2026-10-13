"""Event detection for Full-Duplex-Bench-JA.

Detects three turn-taking event types from character-level time-aligned
dialogue transcripts (jchat format: a list of
``{"speaker", "word", "start", "end"}``), aligned with the original
Full-Duplex-Bench (Lin et al., 2025):

  - Pause Handling   : intra-turn gap (0.4-1.0s) in speaker A where B is silent
  - Smooth Turn-Taking: A (>=4s) -> gap (<0.4s) -> B (>=4s), no overlap
  - Backchannel       : A utterance (>=3s) during which human B only backchannels

Ground truth follows the human interlocutor's behavior: the model is expected
to stay silent during Pause/BC (desired TO=0) and to take the turn at Smooth
TT (desired TO=1).
"""

from __future__ import annotations

from typing import List, Dict, Any

JChat = List[Dict[str, Any]]
Utterance = Dict[str, Any]


# ---------------------------------------------------------------------------
# Timeline construction
# ---------------------------------------------------------------------------

def group_chars_to_utterances(jchat: JChat, speaker: str,
                              gap_threshold: float = 0.3) -> List[Utterance]:
    """Group consecutive same-speaker characters into utterances.

    Characters separated by a silence longer than ``gap_threshold`` start a
    new utterance. Returns dicts with ``start``, ``end``, ``text``,
    ``num_chars``, ``chars``.
    """
    chars = [c for c in jchat if c["speaker"] == speaker]
    if not chars:
        return []

    utterances: List[Utterance] = []
    current = [chars[0]]
    for c in chars[1:]:
        if c["start"] - current[-1]["end"] > gap_threshold:
            utterances.append(_chars_to_utterance(current))
            current = [c]
        else:
            current.append(c)
    utterances.append(_chars_to_utterance(current))
    return utterances


def _chars_to_utterance(chars: JChat) -> Utterance:
    return {
        "start": chars[0]["start"],
        "end": chars[-1]["end"],
        "text": "".join(c["word"] for c in chars),
        "num_chars": len(chars),
        "chars": chars,
    }


def intervals_overlap(s1: float, e1: float, s2: float, e2: float) -> bool:
    return s1 < e2 and s2 < e1


def is_backchannel(utt: Utterance, max_chars: int = 6,
                   max_duration: float = 1.5) -> bool:
    """A short utterance counts as a backchannel."""
    return (utt["num_chars"] <= max_chars
            and (utt["end"] - utt["start"]) <= max_duration)


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------

def detect_pauses(jchat: JChat, min_gap: float = 0.4, max_gap: float = 1.0,
                  min_a_turn_duration: float = 5.0,
                  bc_max_chars: int = 6, bc_max_duration: float = 1.5
                  ) -> List[Dict[str, Any]]:
    """Intra-turn pauses in A where B stays silent (desired TO=0)."""
    a_utts = group_chars_to_utterances(jchat, "A")
    b_utts = group_chars_to_utterances(jchat, "B")

    pauses = []
    for i in range(len(a_utts) - 1):
        a_before, a_after = a_utts[i], a_utts[i + 1]
        gap_start, gap_end = a_before["end"], a_after["start"]
        gap_duration = gap_end - gap_start

        if not (min_gap <= gap_duration <= max_gap):
            continue
        if is_backchannel(a_before, bc_max_chars, bc_max_duration):
            continue
        if is_backchannel(a_after, bc_max_chars, bc_max_duration):
            continue

        # contiguous A turn around the gap (extend over sub-max_gap gaps)
        turn_start, turn_end = a_before["start"], a_after["end"]
        j = i - 1
        while j >= 0 and (a_utts[j + 1]["start"] - a_utts[j]["end"]) <= max_gap:
            turn_start = a_utts[j]["start"]
            j -= 1
        j = i + 2
        while j < len(a_utts) and (a_utts[j]["start"] - a_utts[j - 1]["end"]) <= max_gap:
            turn_end = a_utts[j]["end"]
            j += 1
        if (turn_end - turn_start) < min_a_turn_duration:
            continue

        b_spoke = any(
            intervals_overlap(b["start"], b["end"], gap_start, gap_end)
            and (min(b["end"], gap_end) - max(b["start"], gap_start)) >= 0.1
            for b in b_utts
        )
        if b_spoke:
            continue

        pauses.append({
            "start": gap_start,
            "end": gap_end,
            "duration": gap_duration,
            "a_before_text": a_before["text"],
            "a_after_text": a_after["text"],
            "a_turn_duration": turn_end - turn_start,
        })
    return pauses


def detect_smooth_tt(jchat: JChat, min_a_duration: float = 4.0,
                     max_gap: float = 0.4, min_b_duration: float = 4.0
                     ) -> List[Dict[str, Any]]:
    """Smooth inter-turn transitions A->B (desired TO=1)."""
    a_utts = group_chars_to_utterances(jchat, "A")
    b_utts = group_chars_to_utterances(jchat, "B")

    events = []
    for a in a_utts:
        if (a["end"] - a["start"]) < min_a_duration:
            continue
        for b in b_utts:
            if (b["end"] - b["start"]) < min_b_duration:
                continue
            gap = b["start"] - a["end"]
            if gap < 0 or gap >= max_gap:
                continue
            if intervals_overlap(a["start"], a["end"], b["start"], b["end"]):
                continue
            events.append({
                "a_end": a["end"],
                "b_start": b["start"],
                "gap": gap,
                "a_duration": a["end"] - a["start"],
                "b_duration": b["end"] - b["start"],
                "a_text": a["text"],
                "b_text": b["text"],
                "latency": gap,
            })
    return events


def detect_backchannels(jchat: JChat, min_a_duration: float = 3.0,
                        bc_max_chars: int = 6, bc_max_duration: float = 1.5
                        ) -> List[Dict[str, Any]]:
    """Backchannel opportunities: A (>=3s) where human B only backchannels
    (desired TO=0)."""
    a_utts = group_chars_to_utterances(jchat, "A")
    b_utts = group_chars_to_utterances(jchat, "B")

    events = []
    for a in a_utts:
        a_dur = a["end"] - a["start"]
        if a_dur < min_a_duration:
            continue
        took_over = False
        bc_count = 0
        bc_times: List[float] = []
        for b in b_utts:
            if intervals_overlap(b["start"], b["end"], a["start"], a["end"]):
                if not is_backchannel(b, bc_max_chars, bc_max_duration):
                    took_over = True
                    break
                bc_count += 1
                bc_times.append(b["start"])
        # A backchannel opportunity must contain an observed human
        # backchannel.  Without this guard, ordinary A-only speech was
        # incorrectly labelled as a backchannel event.
        if not took_over and bc_count >= 1:
            events.append({
                "a_utt_start": a["start"],
                "a_utt_end": a["end"],
                "a_duration": a_dur,
                "a_text": a["text"],
                "human_bc_count": bc_count,
                "human_freq": bc_count / a_dur if a_dur > 0 else 0,
                # onset times of the human backchannels (for timing JSD)
                "human_bc_times": bc_times,
            })
    return events


# desired Takeover per task: Pause/BC -> stay silent (0), Smooth TT -> take (1)
DESIRED_TO = {"pause": 0, "smooth_tt": 1, "bc": 0}

DETECTORS = {
    "pause": detect_pauses,
    "smooth_tt": detect_smooth_tt,
    "bc": detect_backchannels,
}
