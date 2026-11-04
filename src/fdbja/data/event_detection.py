"""Event detection for Full-Duplex-Bench-JA.

Detects three turn-taking event types from character-level time-aligned
dialogue transcripts (jchat format: a list of
``{"speaker", "word", "start", "end"}``), aligned with the original
Full-Duplex-Bench (Lin et al., 2025):

  - Pause Handling   : intra-turn gap (0.4-1.0s) in speaker A where B is silent
  - Smooth Turn-Taking: A (>=4s) -> gap (<0.4s) -> B (>=4s), no overlap
  - Backchannel       : A utterance (>=3s) during which human B only backchannels

Turn-transition checks (optional, ``SmoothTTRoles``): a Smooth TT event additionally
requires that A is actually the speaker who hands the turn over. Without it,
``group_chars_to_utterances`` joins a run of backchannels ("うんうんうん。")
into one >=4 s "utterance" of A, and B's ongoing turn (shorter pieces of it
are not checked for overlap) is mistaken for a hand-over (about 3 in 10 of the
v4_eval Smooth TT events, measured 2026-10-22).

Ground truth follows the human interlocutor's behavior: the model is expected
to stay silent during Pause/BC (desired TO=0) and to take the turn at Smooth
TT (desired TO=1).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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
# Backchannel words in text (Smooth TT turn-transition checks)
# ---------------------------------------------------------------------------

# Response tokens / backchannels (hiragana, kanji, katakana variants). Single
# kana are left out because they also occur inside content words.
BACKCHANNEL_TOKENS: Tuple[str, ...] = tuple(sorted({
    "うんうん", "ううん", "うーん", "んー", "はいはい", "ええ", "えー", "そうですね",
    "そうですよね", "そうなんですね", "そうなんだ", "そうなんや", "そうそう", "そうか",
    "そっか", "そうだね", "そうね", "そう", "なるほど", "へえ", "へー", "へぇ", "ほう",
    "ほー", "ふーん", "あー", "ああ", "あぁ", "おー", "おお", "わかる", "分かる",
    "わかります", "分かります", "確かに", "たしかに", "ですね", "ですよね", "よね",
    "まあ", "まぁ", "いや", "いやいや", "本当", "ほんと", "ほんとに", "本当に", "すごい",
    "すご", "えっ", "はー", "うわ", "うん", "はい", "あっ", "んっ", "おっ"},
    key=lambda w: (-len(w), w)))
_LAUGH = re.compile(r"[アハフヘホ]{2,}|[ハフ]+[!！]?")
_PUNCT = re.compile(r"[、。，．,.!?！？…・\s「」ー]")


def backchannel_residual(text: str) -> Tuple[int, int]:
    """(characters, characters left after removing laughter and backchannel
    words). Punctuation, spaces and the long-vowel mark are not counted."""
    t = _PUNCT.sub("", _LAUGH.sub("", str(text)))
    n = len(t)
    for w in BACKCHANNEL_TOKENS:
        t = t.replace(w, "")
    return n, len(t)


@dataclass(frozen=True)
class SmoothTTRoles:
    """Checks that a Smooth TT event is a hand-over from A to B.

    (a) A's utterance is not backchannel-dominant: after removing backchannel
        words it keeps more than ``a_max_resid_chars`` characters (the
        ``is_backchannel`` character limit), at least ``a_min_resid_cps``
        characters per second and a share of at least ``a_min_resid_ratio``;
    (b) B is not speaking during the ``b_pre_window`` seconds before ``a_end``:
        at most ``b_pre_max`` seconds of B's characters (and, with audio
        boundaries, of B's voice activity) fall in that window;
    (c) B's utterance is not a chain of backchannels: it keeps more than
        ``b_max_resid_chars`` characters and a share of at least
        ``b_min_resid_ratio`` after removing backchannel words.

    ``enabled=False`` (default) reproduces the unrestricted event set exactly.
    """
    enabled: bool = False
    a_max_resid_chars: int = 6
    a_min_resid_cps: float = 1.5
    a_min_resid_ratio: float = 0.4
    b_pre_window: float = 2.0
    b_pre_max: float = 0.5
    b_max_resid_chars: int = 2
    b_min_resid_ratio: float = 0.3

    @classmethod
    def from_config(cls, cfg: Optional[Dict[str, Any]]) -> "SmoothTTRoles":
        if not cfg:
            return cls()
        if isinstance(cfg, cls):
            return cfg
        unknown = set(cfg) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown detection.smooth_tt_roles keys: {sorted(unknown)}")
        return cls(**cfg)


def _covered_seconds(intervals, t0: float, t1: float) -> float:
    """Length of the union of ``intervals`` [(s, e)] clipped to [t0, t1]."""
    segs = sorted((max(s, t0), min(e, t1)) for s, e in intervals if e > t0 and s < t1)
    total, cur_s, cur_e = 0.0, None, None
    for s, e in segs:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def smooth_tt_role_flags(event: Dict[str, Any], jchat: JChat,
                         roles: SmoothTTRoles = SmoothTTRoles()) -> List[str]:
    """Text-side turn-transition checks of one Smooth TT event (``detect_smooth_tt``
    output). Returns the failed checks: ``a_backchannel`` (a),
    ``b_talking_before`` (b, characters), ``b_backchannel`` (c)."""
    flags: List[str] = []
    n, r = backchannel_residual(event["a_text"])
    a_dur = max(event["a_duration"], 1e-3)
    if (r <= roles.a_max_resid_chars or r / a_dur < roles.a_min_resid_cps
            or r / max(n, 1) < roles.a_min_resid_ratio):
        flags.append("a_backchannel")
    t = event["a_end"]
    b_chars = [(c["start"], c["end"]) for c in jchat if c["speaker"] == "B"]
    if _covered_seconds(b_chars, t - roles.b_pre_window, t) > roles.b_pre_max:
        flags.append("b_talking_before")
    n, r = backchannel_residual(event["b_text"])
    if r <= roles.b_max_resid_chars or r / max(n, 1) < roles.b_min_resid_ratio:
        flags.append("b_backchannel")
    return flags


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
                     max_gap: float = 0.4, min_b_duration: float = 4.0,
                     roles: Optional[SmoothTTRoles] = None
                     ) -> List[Dict[str, Any]]:
    """Smooth inter-turn transitions A->B (desired TO=1).

    With ``roles.enabled`` the events failing the text checks of
    ``smooth_tt_role_flags`` are dropped; the kept events carry
    ``source_index`` (their index without the checks, so that results stay
    joinable with the unrestricted event set) and an empty ``role_flags`` list that the audio
    check (``event_boundaries.refine_events``) may fill."""
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
    if roles is not None and roles.enabled:
        kept = []
        for i, ev in enumerate(events):
            if smooth_tt_role_flags(ev, jchat, roles):
                continue
            kept.append({**ev, "source_index": i, "role_flags": []})
        events = kept
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

def detect(task: str, jchat: JChat,
           roles: Optional[SmoothTTRoles] = None) -> List[Dict[str, Any]]:
    """``DETECTORS[task]`` with the Smooth TT turn-transition checks when enabled."""
    if task == "smooth_tt" and roles is not None and roles.enabled:
        return detect_smooth_tt(jchat, roles=roles)
    return DETECTORS[task](jchat)


DETECTORS = {
    "pause": detect_pauses,
    "smooth_tt": detect_smooth_tt,
    "bc": detect_backchannels,
}
