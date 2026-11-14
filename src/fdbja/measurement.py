"""Timing of any two-channel recording: ``fdbja measure`` / :func:`measure`.

The benchmark (``score-offline``) scores a system on fixed events of a fixed
corpus and needs transcripts. This module is the other entry point: it takes a
recording of *any* dialogue between a user and a system, one speaker per
channel, and describes its timing from the audio alone. Only numpy and
soundfile are needed; no transcript, no annotation, no reference response.

What is detected (frame-energy voice activity, the detector of
``detection.boundaries: audio``; voiced runs separated by at most
``merge_gap`` are one utterance; utterances of one speaker separated by at
most ``pause_max_gap`` are one turn):

  * **turn end**: the end of a user turn of at least ``turn_min_seconds`` at
    which the system was not already speaking (at most
    ``pre_voiced_max_seconds`` of system voice in the ``pre_window_seconds``
    before it). The response is the first system utterance of at least
    ``response_min_seconds`` that starts within ``response_window`` of the
    turn end; its latency keeps the sign (negative = started before the user
    had finished);
  * **pause**: a silence of ``pause_min_gap`` to ``pause_max_gap`` inside a
    user turn of at least ``pause_min_turn_seconds``, with more than
    ``short_max_seconds`` of the turn on either side. System speech that
    overlaps the pause by at least ``min_overlap`` is ``short`` (at most
    ``short_max_seconds``) or ``long``. This is judged **by length only**: a
    short vocalisation is not necessarily a backchannel and a long one is not
    necessarily a take-over;
  * **start during user speech**: a system utterance that starts while the
    user is speaking and overlaps the user by at least ``min_overlap``
    (responses already counted at a turn end are listed separately).

The defaults are the rule constants of the benchmark (window -0.5 to +2.0 s,
short vocalisation up to 1.5 s, overlaps under 0.1 s ignored, pauses of 0.4
to 1.0 s in turns of 5 s or more); see ``docs/design/measure-entry.md`` for
where they differ from the benchmark's transcript-based detectors.

The output is descriptive. **It is not comparable with the leaderboard**: the
events are found in the recording itself instead of being the fixed events of
the evaluation corpus, and no transcript is used. The keys are named
differently from those of ``results.json`` for that reason.

A *profile* (:func:`load_profile`, :func:`make_profile`) adds a score: the
share of responses whose latency falls in the profile's tolerance range, the
distance from that range, and the pause reaction rule of
``metrics/pause_response.py`` applied to the detected pauses. The built-in
``generic`` profile is made of the rule constants only and is **not a norm
derived from human data**; a profile for a language, relationship, situation
and medium is built from human-human recordings with ``fdbja make-profile``.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from fdbja.data.event_boundaries import DEFAULT_VAD, VADConfig, voice_activity
from fdbja.data.event_detection import MIN_OVERLAP
from fdbja.data.system_activity import speech_segments
from fdbja.metrics.pause_response import (
    DEFAULT_PAUSE_RESPONSE,
    PauseResponseConfig,
    judge_event,
)

TOL = 1e-6
PROFILE_FORMAT = "fdbja-profile/1"
#: a profile built from fewer responses than this carries a warning (a rule of
#: thumb for the quantiles of the tolerance range, not a validated limit)
FEW_EVENTS = 30
NOT_COMPARABLE = ("not comparable with the leaderboard (events are detected in "
                  "this recording from the audio alone; no transcript, no fixed "
                  "evaluation corpus)")

#: fields of ``VADConfig`` that the voice activity detector reads (the others
#: belong to the boundary search of ``detection.boundaries: audio``)
VAD_KEYS = ("frame", "mode", "margin_db", "rel_db", "fixed_db", "min_snr_db",
            "floor_pct", "min_voiced", "min_hole")
#: thresholds of the pause reaction rule kept in a profile
PAUSE_RULE_KEYS = ("min_response_seconds", "persist_after_resume_seconds",
                   "pre_window_seconds", "pre_overlap_max_seconds")

Segment = Tuple[float, float]
PathLike = Union[str, "os.PathLike[str]"]


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MeasureConfig:
    """Detection settings; the defaults are the benchmark's rule constants."""
    vad: VADConfig = DEFAULT_VAD
    merge_gap: float = 0.3                  # voiced runs this close are one utterance
    short_max_seconds: float = 1.5          # a short vocalisation lasts at most this
    min_overlap: float = MIN_OVERLAP        # overlaps shorter than this are ignored
    # turn ends and responses
    turn_min_seconds: float = 1.5           # user turns shorter than this are not turn ends
    response_window: Tuple[float, float] = (-0.5, 2.0)   # response onset relative to the turn end
    response_min_seconds: float = 0.2       # a response lasts at least this
    pre_window_seconds: float = 2.0         # window before the turn end ...
    pre_voiced_max_seconds: float = 0.5     # ... in which the system may be voiced this long
    # pauses inside a turn
    pause_min_gap: float = 0.4
    pause_max_gap: float = 1.0              # a longer silence ends the turn
    pause_min_turn_seconds: float = 5.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["vad"] = {k: d["vad"][k] for k in VAD_KEYS}
        d["response_window"] = [float(x) for x in self.response_window]
        return d

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]]) -> "MeasureConfig":
        """From a mapping of the fields above; ``vad`` is a mapping of
        ``VADConfig`` fields. Unknown keys are an error."""
        cfg = dict(cfg or {})
        names = {f.name for f in fields(cls)}
        unknown = set(cfg) - names
        if unknown:
            raise ValueError(f"unknown measure settings: {sorted(unknown)} "
                             f"(allowed: {sorted(names)})")
        vad_cfg = dict(cfg.pop("vad", None) or {})
        bad = set(vad_cfg) - set(VAD_KEYS)
        if bad:
            raise ValueError(f"unknown measure vad settings: {sorted(bad)} "
                             f"(allowed: {list(VAD_KEYS)})")
        if "response_window" in cfg:
            lo, hi = cfg["response_window"]
            cfg["response_window"] = (float(lo), float(hi))
        return cls(vad=replace(DEFAULT_VAD, **vad_cfg), **cfg)


DEFAULT_MEASURE = MeasureConfig()
ConfigLike = Union[MeasureConfig, Dict[str, Any], None]


def as_config(config: ConfigLike) -> MeasureConfig:
    if isinstance(config, MeasureConfig):
        return config
    return MeasureConfig.from_dict(config)


def parse_settings(pairs: Sequence[str]) -> Dict[str, Any]:
    """``["turn_min_seconds=4", "vad.margin_db=12"]`` -> a settings mapping
    for :meth:`MeasureConfig.from_dict` (values are read as JSON, else kept
    as text; ``response_window=-0.5,2.0`` is also accepted)."""
    out: Dict[str, Any] = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"expected key=value, got {p!r}")
        key, raw = p.split("=", 1)
        try:
            val: Any = json.loads(raw)
        except json.JSONDecodeError:
            val = [float(x) for x in raw.split(",")] if "," in raw else raw
        if key.startswith("vad."):
            out.setdefault("vad", {})[key[4:]] = val
        else:
            out[key] = val
    return out


# ---------------------------------------------------------------------------
# Events of one recording
# ---------------------------------------------------------------------------

def _turns(utts: Sequence[Segment], max_gap: float) -> List[List[int]]:
    """Indices of the utterances of each turn (chained over ``max_gap``)."""
    turns: List[List[int]] = []
    for i, (s, _e) in enumerate(utts):
        if turns and s - utts[turns[-1][-1]][1] <= max_gap + TOL:
            turns[-1].append(i)
        else:
            turns.append([i])
    return turns


def _voiced_seconds(va: np.ndarray, dt: float, t0: float, t1: float) -> float:
    i0 = max(0, int(round(t0 / dt)))
    i1 = min(len(va), int(round(t1 / dt)))
    return float(va[i0:i1].sum()) * dt if i1 > i0 else 0.0


def _overlap(a: Segment, b: Segment) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def analyse(va_user: np.ndarray, va_system: np.ndarray, dt: float,
            cfg: MeasureConfig = DEFAULT_MEASURE,
            pause_rule: PauseResponseConfig = DEFAULT_PAUSE_RESPONSE
            ) -> Dict[str, Any]:
    """Events and totals of one recording from the frame-level voice activity
    of the two speakers (``dt`` seconds per frame)."""
    n = max(len(va_user), len(va_system))
    vu = np.zeros(n, dtype=bool)
    vs = np.zeros(n, dtype=bool)
    vu[:len(va_user)] = va_user
    vs[:len(va_system)] = va_system
    duration = n * dt
    u_utts = speech_segments(vu, dt, cfg.merge_gap)
    s_utts = speech_segments(vs, dt, cfg.merge_gap)
    lo, hi = cfg.response_window
    rule = replace(pause_rule, frame_rate=max(1, int(round(1.0 / dt))))
    times = np.arange(n) * dt

    turn_ends: List[Dict[str, Any]] = []
    pauses: List[Dict[str, Any]] = []
    skipped = {"short_turn": 0, "system_speaking": 0, "truncated": 0}
    responses = set()
    for turn in _turns(u_utts, cfg.pause_max_gap):
        t_s, t_e = u_utts[turn[0]][0], u_utts[turn[-1]][1]
        # --- pauses inside the turn
        if t_e - t_s >= cfg.pause_min_turn_seconds - TOL:
            for a, b in zip(turn, turn[1:]):
                g_s, g_e = u_utts[a][1], u_utts[b][0]
                if not (cfg.pause_min_gap - TOL <= g_e - g_s <= cfg.pause_max_gap + TOL):
                    continue
                if (g_s - t_s <= cfg.short_max_seconds + TOL
                        or t_e - g_e <= cfg.short_max_seconds + TOL):
                    continue
                inside = [u for u in s_utts if _overlap(u, (g_s, g_e)) >= cfg.min_overlap - TOL]
                longest = max((e - s for s, e in inside), default=0.0)
                outcome = ("silent" if not inside else
                           "long" if longest > cfg.short_max_seconds + TOL else "short")
                j = judge_event(vs, vu, times, g_s, g_e, rule)
                pauses.append({
                    "type": "pause", "start": g_s, "end": g_e,
                    "duration": round(g_e - g_s, 3), "outcome": outcome,
                    "system_speech": [list(u) for u in inside],
                    "no_reaction": j["no_response"], "barge_in": j["interrupt"],
                })
        # --- the end of the turn
        if t_e - t_s < cfg.turn_min_seconds - TOL:
            skipped["short_turn"] += 1
            continue
        if t_e + hi > duration + TOL:
            skipped["truncated"] += 1       # the response window runs past the recording
            continue
        if (_voiced_seconds(vs, dt, t_e - cfg.pre_window_seconds, t_e)
                > cfg.pre_voiced_max_seconds + TOL):
            skipped["system_speaking"] += 1
            continue
        found = None
        for k, (s, e) in enumerate(s_utts):
            if lo - TOL <= s - t_e <= hi + TOL and e - s >= cfg.response_min_seconds - TOL:
                found = k
                break
        if found is None:
            turn_ends.append({"type": "turn_end", "time": t_e, "outcome": "no_response",
                              "latency": None, "turn_seconds": round(t_e - t_s, 3)})
            continue
        responses.add(found)
        s, e = s_utts[found]
        turn_ends.append({
            "type": "turn_end", "time": t_e,
            "outcome": "early" if s - t_e < 0 else "after",
            "latency": round(s - t_e, 3), "turn_seconds": round(t_e - t_s, 3),
            "response": [s, e],
        })

    starts: List[Dict[str, Any]] = []
    for k, (s, e) in enumerate(s_utts):
        host = next((u for u in u_utts if u[0] < s - TOL and s < u[1] - TOL), None)
        if host is None or min(e, host[1]) - s < cfg.min_overlap - TOL:
            continue
        starts.append({
            "type": "start_in_user_speech", "time": s, "end": e,
            "duration": round(e - s, 3),
            "overlap": round(min(e, host[1]) - s, 3),
            "length": "long" if e - s > cfg.short_max_seconds + TOL else "short",
            "at_turn_end": k in responses,
        })
    return {
        "duration": round(duration, 3),
        "user_seconds": round(float(vu.sum()) * dt, 3),
        "system_seconds": round(float(vs.sum()) * dt, 3),
        "overlap_seconds": round(float((vu & vs).sum()) * dt, 3),
        "turn_ends": turn_ends, "pauses": pauses, "starts": starts,
        "skipped_turn_ends": skipped,
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _r(x: Optional[float]) -> Optional[float]:
    return None if x is None else round(float(x), 4)


def _ratio(a: float, b: float) -> Optional[float]:
    return _r(a / b) if b > 0 else None


def _q(x: np.ndarray, q: float) -> Optional[float]:
    return _r(np.percentile(x, q)) if len(x) else None


def summarise(analyses: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Descriptive statistics pooled over the events of ``analyses``."""
    ends = [e for a in analyses for e in a["turn_ends"]]
    pauses = [e for a in analyses for e in a["pauses"]]
    starts = [e for a in analyses for e in a["starts"] if not e["at_turn_end"]]
    at_end = sum(1 for a in analyses for e in a["starts"] if e["at_turn_end"])
    lat = np.array([e["latency"] for e in ends if e["latency"] is not None], dtype=np.float64)
    dur = sum(a["duration"] for a in analyses)
    user = sum(a["user_seconds"] for a in analyses)
    system = sum(a["system_seconds"] for a in analyses)
    both = sum(a["overlap_seconds"] for a in analyses)
    skipped: Dict[str, int] = {}
    for a in analyses:
        for k, v in a["skipped_turn_ends"].items():
            skipped[k] = skipped.get(k, 0) + v
    n_short = sum(1 for p in pauses if p["outcome"] == "short")
    n_long = sum(1 for p in pauses if p["outcome"] == "long")
    return {
        "n_recordings": len(analyses),
        "duration_seconds": _r(dur),
        "turn_end": {
            "n": len(ends),
            "n_response": int(len(lat)),
            "response_rate": _ratio(len(lat), len(ends)),
            "no_response_rate": _ratio(len(ends) - len(lat), len(ends)),
            "latency_median": _q(lat, 50),
            "latency_q1": _q(lat, 25),
            "latency_q3": _q(lat, 75),
            "latency_mean": _r(lat.mean()) if len(lat) else None,
            # share of the responses that started before the user had finished
            "early_start_rate": _r((lat < 0).mean()) if len(lat) else None,
            "skipped": skipped,
        },
        "pause": {
            "n": len(pauses),
            "voiced_rate": _ratio(n_short + n_long, len(pauses)),
            "short_rate": _ratio(n_short, len(pauses)),
            "long_rate": _ratio(n_long, len(pauses)),
            "n_short": n_short,
            "n_long": n_long,
        },
        "start_in_user_speech": {
            "n": len(starts),
            "per_minute_of_user_speech": _ratio(len(starts) * 60.0, user),
            "n_short": sum(1 for s in starts if s["length"] == "short"),
            "n_long": sum(1 for s in starts if s["length"] == "long"),
            "n_at_turn_end": at_end,
        },
        "overlap": {
            "seconds": _r(both),
            "share_of_recording": _ratio(both, dur),
            "share_of_user_speech": _ratio(both, user),
        },
        "speaking": {
            "user_seconds": _r(user),
            "system_seconds": _r(system),
            "user_share": _ratio(user, dur),
            "system_share": _ratio(system, dur),
            "system_share_of_speech": _ratio(system, user + system),
        },
    }


def score_against(analyses: Sequence[Dict[str, Any]], profile: Dict[str, Any]
                  ) -> Dict[str, Any]:
    """Scores of the detected events under ``profile``.

    * ``latency_in_tolerance_rate``: share of the turn ends whose response
      started inside the tolerance range (no response counts as outside);
    * ``latency_deviation_mean``: mean distance in seconds of the response
      latency from the range (0 inside it), over the responses;
    * ``pause_no_reaction_rate`` / ``pause_barge_in_rate`` /
      ``pause_reaction_penalty``: the rule of ``metrics/pause_response.py``
      on the pauses detected here (penalty = the sum of the two, 0-2, lower is
      better). Not the published ``pause_response``: the events and the voice
      activity detector differ.
    """
    tol = profile["norms"]["response_latency"]
    low, high = float(tol["low"]), float(tol["high"])
    ends = [e for a in analyses for e in a["turn_ends"]]
    pauses = [e for a in analyses for e in a["pauses"]]
    lat = np.array([e["latency"] for e in ends if e["latency"] is not None], dtype=np.float64)
    dev = np.maximum(0.0, np.maximum(low - lat, lat - high))
    no_reaction = [p["no_reaction"] for p in pauses]
    barge_in = [p["barge_in"] for p in pauses]
    out = {
        "profile": profile_id(profile),
        "latency_tolerance": [low, high],
        "latency_in_tolerance_rate": _ratio(float((dev <= TOL).sum()), len(ends)),
        "latency_deviation_mean": _r(dev.mean()) if len(dev) else None,
        "n_turn_ends": len(ends),
        "pause_no_reaction_rate": _r(np.mean(no_reaction)) if pauses else None,
        "pause_barge_in_rate": _r(np.mean(barge_in)) if pauses else None,
        "pause_reaction_penalty": (_r(np.mean(no_reaction) + np.mean(barge_in))
                                   if pauses else None),
        "n_pauses": len(pauses),
        "reference": profile.get("reference"),
    }
    if not profile.get("reference"):
        out["note"] = ("the profile carries no human reference: the tolerance range is "
                       "a rule constant, not a norm measured on human dialogues")
    return out


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

def _version() -> str:
    from fdbja import __version__
    return __version__


def _pause_rule_dict(rule: PauseResponseConfig = DEFAULT_PAUSE_RESPONSE) -> Dict[str, float]:
    return {k: getattr(rule, k) for k in PAUSE_RULE_KEYS}


def generic_profile() -> Dict[str, Any]:
    """The built-in profile: the rule constants of the toolkit and nothing
    else. Not derived from any corpus and not a norm of human behaviour; with
    it the tolerance range equals the response window, so
    ``latency_in_tolerance_rate`` equals the response rate."""
    cfg = DEFAULT_MEASURE
    return {
        "format": PROFILE_FORMAT,
        "name": "generic",
        "description": ("Rule constants of the toolkit only (response window, pause "
                        "reaction thresholds). Not derived from human data: use "
                        "`fdbja make-profile` on human-human recordings of the target "
                        "language, relationship, situation and medium for a norm."),
        "meta": {
            "language": None, "relationship": None, "situation": None, "medium": None,
            "source": "rule constants; no recordings",
            "n_recordings": 0,
            "n_events": {"turn_ends": 0, "responses": 0, "pauses": 0},
            "detection": cfg.to_dict(),
            "fdbja_version": _version(),
        },
        "norms": {
            "response_latency": {"low": cfg.response_window[0],
                                 "high": cfg.response_window[1],
                                 "basis": "response window (rule constant)"},
            "pause": _pause_rule_dict(),
        },
        "reference": None,
    }


def validate_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(profile, dict) or profile.get("format") != PROFILE_FORMAT:
        raise ValueError(f"not a profile of format {PROFILE_FORMAT!r}")
    try:
        tol = profile["norms"]["response_latency"]
        low, high = float(tol["low"]), float(tol["high"])
        rule = profile["norms"]["pause"]
        meta = profile["meta"]
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"malformed profile: {e!r}") from None
    if low > high:
        raise ValueError(f"profile tolerance range is empty: low {low} > high {high}")
    unknown = set(rule) - set(PAUSE_RULE_KEYS)
    if unknown:
        raise ValueError(f"unknown profile norms.pause keys: {sorted(unknown)}")
    if not isinstance(meta, dict):
        raise ValueError("malformed profile: meta must be a mapping")
    MeasureConfig.from_dict(meta.get("detection"))      # raises on unknown keys
    return profile


def load_profile(profile: Union[str, PathLike, Dict[str, Any]] = "generic"
                 ) -> Dict[str, Any]:
    """``"generic"``, the path of a profile JSON, or a profile mapping."""
    if isinstance(profile, dict):
        return validate_profile(profile)
    if str(profile) == "generic":
        return generic_profile()
    with open(profile, encoding="utf-8") as f:
        return validate_profile(json.load(f))


def profile_id(profile: Dict[str, Any]) -> str:
    """Name of a profile; any profile other than the built-in one carries the
    first 8 hex digits of the SHA-256 of its content."""
    name = str(profile.get("name") or "unnamed")
    if profile == generic_profile():
        return name
    blob = json.dumps(profile, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return f"{name}@{hashlib.sha256(blob).hexdigest()[:8]}"


def _pause_rule(profile: Optional[Dict[str, Any]]) -> PauseResponseConfig:
    if not profile:
        return DEFAULT_PAUSE_RESPONSE
    return replace(DEFAULT_PAUSE_RESPONSE, **profile["norms"]["pause"])


def make_profile(recordings: Union[PathLike, Sequence[PathLike]], *,
                 language: str, relationship: str, situation: str, medium: str,
                 source: Optional[str] = None, name: Optional[str] = None,
                 description: Optional[str] = None, config: ConfigLike = None,
                 tolerance_percentiles: Tuple[float, float] = (5.0, 95.0)
                 ) -> Dict[str, Any]:
    """A profile from two-channel recordings of human-human dialogues.

    Each recording is measured in both directions with the detector of
    :func:`measure` (each speaker in turn as the "user"), so the two speakers
    are treated alike. The tolerance range of the response latency is the
    ``tolerance_percentiles`` of the pooled latencies; the descriptive
    statistics are stored under ``reference``. ``language`` (e.g. ``ja``),
    ``relationship`` (e.g. ``strangers``), ``situation`` (e.g. ``chat``) and
    ``medium`` (e.g. ``video call``) say what the profile is a norm *of*;
    ``source`` names the recordings.
    """
    cfg = as_config(config)
    paths = _expand(recordings)
    analyses = []
    for p in paths:
        x, sr = _read(p)
        if x.shape[1] != 2:
            raise ValueError(f"{p}: make-profile needs two-channel recordings, "
                             f"got {x.shape[1]} channel(s)")
        va0, dt = _activity(x[:, 0], sr, cfg)
        va1, _ = _activity(x[:, 1], sr, cfg)
        analyses.append(analyse(va0, va1, dt, cfg))
        analyses.append(analyse(va1, va0, dt, cfg))
    lat = np.array([e["latency"] for a in analyses for e in a["turn_ends"]
                    if e["latency"] is not None], dtype=np.float64)
    if len(lat) == 0:
        raise ValueError("no response to a turn end was found in the recordings; "
                         "a profile needs at least one")
    p_lo, p_hi = tolerance_percentiles
    stats = summarise(analyses)
    # every recording was read twice (once per direction)
    stats["n_recordings"] = len(paths)
    stats["duration_seconds"] = _r(stats["duration_seconds"] / 2)
    warnings = []
    if len(lat) < FEW_EVENTS:
        warnings.append(f"tolerance range estimated from {len(lat)} responses "
                        f"(fewer than {FEW_EVENTS})")
    profile = {
        "format": PROFILE_FORMAT,
        "name": name or "-".join(str(x).replace(" ", "_")
                                 for x in (language, relationship, situation, medium)),
        "description": description or (
            "Measured on human-human recordings with `fdbja make-profile`, both "
            "speakers of each recording in turn as the user."),
        "meta": {
            "language": language, "relationship": relationship,
            "situation": situation, "medium": medium,
            "source": source,
            "n_recordings": len(paths),
            "n_events": {"turn_ends": stats["turn_end"]["n"],
                         "responses": int(len(lat)),
                         "pauses": stats["pause"]["n"]},
            "detection": cfg.to_dict(),
            "directions": "both speakers as the user",
            "fdbja_version": _version(),
            "warnings": warnings,
        },
        "norms": {
            "response_latency": {
                "low": _r(np.percentile(lat, p_lo)),
                "high": _r(np.percentile(lat, p_hi)),
                "basis": f"percentiles {p_lo:g} and {p_hi:g} of the human response latency",
            },
            # thresholds of the reaction rule: fixed constants, not estimated here
            "pause": _pause_rule_dict(),
        },
        "reference": {k: stats[k] for k in ("turn_end", "pause", "start_in_user_speech",
                                            "overlap", "speaking")},
    }
    return validate_profile(profile)


def write_profile(profile: Dict[str, Any], path: PathLike) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=1)
        f.write("\n")
    return str(path)


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

def _changed(cur: Dict[str, Any], default: Dict[str, Any]) -> str:
    def fmt(v: Any) -> str:
        return ",".join(f"{x:g}" for x in v) if isinstance(v, list) else f"{v}"
    diff = [f"{k}={fmt(cur[k])}" for k in sorted(cur) if cur[k] != default[k]]
    return ";".join(diff)


def signature(cfg: MeasureConfig = DEFAULT_MEASURE,
              profile: Optional[Dict[str, Any]] = None,
              channels: str = "user=0,sys=1") -> str:
    """One-line statement of how the numbers were made, to be quoted with
    them (in the manner of sacreBLEU signatures)::

        fdbja:0.2.0|measure|vad:energy(p56)|events:default|profile:generic|ch:user=0,sys=1

    Settings that differ from the defaults are spelled out, e.g.
    ``vad:energy(p56;margin_db=12.0)`` or ``events:custom(turn_min_seconds=4.0)``.
    """
    cur, default = cfg.to_dict(), DEFAULT_MEASURE.to_dict()
    vad_diff = _changed({k: v for k, v in cur["vad"].items() if k != "mode"},
                        default["vad"])
    cur_ev = {k: v for k, v in cur.items() if k != "vad"}
    ev_diff = _changed(cur_ev, default)
    vad = f"vad:energy({cur['vad']['mode']}{';' + vad_diff if vad_diff else ''})"
    events = f"events:custom({ev_diff})" if ev_diff else "events:default"
    prof = f"profile:{profile_id(profile) if profile else 'none'}"
    return "|".join([f"fdbja:{_version()}", "measure", vad, events, prof, f"ch:{channels}"])


def text_hash(text: Union[str, bytes]) -> str:
    """``sha256:<12 hex digits>`` of a text, to record a prompt as a condition
    without publishing it."""
    data = text.encode("utf-8") if isinstance(text, str) else text
    return "sha256:" + hashlib.sha256(data).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

def _read(path: PathLike) -> Tuple[np.ndarray, int]:
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return x, int(sr)


def _activity(x: np.ndarray, sr: int, cfg: MeasureConfig) -> Tuple[np.ndarray, float]:
    """Voice activity of a mono signal and the frame length in seconds. The
    signal is used at its own sampling rate (no resampling)."""
    n = int(round(sr * cfg.vad.frame))
    if n < 1:
        raise ValueError(f"sampling rate {sr} Hz is too low for {cfg.vad.frame} s frames")
    return voice_activity(np.asarray(x), sr, cfg.vad), n / sr


def _expand(src: Union[PathLike, Sequence[PathLike]]) -> List[str]:
    """Recordings named by a wav path, a directory (every ``*.wav`` under it,
    sorted) or a list of paths."""
    if isinstance(src, (str, os.PathLike)):
        src = [src]
    out: List[str] = []
    for p in src:
        p = str(p)
        if os.path.isdir(p):
            found = sorted(os.path.join(root, f) for root, _d, files in os.walk(p)
                           for f in files if f.lower().endswith(".wav"))
            if not found:
                raise FileNotFoundError(f"no .wav file under {p}")
            out += found
        elif os.path.isfile(p):
            out.append(p)
        else:
            raise FileNotFoundError(p)
    return out


def _channels(n_channels: int, user: int, system: Optional[int], what: str
              ) -> Tuple[int, int]:
    if n_channels < 2:
        raise ValueError(f"{what}: one channel only; pass the system's recording as "
                         "the second input (two mono signals)")
    if system is None:
        if n_channels != 2:
            raise ValueError(f"{what}: {n_channels} channels; name the system channel")
        system = 1 - user
    if not (0 <= user < n_channels and 0 <= system < n_channels) or user == system:
        raise ValueError(f"{what}: channels user={user}, system={system} do not fit "
                         f"{n_channels} channels")
    return user, system


def _load(audio: Any, system: Any, sample_rate: Optional[int], user_channel: int,
          system_channel: Optional[int], cfg: MeasureConfig
          ) -> Tuple[np.ndarray, np.ndarray, float, str, str]:
    """``(va_user, va_system, dt, name, channel description)`` of one input."""
    is_array = isinstance(audio, np.ndarray)
    if is_array:
        if sample_rate is None:
            raise ValueError("sample_rate is required with arrays")
        x, sr, name = (audio[:, None] if audio.ndim == 1 else audio), int(sample_rate), "array"
        if x.ndim != 2 or (x.shape[1] > 8 and x.shape[0] <= 8):
            raise ValueError(f"expected an array of shape (samples, channels), got {audio.shape}")
    else:
        x, sr = _read(audio)
        name = os.path.basename(str(audio))
    if system is None:
        u, s = _channels(x.shape[1], user_channel, system_channel, name)
        va_u, dt = _activity(x[:, u], sr, cfg)
        va_s, _ = _activity(x[:, s], sr, cfg)
        return va_u, va_s, dt, name, f"user={u},sys={s}"
    if isinstance(system, np.ndarray):
        y, sr_y = (system[:, None] if system.ndim == 1 else system), sr
    else:
        y, sr_y = _read(system)
        name += "+" + os.path.basename(str(system))
    if x.shape[1] != 1 or y.shape[1] != 1:
        raise ValueError("with two inputs each must be mono "
                         f"(got {x.shape[1]} and {y.shape[1]} channels)")
    va_u, dt = _activity(x[:, 0], sr, cfg)
    va_s, dt_s = _activity(y[:, 0], sr_y, cfg)
    if abs(dt - dt_s) > 1e-9:
        raise ValueError(f"the two inputs give different frame lengths ({sr} Hz and "
                         f"{sr_y} Hz); resample one of them")
    return va_u, va_s, dt, name, "user=input1,sys=input2"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@dataclass
class Measurement:
    """Result of :func:`measure`.

    ``stats``: descriptive statistics pooled over the recordings;
    ``score``: scores under the profile (``None`` without one);
    ``recordings``: name, duration and statistics of each recording;
    ``events``: every detected event (``recording`` = its index);
    ``signature``: how the numbers were made (:func:`signature`);
    ``conditions``: what was recorded, as given by the caller.
    """
    stats: Dict[str, Any]
    score: Optional[Dict[str, Any]]
    recordings: List[Dict[str, Any]]
    events: List[Dict[str, Any]]
    signature: str
    conditions: Dict[str, Any]
    config: Dict[str, Any]
    profile: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signature": self.signature,
            "conditions": self.conditions,
            "comparability": NOT_COMPARABLE,
            "stats": self.stats,
            "score": self.score,
            "recordings": self.recordings,
            "events": self.events,
            "config": self.config,
            "profile": self.profile,
            "warnings": self.warnings,
        }

    def __str__(self) -> str:
        return format_report(self)


def measure(audio: Any, system: Any = None, *, sample_rate: Optional[int] = None,
            user_channel: int = 0, system_channel: Optional[int] = None,
            profile: Union[str, PathLike, Dict[str, Any], None] = None,
            config: ConfigLike = None,
            conditions: Optional[Dict[str, Any]] = None) -> Measurement:
    """Timing statistics of a two-channel dialogue recording.

    Args:
        audio: a two-channel wav (channel ``user_channel`` = the user, the
            other = the system), a directory or list of such files (pooled),
            or an array of shape ``(samples, 2)`` with ``sample_rate``. With
            ``system`` given, the user's mono wav or array.
        system: the system's mono wav or array (two mono inputs).
        profile: ``"generic"``, a profile JSON path or mapping; adds
            ``score``. ``None`` = descriptive statistics only.
        config: :class:`MeasureConfig` or a mapping of its fields. ``None``
            uses the detection settings stored in the profile (the defaults
            without a profile), so that the recording is measured the way the
            profile was.
        conditions: free-form description of what was recorded (model, voice,
            prompt hash, network, ...), copied to the output.
    """
    prof = load_profile(profile) if profile is not None else None
    warnings: List[str] = []
    if config is None:
        cfg = as_config(prof["meta"].get("detection") if prof else None)
    else:
        cfg = as_config(config)
        if prof and cfg.to_dict() != as_config(prof["meta"].get("detection")).to_dict():
            warnings.append("the detection settings differ from those the profile was "
                            "made with; the tolerance range may not apply")
    rule = _pause_rule(prof)

    if isinstance(audio, np.ndarray) or system is not None:
        inputs: List[Any] = [audio]
    else:
        inputs = list(_expand(audio))
    analyses, recordings, events, chans = [], [], [], set()
    for i, item in enumerate(inputs):
        va_u, va_s, dt, name, ch = _load(item, system, sample_rate, user_channel,
                                         system_channel, cfg)
        a = analyse(va_u, va_s, dt, cfg, rule)
        analyses.append(a)
        chans.add(ch)
        recordings.append({"name": name, "duration_seconds": a["duration"],
                           "stats": summarise([a])})
        for ev in a["turn_ends"] + a["pauses"] + a["starts"]:
            events.append({"recording": i, **ev})
        if a["user_seconds"] == 0 or a["system_seconds"] == 0:
            who = "user" if a["user_seconds"] == 0 else "system"
            warnings.append(f"{name}: no voice detected on the {who} channel")
    stats = summarise(analyses)
    if stats["turn_end"]["n"] == 0:
        warnings.append("no turn end detected; check the channel assignment "
                        "(user_channel) and the recording level")
    channels = chans.pop() if len(chans) == 1 else "mixed"
    return Measurement(
        stats=stats,
        score=score_against(analyses, prof) if prof else None,
        recordings=recordings,
        events=events,
        signature=signature(cfg, prof, channels),
        conditions=dict(conditions or {}),
        config=cfg.to_dict(),
        profile=({"id": profile_id(prof), "name": prof.get("name"),
                  "description": prof.get("description"), "meta": prof["meta"],
                  "norms": prof["norms"]} if prof else None),
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def _pct(x: Optional[float]) -> str:
    return "-" if x is None else f"{100 * x:.1f} %"


def _sec(x: Optional[float]) -> str:
    return "-" if x is None else f"{x:+.2f} s"


def _recording_table(m: Measurement) -> List[str]:
    cols = ["recording", "dur_s", "turn_ends", "resp_rate", "lat_median",
            "pauses", "voiced", "starts", "overlap"]
    rows = []
    for r in m.recordings + [{"name": "ALL", "duration_seconds": m.stats["duration_seconds"],
                              "stats": m.stats}]:
        s = r["stats"]
        rows.append([r["name"], f"{r['duration_seconds']:.1f}", str(s["turn_end"]["n"]),
                     _pct(s["turn_end"]["response_rate"]), _sec(s["turn_end"]["latency_median"]),
                     str(s["pause"]["n"]), _pct(s["pause"]["voiced_rate"]),
                     str(s["start_in_user_speech"]["n"]),
                     _pct(s["overlap"]["share_of_recording"])])
    width = [max(len(c), *(len(r[i]) for r in rows)) for i, c in enumerate(cols)]

    def line(cells: Sequence[str]) -> str:
        return "  ".join(c.ljust(w) if i == 0 else c.rjust(w)
                         for i, (c, w) in enumerate(zip(cells, width)))
    return [line(cols), "-" * len(line(cols))] + [line(r) for r in rows] + [""]


def format_report(m: Measurement) -> str:
    """The table printed by ``fdbja measure``."""
    s, cfg = m.stats, m.config
    te, pa, st, ov, sp = (s["turn_end"], s["pause"], s["start_in_user_speech"],
                          s["overlap"], s["speaking"])
    lo, hi = cfg["response_window"]
    short = cfg["short_max_seconds"]
    out = [f"{s['n_recordings']} recording(s), {s['duration_seconds']:.1f} s", ""]
    if len(m.recordings) > 1:
        out += _recording_table(m)

    def row(label: str, value: str) -> str:
        return f"  {label:<50}{value}"
    out += [
        f"user turn ends (n = {te['n']})",
        row(f"response within {lo:+.1f} to {hi:+.1f} s",
            f"{te['n_response']} / {te['n']} ({_pct(te['response_rate'])})"),
        row("latency, median [q1, q3]",
            f"{_sec(te['latency_median'])} [{_sec(te['latency_q1'])}, {_sec(te['latency_q3'])}]"),
        row("responses started before the user's end", _pct(te["early_start_rate"])),
        row("no response", _pct(te["no_response_rate"])),
        f"pauses inside user turns (n = {pa['n']})",
        row("system voiced in the pause", _pct(pa["voiced_rate"])),
        row(f"  short (<= {short:g} s) / long, by length only",
            f"{pa['n_short']} / {pa['n_long']}"),
        f"system starts during user speech (n = {st['n']})",
        row("per minute of user speech", "-" if st["per_minute_of_user_speech"] is None
            else f"{st['per_minute_of_user_speech']:.2f}"),
        row(f"  short (<= {short:g} s) / long", f"{st['n_short']} / {st['n_long']}"),
        "overlap and speaking time",
        row("both voiced", f"{ov['seconds']:.1f} s ({_pct(ov['share_of_recording'])} of the recording)"),
        row("speaking time, user / system",
            f"{_pct(sp['user_share'])} / {_pct(sp['system_share'])}"),
    ]
    if m.score:
        sc = m.score
        t0, t1 = sc["latency_tolerance"]
        out += [
            "", f"score under profile {sc['profile']}",
            row(f"latency within {t0:+.2f} to {t1:+.2f} s", _pct(sc["latency_in_tolerance_rate"])),
            row("mean distance from that range", "-" if sc["latency_deviation_mean"] is None
                else f"{sc['latency_deviation_mean']:.3f} s"),
            row("pauses: no reaction / barge-in",
                f"{_pct(sc['pause_no_reaction_rate'])} / {_pct(sc['pause_barge_in_rate'])}"),
            row("pauses: reaction penalty (0-2, lower is better)",
                "-" if sc["pause_reaction_penalty"] is None
                else f"{sc['pause_reaction_penalty']:.3f}"),
        ]
        ref = (sc.get("reference") or {}).get("turn_end")
        if ref:
            out.append(row("profile reference: latency median [q1, q3]",
                           f"{_sec(ref['latency_median'])} [{_sec(ref['latency_q1'])}, "
                           f"{_sec(ref['latency_q3'])}] (n = {ref['n_response']})"))
        if sc.get("note"):
            out.append(f"  note: {sc['note']}")
    out += ["", f"signature: {m.signature}"]
    if m.conditions:
        out.append("conditions: " + "; ".join(f"{k}={v}" for k, v in m.conditions.items()))
    out += [f"warning: {w}" for w in m.warnings]
    out.append(f"note: {NOT_COMPARABLE}")
    return "\n".join(out)
