"""``fdbja measure-contexts``: one system in several dialogue contexts.

What counts as well-timed depends on the dialogue: the task, the scene, the
persona of the system, the relationship of the speakers. :func:`fdbja.measure`
takes one set of recordings and one profile. This module takes several
**contexts** (a name each, e.g. ``chat`` and ``interview``), the recordings of
the system in each of them and, where available, a profile made from human
dialogues of the same context (``fdbja make-profile``), and reports:

* per context, the values of the system, of the human recordings and their
  difference (``gap``: system - human);
* per pair of contexts, the **shift** (context B - context A) of the human
  recordings and of the system, and whether the system moved in the direction
  in which the human recordings moved.

A system that times its speech the same way in every context can be close to
the human recordings of one context and still show no shift where the human
recordings show one; the second table is there to make that visible.

Unit of analysis: the recording. Every value is the mean over the recordings of
a per-recording value (:func:`fdbja.measurement.recording_features`), and every
interval a percentile bootstrap over recordings, so that the events of one
dialogue are not treated as independent observations. With fewer than
:data:`FEW_RECORDINGS` recordings on a side no interval is given and the shift
is not assessed.

What this is not: none of it was compared with listeners' judgement. The
listening tests behind ``pause_reaction_penalty`` used dialogues of one kind
only, so whether what listeners judge natural changes with the context is an
open question; the value is reported per context without a verdict. A shift
between two contexts also contains every other difference between the two
sets of recordings: other users, other topics, another recording set-up. Only
when the user's side is held fixed and nothing but the instruction to the
system changes does the system's shift describe the system alone.

The word "context" is used for what the dialogue is; ``conditions`` (the
``--condition`` labels of ``measure``) stay the free-form description of how a
recording was made.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from fdbja import measurement as ms

#: with fewer recordings than this on a side, no bootstrap interval is given
#: (a rule of thumb for a percentile interval, not a validated limit)
FEW_RECORDINGS = 5
DEFAULT_BOOTSTRAP = 2000
CONFIDENCE = 0.95
CONTEXTS_FORMAT = "fdbja-contexts/1"

#: per-recording values reported per context: name -> (label, unit).
#: ``shift`` lists those compared between contexts.
FEATURES: Dict[str, Tuple[str, str]] = {
    "latency_median": ("response latency, median", "s"),
    "no_response_rate": ("turn ends without a response", "rate"),
    "starts_in_user_speech_per_minute": ("starts during user speech, per minute", "per_min"),
    "overlap_share": ("both voiced, share of the recording", "rate"),
    "pause_voiced_rate": ("pauses with system voice", "rate"),
    "pause_reaction_penalty": ("pause reaction penalty (0-2)", "score"),
}
SHIFT_FEATURES = tuple(k for k in FEATURES if k != "pause_reaction_penalty")

#: verdicts of one feature in one pair of contexts
VERDICTS = {
    "followed": "the human recordings shift and the system shifts in the same direction",
    "opposite": "the human recordings shift and the system shifts in the other direction",
    "no_clear_system_shift": ("the human recordings shift; the interval of the system's "
                              "shift contains 0"),
    "system_only": ("the system shifts; the interval of the human shift contains 0"),
    "no_clear_human_shift": "neither interval excludes 0",
    "not_assessed": ("no interval on one side: fewer than "
                     f"{FEW_RECORDINGS} recordings with a value, or a profile "
                     "without per-recording values"),
}

LAYERS: Dict[str, Dict[str, Any]] = {
    "context_gap": {
        "title": "gap to the human recordings of the same context (system - human)",
        "keys": ["contexts.<name>.features.<feature>.gap"],
        "note": ("deviation from human behaviour in that context, on the mean over "
                 "recordings; not compared with listeners' judgement"),
    },
    "context_shift": {
        "title": "shift between contexts, human recordings and system",
        "keys": ["shifts[].features.<feature>", "summary"],
        "note": ("whether the system moves where the human recordings move; not "
                 "compared with listeners' judgement. A shift contains every "
                 "difference between the two sets of recordings (users, topics, "
                 "set-up), not only the behaviour of the system. Intervals are per "
                 "comparison, not corrected for the number of comparisons; "
                 "'no clear shift' is not evidence of no shift"),
    },
    "listener_judgement": {
        "title": "pause reaction penalty per context",
        "keys": ["contexts.<name>.features.pause_reaction_penalty"],
        "note": ("the rule compared with third-party listeners on dialogues of one "
                 "kind; whether listeners' judgement depends on the context was not "
                 "tested, so no verdict between contexts is given"),
    },
}

PathLike = ms.PathLike
Recordings = Union[PathLike, Sequence[PathLike]]


# ---------------------------------------------------------------------------
# Means over recordings and their bootstrap
# ---------------------------------------------------------------------------

def _column(rows: Optional[Sequence[Mapping[str, Any]]], key: str) -> Optional[np.ndarray]:
    if rows is None:
        return None
    return np.array([np.nan if r.get(key) is None else float(r[key]) for r in rows],
                    dtype=np.float64)


def _boot_means(v: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Means of ``n_boot`` resamples of the recordings (with replacement);
    recordings without a value stay in the resampling and are left out of the
    mean. ``nan`` when a resample holds no value."""
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    x = v[idx]
    ok = ~np.isnan(x)
    count = ok.sum(axis=1)
    total = np.where(ok, x, 0.0).sum(axis=1)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


class _Side:
    """The per-recording values of one side (system or human) of one context."""

    def __init__(self, name: str, side: str, rows: Optional[Sequence[Mapping[str, Any]]],
                 n_boot: int, seed: int):
        self.rows = rows
        self._key = f"{name}|{side}"
        self._n_boot, self._seed = n_boot, seed
        self._boot: Dict[str, Optional[np.ndarray]] = {}

    def values(self, feature: str) -> Optional[np.ndarray]:
        v = _column(self.rows, feature)
        return None if v is None else v[~np.isnan(v)]

    def boot(self, feature: str) -> Optional[np.ndarray]:
        if feature not in self._boot:
            v = _column(self.rows, feature)
            if v is None or int((~np.isnan(v)).sum()) < FEW_RECORDINGS or self._n_boot <= 0:
                self._boot[feature] = None
            else:
                # one stream per (context, side, feature): the draws do not depend
                # on the order or the number of the contexts
                rng = np.random.default_rng(
                    [self._seed, zlib.crc32(f"{self._key}|{feature}".encode("utf-8"))])
                self._boot[feature] = _boot_means(v, self._n_boot, rng)
        return self._boot[feature]

    def estimate(self, feature: str) -> Optional[Dict[str, Any]]:
        v = self.values(feature)
        if v is None:
            return None
        return {"value": ms._r(v.mean()) if len(v) else None, "n": int(len(v)),
                "interval": _interval(self.boot(feature))}


def _interval(draws: Optional[np.ndarray]) -> Optional[List[float]]:
    if draws is None:
        return None
    draws = draws[~np.isnan(draws)]
    if not len(draws):
        return None
    a = 100 * (1 - CONFIDENCE) / 2
    return [ms._r(np.percentile(draws, a)), ms._r(np.percentile(draws, 100 - a))]


def _clear(interval: Optional[List[float]]) -> Optional[bool]:
    """Whether an interval of a difference excludes 0 (``None`` = no interval)."""
    if interval is None:
        return None
    return bool(interval[0] > 0 or interval[1] < 0)


def _difference(minus: _Side, plus: _Side, feature: str) -> Optional[Dict[str, Any]]:
    """``plus - minus`` of the means over recordings, the two sides resampled
    independently."""
    a, b = minus.values(feature), plus.values(feature)
    if a is None or b is None:
        return None
    value = ms._r(b.mean() - a.mean()) if len(a) and len(b) else None
    da, db = minus.boot(feature), plus.boot(feature)
    interval = _interval(db - da) if da is not None and db is not None else None
    return {"value": value, "n": [int(len(a)), int(len(b))], "interval": interval,
            "clear": _clear(interval)}


def _verdict(human: Optional[Dict[str, Any]], system: Optional[Dict[str, Any]]) -> str:
    if (human is None or system is None or human["clear"] is None
            or system["clear"] is None):
        return "not_assessed"
    if not human["clear"]:
        return "system_only" if system["clear"] else "no_clear_human_shift"
    if not system["clear"]:
        return "no_clear_system_shift"
    same = (human["value"] > 0) == (system["value"] > 0)
    return "followed" if same else "opposite"


def _tally(features: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    verdicts = [f["verdict"] for f in features.values()]
    clear = sum(v in ("followed", "opposite", "no_clear_system_shift") for v in verdicts)
    followed = verdicts.count("followed")
    return {"n_features": len(verdicts),
            "n_clear_human_shifts": clear,
            "n_followed": followed,
            "n_opposite": verdicts.count("opposite"),
            "n_no_clear_system_shift": verdicts.count("no_clear_system_shift"),
            "n_system_only": verdicts.count("system_only"),
            "n_no_clear_human_shift": verdicts.count("no_clear_human_shift"),
            "n_not_assessed": verdicts.count("not_assessed"),
            "followed_rate": ms._ratio(followed, clear)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@dataclass
class ContextMeasurement:
    """Result of :func:`measure_contexts`.

    ``contexts``: per context, the pooled measurement of the system
    (``stats`` / ``score`` / ``signature`` of :func:`fdbja.measure`), the
    profile, and ``features``: per value the ``system`` and ``human``
    estimates (``value`` = mean over recordings, ``n`` = recordings with a
    value, ``interval``) and their ``gap``;
    ``shifts``: per pair of contexts and value, the ``human`` and ``system``
    shifts (``to`` - ``from``), the ``verdict`` (:data:`VERDICTS`) and
    ``ratio`` (system shift / human shift, when the human shift is clear);
    ``summary``: the verdicts counted over all pairs.
    """
    contexts: Dict[str, Dict[str, Any]]
    shifts: List[Dict[str, Any]]
    summary: Dict[str, Any]
    factor: Optional[str]
    baseline: Optional[str]
    config: Dict[str, Any]
    bootstrap: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": CONTEXTS_FORMAT,
            "comparability": ms.NOT_COMPARABLE,
            "factor": self.factor,
            "baseline": self.baseline,
            "unit": "recording (values are means over recordings)",
            "bootstrap": self.bootstrap,
            "layers": {k: dict(v) for k, v in LAYERS.items()},
            "verdicts": dict(VERDICTS),
            "features": {k: {"label": label, "unit": unit, "shift": k in SHIFT_FEATURES}
                         for k, (label, unit) in FEATURES.items()},
            "contexts": self.contexts,
            "shifts": self.shifts,
            "summary": self.summary,
            "config": self.config,
            "warnings": self.warnings,
        }

    def __str__(self) -> str:
        return format_context_report(self)


def _settings(config: ms.ConfigLike, profiles: Mapping[str, Dict[str, Any]],
              warnings: List[str]) -> ms.MeasureConfig:
    """One set of detection settings for every context: the given one, else
    the one the profiles were made with (they must agree)."""
    stored = {name: ms.as_config(p["meta"].get("detection")).to_dict()
              for name, p in profiles.items()}
    distinct = {repr(sorted(d.items())) for d in stored.values()}
    if config is not None:
        cfg = ms.as_config(config)
        if any(d != cfg.to_dict() for d in stored.values()):
            warnings.append("the detection settings differ from those a profile was made "
                            "with; the human values may not be comparable")
        return cfg
    if len(distinct) > 1:
        raise ValueError(
            "the profiles were made with different detection settings "
            f"({', '.join(sorted(stored))}); contexts can only be compared under one "
            "set of settings: rebuild the profiles with the same settings")
    return ms.as_config(next(iter(stored.values()))) if stored else ms.DEFAULT_MEASURE


def measure_contexts(recordings: Optional[Mapping[str, Recordings]] = None,
                     profiles: Optional[Mapping[str, Any]] = None, *,
                     factor: Optional[str] = None, baseline: Optional[str] = None,
                     config: ms.ConfigLike = None, user_channel: int = 0,
                     system_channel: Optional[int] = None,
                     n_boot: int = DEFAULT_BOOTSTRAP, seed: int = 0) -> ContextMeasurement:
    """One system measured in several contexts and compared with the human
    recordings of each.

    Args:
        recordings: context name -> the system's recordings in that context
            (a two-channel wav, a directory or a list of paths, as for
            :func:`fdbja.measure`).
        profiles: context name -> the profile of the human dialogues of that
            context (a path or a mapping, from ``fdbja make-profile``). A
            context may have recordings only (no human side) or a profile
            only (no system side: the human shifts alone are reported).
        factor: what the contexts differ in (``task``, ``situation``,
            ``persona``, ``relationship``, ...), copied to the output.
        baseline: compare every other context with this one only; ``None``
            = every pair, in the order given.
        config: detection settings for every context; ``None`` = those the
            profiles were made with (they must agree), else the defaults.
        n_boot, seed: bootstrap over recordings (``n_boot=0`` = no intervals).
    """
    recordings = dict(recordings or {})
    profs = {name: ms.load_profile(p) for name, p in (profiles or {}).items()}
    names = list(recordings) + [n for n in profs if n not in recordings]
    if len(names) < 2:
        raise ValueError("measure-contexts needs at least two contexts")
    if baseline is not None and baseline not in names:
        raise ValueError(f"baseline {baseline!r} is not one of the contexts {names}")
    warnings: List[str] = []
    cfg = _settings(config, profs, warnings)

    contexts: Dict[str, Dict[str, Any]] = {}
    sides: Dict[str, Dict[str, _Side]] = {}
    for name in names:
        prof = profs.get(name)
        entry: Dict[str, Any] = {"n_recordings": None, "signature": None, "stats": None,
                                 "score": None, "profile": None}
        system_rows = None
        if name in recordings:
            m = ms.measure(recordings[name], user_channel=user_channel,
                           system_channel=system_channel, profile=prof, config=cfg)
            pauses: Dict[int, List[Dict[str, Any]]] = {}
            for ev in m.events:
                if ev["type"] == "pause":
                    pauses.setdefault(ev["recording"], []).append(ev)
            system_rows = [ms.recording_features(r["stats"], pauses.get(i, []))
                           for i, r in enumerate(m.recordings)]
            score = dict(m.score or {})
            score.pop("reference", None)          # the profile's statistics: see `profile`
            entry.update(n_recordings=len(m.recordings), signature=m.signature,
                         stats=m.stats, score=score, quality=m.quality)
            warnings += [f"{name}: {w}" for w in m.warnings
                         if not w.startswith("the detection settings differ")]
        human_rows = None
        if prof is not None:
            human_rows = prof.get("reference_recordings")
            entry["profile"] = {"id": ms.profile_id(prof), "name": prof.get("name"),
                                "meta": prof["meta"], "norms": prof["norms"],
                                "n_recordings": prof["meta"].get("n_recordings")}
            if human_rows is None:
                warnings.append(
                    f"{name}: the profile carries no per-recording values (the built-in "
                    "profile, or one written by an earlier version); rebuild it with "
                    "`fdbja make-profile` to compare this context with human recordings")
        s = {"system": _Side(name, "system", system_rows, n_boot, seed),
             "human": _Side(name, "human", human_rows, n_boot, seed)}
        sides[name] = s
        entry["features"] = {
            f: {"system": s["system"].estimate(f), "human": s["human"].estimate(f),
                "gap": _difference(s["human"], s["system"], f)}
            for f in FEATURES}
        contexts[name] = entry

    pairs = ([(baseline, n) for n in names if n != baseline] if baseline is not None
             else [(a, b) for i, a in enumerate(names) for b in names[i + 1:]])
    shifts = []
    for a, b in pairs:
        feats: Dict[str, Dict[str, Any]] = {}
        for f in SHIFT_FEATURES:
            human = _difference(sides[a]["human"], sides[b]["human"], f)
            system = _difference(sides[a]["system"], sides[b]["system"], f)
            verdict = _verdict(human, system)
            ratio = None
            if verdict in ("followed", "opposite", "no_clear_system_shift"):
                ratio = ms._r(system["value"] / human["value"])
            feats[f] = {"human": human, "system": system, "verdict": verdict, "ratio": ratio}
        shifts.append({"from": a, "to": b, "features": feats, "summary": _tally(feats)})

    keys = ("n_features", "n_clear_human_shifts", "n_followed", "n_opposite",
            "n_no_clear_system_shift", "n_system_only", "n_no_clear_human_shift",
            "n_not_assessed")
    summary: Dict[str, Any] = {k: sum(s["summary"][k] for s in shifts) for k in keys}
    summary["n_pairs"] = len(shifts)
    summary["followed_rate"] = ms._ratio(summary["n_followed"],
                                         summary["n_clear_human_shifts"])
    return ContextMeasurement(
        contexts=contexts, shifts=shifts, summary=summary, factor=factor,
        baseline=baseline, config=cfg.to_dict(),
        bootstrap={"n": int(n_boot), "seed": int(seed), "confidence": CONFIDENCE,
                   "min_recordings": FEW_RECORDINGS,
                   "method": "percentile, recordings resampled within each context and side"},
        warnings=warnings)


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def _fmt(x: Optional[float], unit: str, signed: bool) -> str:
    if x is None:
        return "-"
    sign = "+" if signed else ""
    if unit == "s":
        return f"{x:{sign}.2f} s"
    if unit == "rate":
        return f"{100 * x:{sign}.1f} {'pt' if signed else '%'}"
    if unit == "per_min":
        return f"{x:{sign}.2f} /min"
    return f"{x:{sign}.3f}"


def _cell(est: Optional[Dict[str, Any]], unit: str, signed: bool = False) -> str:
    """``value [low, high]``; ``*`` after an interval of a difference that
    excludes 0."""
    if est is None or est["value"] is None:
        return "-"
    out = _fmt(est["value"], unit, signed)
    if est["interval"] is not None:
        lo, hi = (_fmt(v, unit, signed).rsplit(" ", 1)[0] if unit != "score"
                  else _fmt(v, unit, signed) for v in est["interval"])
        out += f" [{lo}, {hi}]"
        if est.get("clear"):
            out += " *"
    return out


def _table(head: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    width = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(head)]
    return ["  " + "  ".join(c.ljust(w) for c, w in zip(cells, width)).rstrip()
            for cells in [head, *rows]]


def format_context_report(c: ContextMeasurement) -> str:
    """The text printed by ``fdbja measure-contexts``."""
    names = list(c.contexts)
    boot = c.bootstrap
    out = [f"{len(names)} contexts" + (f" by {c.factor}" if c.factor else "") + ": "
           + ", ".join(names) + (f" (baseline: {c.baseline})" if c.baseline else ""),
           "values are means over recordings; [low, high] = "
           f"{100 * boot['confidence']:g} % bootstrap interval over recordings "
           f"({boot['n']} draws), given from {boot['min_recordings']} recordings; "
           "* = the interval of a difference excludes 0", ""]
    for name in names:
        e = c.contexts[name]
        prof = e["profile"]
        n = e["n_recordings"]
        out.append(f"context: {name}   system: "
                   + ("-" if n is None else f"{n} recording(s)"
                      + (f" (fewer than {boot['min_recordings']}: no interval)"
                         if n < boot["min_recordings"] else ""))
                   + "   human: "
                   + ("-" if prof is None
                      else f"{prof['id']} ({prof['n_recordings']} recording(s))"))
        rows = [[label, _cell(e["features"][f]["system"], unit),
                 _cell(e["features"][f]["human"], unit),
                 _cell(e["features"][f]["gap"], unit, signed=True)]
                for f, (label, unit) in FEATURES.items()]
        out += _table(["", "system", "human", "system - human"], rows)
        sc = e["score"] or {}
        if sc.get("latency_in_tolerance_rate") is not None:
            t0, t1 = sc["latency_tolerance"]
            own = (sc.get("reference_score") or {}).get("latency_in_tolerance_rate")
            out.append(f"  latency within {t0:+.2f} to {t1:+.2f} s (pooled events): "
                       f"{100 * sc['latency_in_tolerance_rate']:.1f} %"
                       + ("" if own is None
                          else f" (profile's own recordings: {100 * own:.1f} %)"))
        if e["signature"]:
            out.append(f"  signature: {e['signature']}")
        out.append("")
    for s in c.shifts:
        out.append(f"shift: {s['from']} -> {s['to']}")
        rows = []
        for f in SHIFT_FEATURES:
            label, unit = FEATURES[f]
            d = s["features"][f]
            rows.append([label, _cell(d["human"], unit, signed=True),
                         _cell(d["system"], unit, signed=True),
                         d["verdict"].replace("_", " ")
                         + ("" if d["ratio"] is None else f" (ratio {d['ratio']:+.2f})")])
        out += _table(["", "human", "system", "verdict"], rows)
        t = s["summary"]
        out.append(f"  followed {t['n_followed']} of {t['n_clear_human_shifts']} clear human "
                   f"shift(s); opposite {t['n_opposite']}, no clear system shift "
                   f"{t['n_no_clear_system_shift']}. Without a clear human shift: "
                   f"{t['n_no_clear_human_shift'] + t['n_system_only']} "
                   f"(the system shifts in {t['n_system_only']}); not assessed "
                   f"{t['n_not_assessed']}")
        out.append("")
    t = c.summary
    out.append(f"all pairs ({t['n_pairs']}): followed {t['n_followed']} of "
               f"{t['n_clear_human_shifts']} clear human shift(s)"
               + ("" if t["followed_rate"] is None
                  else f" ({100 * t['followed_rate']:.1f} %)"))
    for label, key in (("system - human", "context_gap"), ("shift", "context_shift"),
                       ("pause reaction penalty", "listener_judgement")):
        out += ms._note(f"{label}: {LAYERS[key]['note']}")
    out += [f"warning: {w}" for w in c.warnings]
    out += ms._note(ms.NOT_COMPARABLE)
    return "\n".join(out)
