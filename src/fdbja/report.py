"""Result exporters: JSON file, a printed summary table and the leaderboard
ranking (``summary.json`` / ``summary.csv`` are written by :mod:`fdbja.summary`)."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


def write_json(results: Dict[str, Any], out_dir: str,
               name: str = "results.json") -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    return path


def format_table(results: Dict[str, Any]) -> str:
    cols = ["task", "model", "n", "n_clips", "domain", "TOR", "TOR_ci",
            "lat_median", "overlap_rate",
            "bc_freq", "human_freq", "jsd", "bc_any_rate", "jsd_cond",
            "event_mean_dkl", "AUC", "score"]
    rows = results.get("summary", [])
    # drop columns that are empty for every row (e.g. bc / VAP when disabled)
    cols = [c for c in cols if c in ("task", "model")
            or any(r.get(c) is not None for r in rows)]
    header = " | ".join(f"{c:>14}" if c not in ("task", "model") else f"{c:<16}"
                        for c in cols)
    lines = [header, "-" * len(header)]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            v = "" if v is None else v
            if isinstance(v, list):
                v = "[" + ", ".join(f"{x:.3f}" for x in v) + "]"
            cells.append(f"{str(v):<16}" if c in ("task", "model")
                         else f"{str(v):>14}")
        lines.append(" | ".join(cells))
    cond = [f"{k}={results[k]}" for k in ("boundaries", "system_activity", "asr",
                                          "vap_checkpoint") if results.get(k)]
    if cond:
        lines += ["", "conditions: " + "; ".join(cond)]
    return "\n".join(lines)


#: tasks averaged into the overall agreement (as on the leaderboard page)
LEADERBOARD_TASKS = ("pause", "smooth_tt")
#: scoring conditions that must match for two results to be comparable
COMPARABLE_KEYS = ("boundaries", "asr", "system_activity", "smooth_tt_roles")


def desired_rate(row: Dict[str, Any]) -> Optional[float]:
    """Share of events with the desired behaviour: TOR when the desired TO is
    1 (Smooth TT), 1 - TOR when it is 0 (Pause); the leaderboard's ranking."""
    tor = row.get("TOR")
    if tor is None:
        return None
    return float(tor) if row.get("desired_TO") == 1 else 1.0 - float(tor)


def _is_baseline(name: str, domain: Optional[str]) -> bool:
    return domain == "baseline" or "baseline" in name


def leaderboard_rows(results: Dict[str, Any], source: str = "") -> List[Dict[str, Any]]:
    """One row per model: per-task desired-behaviour rate and their mean."""
    models: Dict[str, Dict[str, Any]] = {}
    for r in results.get("summary", []):
        m = models.setdefault(r["model"], {"model": r["model"], "domain": r.get("domain"),
                                           "source": source})
        m[r["task"]] = desired_rate(r)
    for m in models.values():
        vals = [m[t] for t in LEADERBOARD_TASKS if m.get(t) is not None]
        m["overall"] = sum(vals) / len(vals) if vals else None
        m["baseline"] = _is_baseline(m["model"], m["domain"])
    return list(models.values())


def _condition_value(v: Any) -> Any:
    """The value of a scoring condition without a trailing description:
    ``docs/results.json`` records ``"audio (human-side VAD; ...)"`` where a
    summary written by ``score-offline`` has ``"audio"``."""
    if isinstance(v, str) and v.strip():
        return v.split(None, 1)[0].rstrip(".,;:")
    return v


def condition_mismatches(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    """Scoring conditions recorded in both results that differ."""
    out = []
    for k in COMPARABLE_KEYS:
        if k in a and k in b and _condition_value(a[k]) != _condition_value(b[k]):
            out.append(f"{k}: {a[k]!r} vs {b[k]!r}")
    return out


def format_leaderboard(results: Dict[str, Any],
                       reference: Optional[Dict[str, Any]] = None) -> str:
    """Ranking by the desired-behaviour rate (mean over Pause and Smooth TT);
    trivial baselines are listed last without a rank. With ``reference``
    (e.g. ``docs/results.json``) its rows are merged in, marked ``[ref]``,
    and differing scoring conditions are reported."""
    rows = leaderboard_rows(results, "")
    notes: List[str] = []
    if reference is not None:
        rows += leaderboard_rows(reference, "ref")
        diff = condition_mismatches(results, reference)
        if diff:
            notes = ["WARNING: scoring conditions differ from the reference; "
                     "the values are not comparable:"] + [f"  {d}" for d in diff]

    def key(m):
        return -(m["overall"] if m["overall"] is not None else -1.0)

    systems = sorted([m for m in rows if not m["baseline"]], key=key)
    bases = sorted([m for m in rows if m["baseline"]], key=key)

    def pct(v):
        return "" if v is None else f"{100 * v:5.1f}"

    header = f"{'rank':>4}  {'model':<28} {'overall':>7} {'pause':>7} {'smooth_tt':>9}"
    lines = [header, "-" * len(header)]
    for i, m in enumerate(systems + bases, 1):
        rank = "" if m["baseline"] else str(i)
        name = m["model"] + (" [ref]" if m["source"] == "ref" else "")
        lines.append(f"{rank:>4}  {name:<28} {pct(m['overall']):>7} "
                     f"{pct(m.get('pause')):>7} {pct(m.get('smooth_tt')):>9}")
    lines += ["", "desired-behaviour rate (%): Pause = 1 - TOR, Smooth TT = TOR; "
              "baselines are not ranked"]
    return "\n".join(lines + ([""] + notes if notes else []))
