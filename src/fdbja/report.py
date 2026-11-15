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


#: tasks ranked on the leaderboard page, each by its own metric
LEADERBOARD_TASKS = ("pause", "smooth_tt")
#: scoring conditions that must match for two results to be comparable
COMPARABLE_KEYS = ("boundaries", "asr", "system_activity", "smooth_tt_roles")


def desired_rate(row: Dict[str, Any]) -> Optional[float]:
    """Share of events with the desired behaviour of Full-Duplex-Bench: TOR
    when the desired TO is 1 (Smooth TT, its ranking), 1 - TOR when it is 0
    (Pause, where it is a reference value and not the ranking)."""
    tor = row.get("TOR")
    if tor is None:
        return None
    return float(tor) if row.get("desired_TO") == 1 else 1.0 - float(tor)


def _is_baseline(name: str, domain: Optional[str]) -> bool:
    return domain == "baseline" or "baseline" in name


def leaderboard_rows(results: Dict[str, Any], source: str = "") -> List[Dict[str, Any]]:
    """One row per model with the ranking metric of each task: ``pause`` is
    the pause reaction penalty (no-reaction rate + barge-in rate, 0-2, smaller
    is better; ``None`` when ``metrics.pause_response`` was not scored) with
    its two parts and the Pause TOR as a reference, ``smooth_tt`` is TOR."""
    models: Dict[str, Dict[str, Any]] = {}
    for r in results.get("summary", []):
        m = models.setdefault(r["model"], {"model": r["model"], "domain": r.get("domain"),
                                           "source": source})
        if r["task"] == "pause":
            m["pause"] = r.get("pause_response")
            m["pause_no_response"] = r.get("pause_no_response_rate")
            m["pause_interrupt"] = r.get("pause_interrupt_rate")
            m["pause_tor"] = r.get("TOR")
        elif r["task"] == "smooth_tt":
            m["smooth_tt"] = desired_rate(r)
    for m in models.values():
        m["baseline"] = _is_baseline(m["model"], m["domain"])
    return list(models.values())


def _ranks(rows: List[Dict[str, Any]], key: str, smaller_is_better: bool) -> Dict[int, int]:
    """Rank among the systems (baselines get none), best first; equal values
    share a rank. Keyed by ``id(row)`` since a model can appear twice (own
    result and reference)."""
    sys_rows = sorted((m for m in rows if not m["baseline"] and m.get(key) is not None),
                      key=lambda m: m[key] if smaller_is_better else -m[key])
    out: Dict[int, int] = {}
    for i, m in enumerate(sys_rows):
        prev = sys_rows[i - 1] if i else None
        out[id(m)] = out[id(prev)] if prev is not None and prev[key] == m[key] else i + 1
    return out


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
    """The ranks of the leaderboard page: Pause by the pause reaction penalty
    (smaller is better), Smooth TT by TOR (larger is better). The two tasks
    differ in scale and direction, so no combined rank is given; rows are
    listed in the Pause order and trivial baselines last without a rank. With
    ``reference`` (e.g. ``docs/results.json``) its rows are merged in, marked
    ``[ref]``, and differing scoring conditions are reported."""
    rows = leaderboard_rows(results, "")
    notes: List[str] = []
    if any(m.get("pause") is None and m.get("pause_tor") is not None for m in rows):
        notes += ["NOTE: the summary has no pause reaction penalty, so no Pause rank is given. "
                  "Score with metrics.pause_response enabled (it needs the output audio)."]
    if reference is not None:
        rows += leaderboard_rows(reference, "ref")
        diff = condition_mismatches(results, reference)
        if diff:
            notes += ["WARNING: scoring conditions differ from the reference; "
                      "the values are not comparable:"] + [f"  {d}" for d in diff]
    rank_p = _ranks(rows, "pause", True)
    rank_s = _ranks(rows, "smooth_tt", False)

    def key(m):
        p, s = m.get("pause"), m.get("smooth_tt")
        return (m["baseline"], p is None, p if p is not None else 0.0,
                -(s if s is not None else -1.0))

    def num(v):
        return "" if v is None else f"{v:.3f}"

    def pct(v):
        return "" if v is None else f"{100 * v:.1f}"

    header = (f"{'P':>3}  {'model':<28} {'penalty':>7} {'no-react':>8} {'barge-in':>8} "
              f"{'TOR(ref)':>8}  {'S':>3} {'smooth_tt':>9}")
    lines = [header, "-" * len(header)]
    for m in sorted(rows, key=key):
        name = m["model"] + (" [ref]" if m["source"] == "ref" else "")
        lines.append(f"{str(rank_p.get(id(m), '-')):>3}  {name:<28} {num(m.get('pause')):>7} "
                     f"{pct(m.get('pause_no_response')):>8} {pct(m.get('pause_interrupt')):>8} "
                     f"{num(m.get('pause_tor')):>8}  {str(rank_s.get(id(m), '-')):>3} "
                     f"{pct(m.get('smooth_tt')):>9}")
    lines += ["",
              "P = Pause rank by the reaction penalty (no-reaction rate + barge-in rate, 0-2, "
              "smaller is better);",
              "    no-react / barge-in in %, TOR(ref) = Pause TOR, a reference value that is not ranked",
              "S = Smooth TT rank by TOR (%), larger is better",
              "baselines are not ranked; no combined rank is given"]
    return "\n".join(lines + ([""] + notes if notes else []))
