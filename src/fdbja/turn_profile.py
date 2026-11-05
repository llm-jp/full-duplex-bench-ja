"""Turn-taking profile of each system from a ``score-offline`` results.json.

``analyze-profile`` sorts every event into an outcome and writes

* ``profile.csv`` / ``profile.md``: one row per system. Pause: share of
  pauses the system talked through (``pause_takeover``, = TOR), answered with
  backchannel-sized speech only (``pause_short``) or left silent
  (``pause_silent``). Smooth TT: share of turn ends the system took
  (``stt_taken``, = TOR), of which started before A had finished
  (``stt_overlap``), and left unanswered (``stt_no_response``), with the
  quartiles of the signed response onset.
* ``events.csv``: one row per event and system: event time, judgement
  window, the system's utterances near it, the outcome and the reason.
* ``profile.png`` / ``profile.svg`` (needs ``matplotlib``, the ``analysis``
  extra): Pause takeover rate against Smooth TT no-response rate, one point
  per system. Top left = stays silent (misses turns), bottom right = talks
  over pauses, bottom left = desired; the label carries the median onset,
  which separates slow responders.

The outcomes follow the TOR rules of ``metrics/tor.py`` (backchannel limit
6 characters and 1.5 s, overlaps under 0.1 s ignored; a turn is taken by an
utterance of 3 or more characters starting within -0.5 to +2.0 s of A's end).
Needs results written by this version of ``score-offline`` (per-event
``sys_segments``).
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

BC_MAX_CHARS = 6
BC_MAX_DURATION = 1.5
MIN_OVERLAP = 0.1
MIN_RESPONSE_CHARS = 3
STT_WINDOW = (-0.5, 2.0)


def _fmt_seg(seg) -> str:
    s, e, n = seg
    return f"{s:.2f}-{e:.2f}s({n})"


def classify_pause(row: Dict[str, Any]) -> Tuple[str, str]:
    s, e = row["t_start"], row["t_end"]
    short = None
    for seg in row.get("sys_segments", []):
        us, ue, n = seg
        ov = min(ue, e) - max(us, s)
        if ov <= 0 or ov < MIN_OVERLAP:
            continue
        if n > BC_MAX_CHARS or (ue - us) > BC_MAX_DURATION:
            return "takeover", (f"utterance {_fmt_seg(seg)} overlaps the pause by {ov:.2f} s "
                                f"and exceeds a backchannel (<= {BC_MAX_CHARS} chars, "
                                f"<= {BC_MAX_DURATION} s)")
        short = short or seg
    if short:
        return "short", f"only backchannel-sized speech in the pause: {_fmt_seg(short)}"
    return "silent", f"no utterance overlaps the pause by >= {MIN_OVERLAP} s"


def classify_smooth_tt(row: Dict[str, Any]) -> Tuple[str, str]:
    a_end = row["t_start"]
    lo, hi = STT_WINDOW
    later = None
    for seg in row.get("sys_segments", []):
        us, _ue, n = seg
        d = us - a_end
        if lo <= d <= hi and n >= MIN_RESPONSE_CHARS:
            if d < 0:
                return "overlap", (f"utterance {_fmt_seg(seg)} starts {-d:.2f} s before "
                                   "A's end")
            return "taken", f"utterance {_fmt_seg(seg)} starts {d:.2f} s after A's end"
        if d > hi and later is None:
            later = (d, seg)
    why = (f"no utterance of >= {MIN_RESPONSE_CHARS} chars starts within "
           f"{lo:+.1f}..{hi:+.1f} s of A's end")
    if later:
        why += f" (next starts {later[0]:+.2f} s: {_fmt_seg(later[1])})"
    return "no_response", why


CLASSIFIERS = {"pause": classify_pause, "smooth_tt": classify_smooth_tt}
WINDOWS = {
    "pause": lambda r: (r["t_start"], r["t_end"]),
    "smooth_tt": lambda r: (r["t_start"] + STT_WINDOW[0], r["t_start"] + STT_WINDOW[1]),
}


def event_rows(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for r in results.get("events", []):
        task = r["task"]
        if task not in CLASSIFIERS:
            continue
        if "sys_segments" not in r or "t_start" not in r:
            raise ValueError("results.json lacks per-event sys_segments / t_start: "
                             "re-run score-offline with this version")
        outcome, reason = CLASSIFIERS[task](r)
        w0, w1 = WINDOWS[task](r)
        out.append({
            "task": task, "model": r["model"], "clip": r["clip"],
            "event_index": r["event_index"],
            "event_start": round(r["t_start"], 3), "event_end": round(r["t_end"], 3),
            "window_start": round(w0, 3), "window_end": round(w1, 3),
            "TO": r["TO"], "onset": None if r.get("onset") is None else round(r["onset"], 3),
            "outcome": outcome, "reason": reason,
            "system_segments": " ".join(_fmt_seg(s) for s in r["sys_segments"]),
        })
    return out


def _rate(xs: List[str], *keys: str) -> Optional[float]:
    return round(sum(x in keys for x in xs) / len(xs), 4) if xs else None


def profile_rows(events: List[Dict[str, Any]], results: Dict[str, Any],
                 takeover_hi: float = 0.5, no_response_hi: float = 0.5,
                 slow_onset: float = 1.0) -> List[Dict[str, Any]]:
    models = list(dict.fromkeys(e["model"] for e in events))
    domain = {(s["task"], s["model"]): s.get("domain") for s in results.get("summary", [])}
    rows = []
    for m in models:
        pz = [e["outcome"] for e in events if e["model"] == m and e["task"] == "pause"]
        st = [e for e in events if e["model"] == m and e["task"] == "smooth_tt"]
        so = [e["outcome"] for e in st]
        on = np.array([e["onset"] for e in st if e["onset"] is not None])

        def q(p, on=on):
            return round(float(np.percentile(on, p)), 3) if len(on) else None
        row = {
            "model": m,
            "domain": domain.get(("pause", m)) or domain.get(("smooth_tt", m)),
            "pause_n": len(pz),
            "pause_takeover": _rate(pz, "takeover"),
            "pause_short": _rate(pz, "short"),
            "pause_silent": _rate(pz, "silent"),
            "stt_n": len(so),
            "stt_taken": _rate(so, "taken", "overlap"),
            "stt_overlap": _rate(so, "overlap"),
            "stt_no_response": _rate(so, "no_response"),
            "onset_q1": q(25), "onset_median": q(50), "onset_q3": q(75),
        }
        tags = []
        if row["pause_takeover"] is not None and row["pause_takeover"] >= takeover_hi:
            tags.append("overlapping")
        if row["stt_no_response"] is not None and row["stt_no_response"] >= no_response_hi:
            tags.append("silent")
        if row["onset_median"] is not None and row["onset_median"] >= slow_onset:
            tags.append("slow")
        row["tendency"] = "+".join(tags) or "-"
        rows.append(row)
    return rows


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})


def _fmt(v, pct=False):
    if v is None:
        return "-"
    return f"{100 * v:.0f}%" if pct else f"{v:+.2f}"


def write_markdown(path: str, rows: List[Dict[str, Any]], thresholds: Dict[str, float]) -> None:
    lines = [
        "| system | Pause: talked through | backchannel only | silent | "
        "Smooth TT: took the turn | of which overlapping A | no response | "
        "onset Q1 / median / Q3 (s) | tendency |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {_fmt(r['pause_takeover'], 1)} | {_fmt(r['pause_short'], 1)} | "
            f"{_fmt(r['pause_silent'], 1)} | {_fmt(r['stt_taken'], 1)} | "
            f"{_fmt(r['stt_overlap'], 1)} | {_fmt(r['stt_no_response'], 1)} | "
            f"{_fmt(r['onset_q1'])} / {_fmt(r['onset_median'])} / {_fmt(r['onset_q3'])} | "
            f"{r['tendency']} |")
    lines += [
        "",
        f"tendency (rough guide): overlapping = Pause talked through >= "
        f"{thresholds['takeover_hi']:.0%}; silent = Smooth TT no response >= "
        f"{thresholds['no_response_hi']:.0%}; slow = median onset >= "
        f"{thresholds['slow_onset']} s. Onset < 0 = started before A finished.",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def plot(rows: List[Dict[str, Any]], out_base: str) -> List[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    pts = [r for r in rows if r["pause_takeover"] is not None
           and r["stt_no_response"] is not None]
    if not pts:
        return []
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.axvline(0.5, color="#bbbbbb", lw=0.8, zorder=0)
    ax.axhline(0.5, color="#bbbbbb", lw=0.8, zorder=0)
    for r in pts:
        x, y = r["pause_takeover"], r["stt_no_response"]
        ax.scatter([x], [y], s=36, color="#2a6fbb", zorder=2)
        lab = r["model"]
        if r["onset_median"] is not None:
            lab += f"\n(onset {r['onset_median']:+.2f} s)"
        ax.annotate(lab, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=7)
    for x, y, lab in ((0.25, 0.75, "stays silent"), (0.75, 0.25, "talks over"),
                      (0.75, 0.75, "talks over and\nmisses turns"), (0.25, 0.25, "desired")):
        ax.text(x, y, lab, ha="center", va="center", fontsize=9, color="#b0b0b0",
                zorder=0)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Pause: share talked through (lower is better)")
    ax.set_ylabel("Smooth TT: share without a response (lower is better)")
    fig.tight_layout()
    paths = []
    for ext in ("png", "svg"):
        p = f"{out_base}.{ext}"
        fig.savefig(p, dpi=160)
        paths.append(p)
    plt.close(fig)
    return paths


def run(results_path: str, out_dir: str, models=None, takeover_hi: float = 0.5,
        no_response_hi: float = 0.5, slow_onset: float = 1.0, log=print) -> Dict[str, Any]:
    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)
    events = event_rows(results)
    if models:
        events = [e for e in events if e["model"] in set(models)]
    rows = profile_rows(events, results, takeover_hi, no_response_hi, slow_onset)
    os.makedirs(out_dir, exist_ok=True)
    _write_csv(os.path.join(out_dir, "events.csv"), events)
    _write_csv(os.path.join(out_dir, "profile.csv"), rows)
    write_markdown(os.path.join(out_dir, "profile.md"), rows,
                   {"takeover_hi": takeover_hi, "no_response_hi": no_response_hi,
                    "slow_onset": slow_onset})
    figs = plot(rows, os.path.join(out_dir, "profile"))
    log(f"[fdbja] {len(rows)} systems, {len(events)} events -> {out_dir}"
        + ("" if figs else " (no figure: install the analysis extra for matplotlib)"))
    return {"profile": rows, "events": events, "figures": figs}
