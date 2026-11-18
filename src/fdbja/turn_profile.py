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

When the results carry the Pause reaction rule (``metrics.pause_response``,
on by default; ``no_response`` / ``interrupt`` per event), the Pause side of
the profile also reports it, and the tendency and the figure use it instead
of the transcript-based outcome:

* ``pause_no_reaction`` / ``pause_barge_in``: the two rates of the
  leaderboard (the rule that was checked against listeners).
* ``pause_brief``: share of pauses with a reaction and no barge-in, and
  what those reactions were according to the transcript:
  ``brief_backchannel`` (backchannel words or laughter only),
  ``brief_other`` (other words) and ``brief_no_text`` (nothing transcribed:
  a breath, noise, or speech the ASR missed). These three are a reference:
  they depend on the ASR and its character times, are not used for the
  ranking and were not checked against listeners.

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


def classify_reaction(row: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """``(reaction, text)`` of a Pause event from the audio-based rule.

    reaction: ``barge_in`` (interrupt = 1), ``none`` (no reaction, no
    barge-in) or ``brief`` (reacted, no barge-in); ``None`` when the rule
    was not scored. text (brief reactions only): ``backchannel`` /
    ``other`` / ``no_text``; ``None`` when the results have no character
    counts (written by an older ``score-offline``)."""
    if row.get("no_response") is None or row.get("interrupt") is None:
        return None, None
    if row["interrupt"]:
        return "barge_in", None
    if row["no_response"]:
        return "none", None
    n, r = row.get("reaction_chars"), row.get("reaction_residual_chars")
    if n is None or r is None:
        return "brief", None
    return "brief", "no_text" if n == 0 else ("backchannel" if r == 0 else "other")


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
        reaction, text = classify_reaction(r) if task == "pause" else (None, None)
        out.append({
            "task": task, "model": r["model"], "clip": r["clip"],
            "event_index": r["event_index"],
            "event_start": round(r["t_start"], 3), "event_end": round(r["t_end"], 3),
            "window_start": round(w0, 3), "window_end": round(w1, 3),
            "TO": r["TO"], "onset": None if r.get("onset") is None else round(r["onset"], 3),
            "outcome": outcome, "reason": reason,
            # audio-based Pause reaction rule (empty for Smooth TT / when not scored)
            "no_response": r.get("no_response") if task == "pause" else None,
            "interrupt": r.get("interrupt") if task == "pause" else None,
            "reaction": reaction, "reaction_text": text,
            "system_segments": " ".join(_fmt_seg(s) for s in r["sys_segments"]),
        })
    return out


def _rate(xs: List[str], *keys: str) -> Optional[float]:
    return round(sum(x in keys for x in xs) / len(xs), 4) if xs else None


def profile_rows(events: List[Dict[str, Any]], results: Dict[str, Any],
                 takeover_hi: float = 0.5, no_response_hi: float = 0.5,
                 slow_onset: float = 1.0, barge_in_hi: float = 0.5,
                 no_reaction_hi: float = 0.5) -> List[Dict[str, Any]]:
    models = list(dict.fromkeys(e["model"] for e in events))
    domain = {(s["task"], s["model"]): s.get("domain") for s in results.get("summary", [])}
    rows = []
    for m in models:
        pe = [e for e in events if e["model"] == m and e["task"] == "pause"]
        pz = [e["outcome"] for e in pe]
        # audio-based reaction rule: only when every Pause event was scored
        scored = bool(pe) and all(e.get("reaction") is not None for e in pe)
        brief = [e for e in pe if e.get("reaction") == "brief"] if scored else []
        texts = [e["reaction_text"] for e in brief]
        has_text = bool(brief) and all(t is not None for t in texts)

        def mean(key, pe=pe, scored=scored):
            return round(sum(e[key] for e in pe) / len(pe), 4) if scored else None
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
            "pause_no_reaction": mean("no_response"),
            "pause_barge_in": mean("interrupt"),
            "pause_brief": round(len(brief) / len(pe), 4) if scored else None,
            "pause_brief_n": len(brief) if scored else None,
            "brief_backchannel": _rate(texts, "backchannel") if has_text else None,
            "brief_other": _rate(texts, "other") if has_text else None,
            "brief_no_text": _rate(texts, "no_text") if has_text else None,
            "stt_n": len(so),
            "stt_taken": _rate(so, "taken", "overlap"),
            "stt_overlap": _rate(so, "overlap"),
            "stt_no_response": _rate(so, "no_response"),
            "onset_q1": q(25), "onset_median": q(50), "onset_q3": q(75),
        }
        tags = []
        if scored:
            # the rule checked against listeners decides, not the transcript
            if row["pause_barge_in"] >= barge_in_hi:
                tags.append("overlapping")
            if row["pause_no_reaction"] >= no_reaction_hi:
                tags.append("unreactive")
        elif row["pause_takeover"] is not None and row["pause_takeover"] >= takeover_hi:
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
    scored = [r for r in rows if r.get("pause_barge_in") is not None]
    if scored:
        over = (f"overlapping = Pause barge-in >= {thresholds['barge_in_hi']:.0%}; "
                f"unreactive = Pause no reaction >= {thresholds['no_reaction_hi']:.0%}")
        if len(scored) < len(rows):
            over += (f" (systems without the reaction rule: overlapping = Pause talked "
                     f"through >= {thresholds['takeover_hi']:.0%})")
    else:
        over = f"overlapping = Pause talked through >= {thresholds['takeover_hi']:.0%}"
    lines += [
        "",
        f"tendency (rough guide): {over}; silent = Smooth TT no response >= "
        f"{thresholds['no_response_hi']:.0%}; slow = median onset >= "
        f"{thresholds['slow_onset']} s. Onset < 0 = started before A finished.",
    ]
    if scored:
        lines += [
            "",
            "Pause reaction (audio-based rule, the one used for the ranking):",
            "",
            "| system | no reaction | barge-in | brief reaction | "
            "of brief: backchannel words only | other words | no text |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for r in scored:
            lines.append(
                f"| {r['model']} | {_fmt(r['pause_no_reaction'], 1)} | "
                f"{_fmt(r['pause_barge_in'], 1)} | {_fmt(r['pause_brief'], 1)} "
                f"({r['pause_brief_n']}) | {_fmt(r['brief_backchannel'], 1)} | "
                f"{_fmt(r['brief_other'], 1)} | {_fmt(r['brief_no_text'], 1)} |")
        lines += [
            "",
            "brief reaction = reacted in the pause without barging in (count in "
            "parentheses). The three shares on the right split those reactions by the "
            "transcript: a reference that depends on the ASR and its character times, "
            "not used for the ranking and not checked against listeners. A pause can "
            "count as both no reaction and barge-in (overlap before the pause), so the "
            "first three columns need not sum to 100%.",
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
    # x: the barge-in rate of the reaction rule when every system has it
    xkey, xlab = "pause_takeover", "Pause: share talked through (lower is better)"
    if all(r.get("pause_barge_in") is not None for r in pts):
        xkey, xlab = "pause_barge_in", "Pause: barge-in rate (lower is better)"
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.axvline(0.5, color="#bbbbbb", lw=0.8, zorder=0)
    ax.axhline(0.5, color="#bbbbbb", lw=0.8, zorder=0)
    # systems often sit close together: number the points and list the names beside the axes
    for i, r in enumerate(pts, 1):
        x, y = r[xkey], r["stt_no_response"]
        ax.scatter([x], [y], s=36, color="#2a6fbb", zorder=2)
        ax.annotate(str(i), (x, y), xytext=(4, 3), textcoords="offset points", fontsize=8,
                    zorder=3)
        lab = f"{i}  {r['model']}"
        if r["onset_median"] is not None:
            lab += f" (onset {r['onset_median']:+.2f} s)"
        ax.text(1.04, 1.0 - 0.06 * (i - 1), lab, transform=ax.transAxes, fontsize=7,
                va="top", ha="left")
    for x, y, lab in ((0.25, 0.75, "stays silent"), (0.75, 0.25, "talks over"),
                      (0.75, 0.75, "talks over and\nmisses turns"), (0.25, 0.25, "desired")):
        ax.text(x, y, lab, ha="center", va="center", fontsize=9, color="#b0b0b0",
                zorder=0)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel(xlab)
    ax.set_ylabel("Smooth TT: share without a response (lower is better)")
    fig.tight_layout()
    paths = []
    for ext in ("png", "svg"):
        p = f"{out_base}.{ext}"
        fig.savefig(p, dpi=160, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def run(results_path: str, out_dir: str, models=None, takeover_hi: float = 0.5,
        no_response_hi: float = 0.5, slow_onset: float = 1.0, barge_in_hi: float = 0.5,
        no_reaction_hi: float = 0.5, log=print) -> Dict[str, Any]:
    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)
    events = event_rows(results)
    if models:
        events = [e for e in events if e["model"] in set(models)]
    rows = profile_rows(events, results, takeover_hi, no_response_hi, slow_onset,
                        barge_in_hi, no_reaction_hi)
    os.makedirs(out_dir, exist_ok=True)
    _write_csv(os.path.join(out_dir, "events.csv"), events)
    _write_csv(os.path.join(out_dir, "profile.csv"), rows)
    write_markdown(os.path.join(out_dir, "profile.md"), rows,
                   {"takeover_hi": takeover_hi, "no_response_hi": no_response_hi,
                    "slow_onset": slow_onset, "barge_in_hi": barge_in_hi,
                    "no_reaction_hi": no_reaction_hi})
    figs = plot(rows, os.path.join(out_dir, "profile"))
    log(f"[fdbja] {len(rows)} systems, {len(events)} events -> {out_dir}"
        + ("" if figs else " (no figure: install the analysis extra for matplotlib)"))
    return {"profile": rows, "events": events, "figures": figs}
