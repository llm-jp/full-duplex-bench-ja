"""Check of the files written by ``fdbja infer``, before they are scored.

The scores depend on where each adapter puts the model's audio on the time
axis. A recording can look complete (every ``combined.wav`` is there) and
still be unusable: the client stopped listening before the input ended, the
server could not keep up with real time, the systems were recorded under
different timeline rules, or a model directory holds no audio at all. None of
these raises an error during ``infer`` or ``score-offline``. This module reads
the reception record of every clip (``events.json``, written next to
``combined.wav``) and reports them.

Per clip (``clips.csv``):

* whether the output and the record exist;
* number of audio chunks, seconds received, seconds placed on the timeline and
  seconds discarded (after an interruption notice);
* arrival of the first audio and the end of the last one;
* stalls: silence that the timeline had to insert between two chunks of the
  same response because the next chunk arrived after the previous one had
  finished playing (the total is how far the later audio is shifted);
* when and why the client stopped listening (``client.receive_end``);
* error notes and responses that the server reported as failed;
* with the audio: whether the model channel is digital silence, whether
  ``received.wav`` matches the record, and whether the record rebuilds
  ``output.wav`` exactly.

Per system and task (``systems.csv`` and ``response_check.md``): the counts,
the timeline rules in use, and the list of findings. ``error`` findings mean
the scores of that system should not be used as they are; ``warning`` findings
need a look.

The check does not judge the model's behaviour. A system that stays silent or
is often interrupted is reported with its counts, not flagged, unless every
clip of a task is empty.
"""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from statistics import median
from typing import Any, Dict, List, Optional

import numpy as np

from fdbja.adapters._timeline import (
    RECEIVE_END,
    RECORD_FORMAT,
    RULES,
    SESSION,
    _Queue,
    timeline_from_record,
)

#: response statuses that mean the server did not finish a response normally
FAILED_STATUSES = ("failed", "incomplete")
#: slack for comparing the end of listening with the input length (seconds)
END_TOLERANCE_S = 0.05

#: finding -> level and a one-line description (also printed in the report)
FINDINGS = {
    "missing_output": ("error", "no combined.wav for the clip"),
    "bad_output": ("error", "combined.wav cannot be read or has no model channel"),
    "bad_record": ("error", "events.json cannot be read or has an unknown format"),
    "received_mismatch": ("error", "received.wav does not have the length the record states"),
    "rebuild_differs": ("error", "the record does not rebuild output.wav"),
    "ended_early": ("error", "the client stopped listening before the input ended"),
    "all_no_audio": ("error", "no clip of the task holds any model audio"),
    "mixed_rules": ("error", "clips of one system were recorded under different timeline rules"),
    "mixed_settings": ("error", "clips of one system were recorded with different session "
                                "settings (model, voice or instructions)"),
    "response_failed": ("warning", "the server reported a failed or incomplete response"),
    "stall": ("warning", "the audio stream stalled for longer than the limit"),
    "tail_cut": ("warning", "model audio beyond the written length was cut"),
    "no_record": ("warning", "no events.json: the recording cannot be checked"),
    "rules_differ": ("warning", "systems were recorded under different timeline rules"),
}
LEVELS = ("error", "warning")

CLIP_FIELDS = [
    "model", "task", "clip", "findings", "has_output", "has_record", "adapter", "timeline",
    "rule", "settings", "input_seconds", "chunks", "received_seconds", "placed_seconds",
    "discarded_seconds", "interruptions", "chunks_after_interruption", "first_audio",
    "last_audio_end", "stalls", "stall_seconds", "longest_stall", "cut_seconds",
    "receive_end", "receive_end_reason", "error_notes", "failed_responses", "no_audio",
    "rebuild",
]
SYSTEM_FIELDS = [
    "model", "task", "clips", "outputs", "records", "rules", "timelines", "adapters",
    "settings",
    "no_audio", "interruptions", "clips_interrupted", "discarded_seconds", "stalled_clips",
    "longest_stall", "first_audio_median", "ended_early", "errors", "warnings",
]


def replay(record: Dict[str, Any]) -> Dict[str, Any]:
    """Timing facts of one reception record (the audio is not needed)."""
    if record.get("format") != RECORD_FORMAT:
        raise ValueError(f"unknown record format: {record.get('format')!r}")
    rule = record["rule"]
    if rule not in RULES:
        raise ValueError(f"unknown rule: {rule!r}")
    sr = int(record["sample_rate"])
    events = record["events"]
    received = sum(e["n"] for e in events if e["type"] == "audio")

    # spans on the timeline: (start, n, item, index of the event)
    if rule == "concat":
        spans, pos = [], 0
        for i, e in enumerate(events):
            if e["type"] == "audio":
                spans.append((pos, e["n"], e.get("item"), i))
                pos += e["n"]
    else:
        queue = _Queue(sr, flush=(rule == "playout"))
        for i, e in enumerate(events):
            if e["type"] == "audio":
                queue.audio(e["t"], e["n"], i, e.get("item"))
            elif e["type"] == "interruption":
                queue.interrupt(e["t"])
        spans = [(s[0], s[1], s[3], s[2]) for s in queue.spans]

    # a stall: the next chunk of the same response came after the queue ran dry
    stalls = []
    for prev, cur in zip(spans, spans[1:]):
        gap = cur[0] - (prev[0] + prev[1])
        if gap <= 0 or prev[2] != cur[2]:
            continue
        if cur[2] is None and any(e["type"] != "audio" for e in events[prev[3] + 1:cur[3]]):
            continue        # a server event in between: a new response, not a stall
        stalls.append(gap / sr)

    placed = sum(s[1] for s in spans)
    end = max((s[0] + s[1] for s in spans), default=0)
    max_samples = record.get("max_samples")
    cut = max(0, end - max_samples) if max_samples is not None else 0
    notes = [e for e in events if e["type"] == "note"]
    ends = [e for e in notes if e.get("event") == RECEIVE_END]
    session = next((e for e in notes if e.get("event") == SESSION), {})
    settings = ", ".join(f"{k}={session[k]}" for k in sorted(session)
                         if k not in ("t", "type", "event"))
    return {
        "adapter": record.get("adapter", ""),
        "settings": settings,
        "timeline": record.get("timeline", ""),
        "rule": rule,
        "sample_rate": sr,
        "chunks": sum(1 for e in events if e["type"] == "audio"),
        "received_seconds": received / sr,
        "placed_seconds": placed / sr,
        "discarded_seconds": (received - placed) / sr,
        "interruptions": sum(1 for e in events if e["type"] == "interruption"),
        "chunks_after_interruption": sum(
            1 for e in events if e["type"] == "audio" and e.get("after_interruption")),
        "first_audio": spans[0][0] / sr if spans else None,
        "last_audio_end": end / sr if spans else None,
        "stalls": len(stalls),
        "stall_seconds": sum(stalls),
        "longest_stall": max(stalls, default=0.0),
        "cut_seconds": cut / sr,
        "receive_end": ends[-1]["t"] if ends else None,
        "receive_end_reason": ends[-1].get("reason", "") if ends else "",
        "error_notes": sum(1 for e in notes if e.get("event") == "error"),
        "failed_responses": sum(1 for e in notes if e.get("status") in FAILED_STATUSES),
    }


def _same_audio(a: np.ndarray, b: np.ndarray) -> bool:
    """True when two int16 signals agree (within 1 LSB) up to trailing zeros."""
    m = min(len(a), len(b))
    if m and np.max(np.abs(a[:m].astype(np.int32) - b[:m].astype(np.int32))) > 1:
        return False
    rest = a[m:] if len(a) > m else b[m:]
    return not np.any(rest)


def check_clip(combined_path: str, input_seconds: Optional[float] = None,
               stall_limit: float = 0.2, records_only: bool = False) -> Dict[str, Any]:
    """Check the files of one clip. ``combined_path`` is its ``combined.wav``;
    ``input_seconds`` the length of the audio that was sent (``None`` = not
    known, the end of listening is then not judged)."""
    d = os.path.dirname(combined_path)
    row: Dict[str, Any] = {k: "" for k in CLIP_FIELDS}
    findings: List[str] = []
    row["input_seconds"] = input_seconds
    row["has_output"] = os.path.exists(combined_path)
    record_path = os.path.join(d, "events.json")
    row["has_record"] = os.path.exists(record_path)
    if not row["has_output"]:
        row["findings"] = ["missing_output"]
        return row

    record = None
    if row["has_record"]:
        try:
            with open(record_path, encoding="utf-8") as f:
                record = json.load(f)
            row.update(replay(record))
        except (ValueError, KeyError, TypeError, OSError) as e:
            record = None
            findings.append("bad_record")
            row["rebuild"] = f"record: {e}"
    else:
        findings.append("no_record")

    if record is not None:
        if (input_seconds is not None and row["receive_end"] is not None
                and row["receive_end"] < input_seconds - END_TOLERANCE_S):
            findings.append("ended_early")
        if row["failed_responses"]:
            findings.append("response_failed")
        if row["longest_stall"] > stall_limit:
            findings.append("stall")
        if row["cut_seconds"] > 0:
            findings.append("tail_cut")
        row["no_audio"] = row["chunks"] == 0

    if not records_only:
        import soundfile as sf

        try:
            audio, sr = sf.read(combined_path, dtype="int16", always_2d=True)
            if audio.shape[1] < 2:
                raise ValueError("one channel only")
            row["no_audio"] = not np.any(audio[:, 1])
        except (RuntimeError, ValueError, OSError):
            findings.append("bad_output")
            sr = None
        received_path = os.path.join(d, "received.wav")
        output_path = os.path.join(d, "output.wav")
        if record is not None and sr is not None and os.path.exists(received_path):
            try:
                received, rsr = sf.read(received_path, dtype="int16")
            except (RuntimeError, OSError):
                received, rsr = np.zeros(0, dtype=np.int16), None
            if len(received) != record.get("received_samples"):
                findings.append("received_mismatch")
                row["rebuild"] = "not checked"
            elif rsr != record["sample_rate"] or not os.path.exists(output_path):
                row["rebuild"] = "not checked"
            else:
                try:
                    saved, _ = sf.read(output_path, dtype="int16")
                    same = _same_audio(timeline_from_record(record, received), saved)
                except (RuntimeError, OSError):
                    same = False
                row["rebuild"] = "same" if same else "differs"
                if not same:
                    findings.append("rebuild_differs")
        elif record is not None:
            row["rebuild"] = "not checked"
    row["findings"] = findings
    return row


def _level(finding: str) -> str:
    return FINDINGS[finding][0]


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per system and task summary and the list of findings."""
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["model"], r["task"]), []).append(r)
    systems, findings = [], []

    def add(code, model, task, clips):
        findings.append({"level": _level(code), "finding": code, "model": model, "task": task,
                         "clips": len(clips), "examples": sorted(clips)[:5],
                         "meaning": FINDINGS[code][1]})

    rules_by_model: Dict[str, Counter] = {}
    settings_by_model: Dict[str, Dict[str, List[str]]] = {}
    for (model, task), group in groups.items():
        per_code: Dict[str, List[str]] = {}
        for r in group:
            for code in r["findings"]:
                per_code.setdefault(code, []).append(r["clip"])
        with_output = [r for r in group if r["has_output"]]
        with_record = [r for r in with_output if r["rule"]]
        rules = Counter(r["rule"] for r in with_record)
        rules_by_model.setdefault(model, Counter()).update(rules)
        for r in with_record:
            if r["settings"]:
                settings_by_model.setdefault(model, {}).setdefault(
                    r["settings"], []).append(f"{task}/{r['clip']}")
        silent = [r["clip"] for r in with_output if r["no_audio"] is True]
        if with_output and len(silent) == len(with_output):
            per_code["all_no_audio"] = silent
        if len(rules) > 1:
            minority = rules.most_common()[-1][0]
            per_code["mixed_rules"] = [r["clip"] for r in with_record if r["rule"] == minority]
        for code in FINDINGS:
            if code in per_code:
                add(code, model, task, per_code[code])
        first = [r["first_audio"] for r in with_record if r["first_audio"] is not None]
        systems.append({
            "model": model, "task": task, "clips": len(group), "outputs": len(with_output),
            "records": len(with_record),
            "rules": "/".join(sorted(rules)),
            "timelines": "/".join(sorted({r["timeline"] for r in with_record})),
            "adapters": "/".join(sorted({r["adapter"] for r in with_record})),
            "settings": " | ".join(sorted({r["settings"] for r in with_record
                                           if r["settings"]})),
            "no_audio": len(silent),
            "interruptions": sum(r["interruptions"] for r in with_record),
            "clips_interrupted": sum(1 for r in with_record if r["interruptions"]),
            "discarded_seconds": round(sum(r["discarded_seconds"] for r in with_record), 3),
            "stalled_clips": len(per_code.get("stall", [])),
            "longest_stall": round(max((r["longest_stall"] for r in with_record), default=0.0), 3),
            "first_audio_median": round(median(first), 3) if first else "",
            "ended_early": len(per_code.get("ended_early", [])),
            "errors": sum(len(v) for k, v in per_code.items() if _level(k) == "error"),
            "warnings": sum(len(v) for k, v in per_code.items() if _level(k) == "warning"),
        })

    # one system = one model, one voice, one instruction, over all tasks
    for model, by_setting in settings_by_model.items():
        if len(by_setting) > 1:
            common = max(by_setting, key=lambda k: len(by_setting[k]))
            others = sorted(c for k, v in by_setting.items() if k != common for c in v)
            findings.append({
                "level": "error", "finding": "mixed_settings", "model": model,
                "task": "", "clips": len(others), "examples": others[:5],
                "meaning": FINDINGS["mixed_settings"][1]})
    # the timeline rule decides the response latency: it has to be the same
    # for every system that is compared
    used = {m: max(c, key=c.get) for m, c in rules_by_model.items() if c}
    if len(set(used.values())) > 1:
        common = Counter(used.values()).most_common(1)[0][0]
        for model in sorted(m for m, rule in used.items() if rule != common):
            findings.append({
                "level": "warning", "finding": "rules_differ", "model": model, "task": "",
                "clips": 0, "examples": [],
                "meaning": f"recorded under '{used[model]}', most systems under '{common}': "
                           "latencies are not comparable"})
    order = {level: i for i, level in enumerate(LEVELS)}
    findings.sort(key=lambda f: (order[f["level"]], f["model"], f["task"], f["finding"]))
    return {
        "systems": systems, "findings": findings,
        "errors": sum(1 for f in findings if f["level"] == "error"),
        "warnings": sum(1 for f in findings if f["level"] == "warning"),
    }


def _fmt(v) -> str:
    if v is None or v == "":
        return ""
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.3f}"
    if isinstance(v, (list, tuple)):
        return " ".join(str(x) for x in v)
    return str(v)


def _write_csv(path: str, fields: List[str], rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in rows:
            w.writerow([_fmt(r.get(k)) for k in fields])


def write_markdown(path: str, summary: Dict[str, Any], stall_limit: float,
                   records_only: bool) -> None:
    lines = ["# Response check", ""]
    lines.append(f"{summary['errors']} error finding(s), {summary['warnings']} warning "
                 "finding(s). An error means the scores of that system and task should not "
                 "be used as they are.")
    if records_only:
        lines.append("The audio files were not read (`--records-only`): empty outputs of "
                     "systems without a record and the rebuild of `output.wav` were not checked.")
    lines += ["", "## Findings", ""]
    if summary["findings"]:
        lines += ["| level | system | task | finding | clips | examples | meaning |",
                  "|---|---|---|---|---|---|---|"]
        for f in summary["findings"]:
            lines.append(f"| {f['level']} | {f['model']} | {f['task']} | `{f['finding']}` | "
                         f"{f['clips'] or ''} | {' '.join(f['examples'])} | {f['meaning']} |")
    else:
        lines.append("None.")
    lines += ["", "## Per system and task", "",
              "| system | task | clips | outputs | records | rule | no audio | interruption "
              "notices (clips) | discarded (s) | stalled clips | longest stall (s) | "
              "first audio, median (s) | ended early |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summary["systems"]:
        lines.append(
            f"| {s['model']} | {s['task']} | {s['clips']} | {s['outputs']} | {s['records']} | "
            f"{s['rules']} | {s['no_audio']} | {s['interruptions']} ({s['clips_interrupted']}) | "
            f"{_fmt(float(s['discarded_seconds']))} | {s['stalled_clips']} | "
            f"{_fmt(float(s['longest_stall']))} | {_fmt(s['first_audio_median'])} | "
            f"{s['ended_early']} |")
    settings = {}
    for s in summary["systems"]:
        if s["settings"]:
            settings.setdefault((s["model"], s["settings"]), []).append(s["task"])
    if settings:
        lines += ["", "## Session settings", "",
                  "As recorded by the adapter when the session was opened. `instructions` "
                  "is an identifier of the instruction text (first 12 hex digits of its "
                  "SHA-256), not the text.", "",
                  "| system | tasks | settings |", "|---|---|---|"]
        for (model, value), tasks in settings.items():
            lines.append(f"| {model} | {' '.join(tasks)} | {value} |")
    lines += [
        "",
        "- **no audio**: clips whose model channel is empty (without the audio: clips whose "
        "record holds no audio chunk). A system may stay silent; it is an error only when "
        "every clip of a task is empty.",
        "- **discarded**: audio that was received and not played, because the server reported "
        "an interruption (`timeline: playout`).",
        f"- **stalled clips**: clips with a stall longer than {stall_limit:g} s. A stall is "
        "silence inserted between two chunks of the same response because the next chunk "
        "arrived after the previous one had finished playing. For a model that streams "
        "continuously it means the server fell behind real time, and everything after it is "
        "late by that much.",
        "- **first audio**: arrival of the first audio chunk, from the moment the first input "
        "sample was due to be sent. For a model that streams continuously this is the delay of "
        "the pipeline, not the response latency.",
        "- **ended early**: the client stopped listening before the input ended (connection "
        "closed, no message for too long, or the adapter stopped after a turn). Recorded by "
        f"the adapters as the note `{RECEIVE_END}`; empty when the record has no such note.",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run(config, out_dir: str, models=None, tasks=None, stall_limit: float = 0.2,
        records_only: bool = False, log=print) -> Dict[str, Any]:
    """Check every clip of the config's models and write the report.

    Returns the summary (``errors`` / ``warnings`` are the numbers of findings).
    """
    from fdbja.adapters import BASELINE_ADAPTERS
    from fdbja.runner import _clip_ids

    eval_dir, layout = config.eval_dir, config.layout()
    wanted = set(models) if models else None
    rows: List[Dict[str, Any]] = []
    for model in config.models:
        mname = model["name"]
        if wanted is not None and mname not in wanted:
            continue
        if model.get("adapter") in BASELINE_ADAPTERS and wanted is None:
            log(f"[fdbja] {mname}: trivial baseline, not checked")
            continue
        for task in (tasks or config.tasks):
            for clip in _clip_ids(eval_dir, task, layout):
                gt_wav = os.path.join(eval_dir, layout["gt_wav"].format(
                    task=task, clip=clip, model=mname))
                input_seconds = None
                if os.path.exists(gt_wav):
                    import soundfile as sf
                    info = sf.info(gt_wav)
                    input_seconds = info.frames / info.samplerate
                row = check_clip(os.path.join(eval_dir, layout["model_wav"].format(
                    task=task, clip=clip, model=mname)), input_seconds=input_seconds,
                    stall_limit=stall_limit, records_only=records_only)
                row.update(model=mname, task=task, clip=clip)
                rows.append(row)
    summary = summarize(rows)
    os.makedirs(out_dir, exist_ok=True)
    _write_csv(os.path.join(out_dir, "clips.csv"), CLIP_FIELDS, rows)
    _write_csv(os.path.join(out_dir, "systems.csv"), SYSTEM_FIELDS, summary["systems"])
    write_markdown(os.path.join(out_dir, "response_check.md"), summary, stall_limit,
                   records_only)
    for f in summary["findings"]:
        where = f["model"] + (f"/{f['task']}" if f["task"] else "")
        n = f" ({f['clips']} clips)" if f["clips"] else ""
        log(f"[fdbja] {f['level']}: {where}: {f['finding']}{n}: {f['meaning']}")
    log(f"[fdbja] {len(rows)} clips checked, {summary['errors']} error finding(s), "
        f"{summary['warnings']} warning finding(s) -> {out_dir}")
    summary["rows"] = rows
    return summary
