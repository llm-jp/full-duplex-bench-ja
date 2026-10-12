"""Build evaluation clips from a dialogue corpus (LLM-jp-Zoom1).

Detects pause / smooth_tt / bc events across every session, samples a target
number per task, and extracts a 20 s stereo clip centered on each event with a
time-rebased jchat transcript. Output matches the layout ``score-offline``
reads: ``{eval_dir}/{task}/{stem}.wav`` + ``.json`` and a ``manifest.jsonl``.

Ported from the research script ``sample_events_v4.py``. Audio/corpus are
obtained separately (see DATASET.md); nothing here is bundled.
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List

from fdbja.data.event_detection import DETECTORS

CLIP_DURATION = 20.0
_TOL = 0.05


def _event_time(task: str, ev: dict) -> float:
    if task == "pause":
        return (ev["start"] + ev["end"]) / 2
    if task == "smooth_tt":
        return ev["a_end"]
    if task == "bc":
        return (ev["a_utt_start"] + ev["a_utt_end"]) / 2
    raise ValueError(task)


def slice_jchat(full: List[dict], t0: float, t1: float) -> List[dict]:
    """Characters within [t0, t1], timestamps rebased to clip start."""
    out = []
    for c in full:
        if c["start"] >= t0 - _TOL and c["end"] <= t1 + _TOL:
            out.append({
                "speaker": c["speaker"],
                "word": c["word"],
                "start": round(max(0.0, c["start"] - t0), 3),
                "end": round(max(0.0, c["end"] - t0), 3),
            })
    return out


def _extract_clip(audio_path, jchat, event_time, clip_duration=CLIP_DURATION):
    import soundfile as sf
    info = sf.info(audio_path)
    clip_start = event_time - clip_duration / 2
    clip_start = max(0.0, min(clip_start, info.duration - clip_duration))
    clip_end = clip_start + clip_duration
    if clip_end > info.duration:
        return None
    s0, s1 = int(clip_start * info.samplerate), int(clip_end * info.samplerate)
    audio, sr = sf.read(audio_path, start=s0, stop=s1)
    sliced = slice_jchat(jchat, clip_start, clip_end)
    if len(sliced) < 5:
        return None
    return audio, sr, sliced, clip_start, clip_end


def load_session_list(path: str) -> set:
    """Read recording ids (one per line; ``#`` comments and blanks ignored).

    Each id is matched against the session directory name *and* the file stem
    (``key``) of the jchat JSON, so either naming convention works.
    """
    ids = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                ids.add(line)
    return ids


def collect_events(text_dir: str, tasks: List[str], limit_sessions: int = 0,
                   log=print, sessions: set | None = None) -> Dict[str, list]:
    files = []
    for root, _dirs, fs in os.walk(text_dir):
        for f in sorted(fs):
            if f.endswith(".json"):
                files.append({"session_id": os.path.basename(root),
                              "key": f[:-5],
                              "json_path": os.path.join(root, f)})
    files.sort(key=lambda x: x["key"])
    if sessions is not None:
        before = len(files)
        files = [x for x in files
                 if x["session_id"] in sessions or x["key"] in sessions]
        log(f"session filter: {len(files)}/{before} sessions kept "
            f"({len(sessions)} ids given)")
    if limit_sessions:
        files = files[:limit_sessions]
    log(f"scanning {len(files)} sessions for events...")

    events = {t: [] for t in tasks}
    for i, finfo in enumerate(files):
        with open(finfo["json_path"], encoding="utf-8") as f:
            data = json.load(f)
        if not data:
            continue
        for task in tasks:
            for ev in DETECTORS[task](data):
                events[task].append({**finfo, "event": ev,
                                     "event_time": _event_time(task, ev)})
        if (i + 1) % 200 == 0:
            log(f"  {i + 1}/{len(files)}  " +
                " ".join(f"{t}:{len(events[t])}" for t in tasks))
    return events


def _sample_extract(events, task, n_samples, audio_dir, out_dir, seed, log):
    import soundfile as sf
    cand_pool = events[task]
    rng = random.Random(seed)
    candidates = rng.sample(cand_pool, min(n_samples * 2, len(cand_pool)))
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    used_sessions = set()
    success = 0
    for cand in candidates:
        if success >= n_samples:
            break
        sid, key = cand["session_id"], cand["key"]
        if sid in used_sessions and success < n_samples * 0.8:
            continue
        audio_path = os.path.join(audio_dir, sid, f"{key}.wav")
        if not os.path.exists(audio_path):
            continue
        with open(cand["json_path"], encoding="utf-8") as f:
            jchat = json.load(f)
        res = _extract_clip(audio_path, jchat, cand["event_time"])
        if res is None:
            continue
        audio, sr, sliced, cs, ce = res
        stem = f"{success:03d}_{key}_{cs:.1f}_{ce:.1f}"
        sf.write(os.path.join(out_dir, f"{stem}.wav"), audio, sr)
        with open(os.path.join(out_dir, f"{stem}.json"), "w",
                  encoding="utf-8") as f:
            json.dump(sliced, f, ensure_ascii=False, indent=2)
        manifest.append({"clip": stem, "session_id": sid, "key": key,
                         "event_time": round(cand["event_time"], 3),
                         "clip_start": round(cs, 3), "clip_end": round(ce, 3)})
        used_sessions.add(sid)
        success += 1
    with open(os.path.join(out_dir, "manifest.jsonl"), "w",
              encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    log(f"[{task}] wrote {success} clips + manifest to {out_dir}")
    return success


def reconstruct(config, limit_sessions: int = 0, log=print,
                sessions_file: str | None = None) -> None:
    """Build eval clips. ``sessions_file`` restricts the scan to the listed
    recordings (e.g. a model's held-out split) so that a ``domain: out``
    evaluation set can be built from the same corpus."""
    zoom1 = config.get("corpus", "zoom1_root")
    text_dir = config.get("corpus", "text_dir", default=os.path.join(zoom1, "text"))
    audio_dir = config.get("corpus", "audio_dir",
                           default=os.path.join(zoom1, "audio_merged"))
    eval_dir = config.eval_dir
    tasks = config.tasks
    n = config.get("detection", "clips_per_task", default=200)
    seed = config.get("detection", "seed", default=42)

    sessions = load_session_list(sessions_file) if sessions_file else None
    events = collect_events(text_dir, tasks, limit_sessions, log,
                            sessions=sessions)
    log("totals: " + ", ".join(f"{t}={len(events[t])}" for t in tasks))
    for task in tasks:
        _sample_extract(events, task, n, audio_dir,
                        os.path.join(eval_dir, task), seed, log)
