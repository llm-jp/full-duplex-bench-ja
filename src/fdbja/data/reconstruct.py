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

from fdbja.data.event_detection import DETECTORS, SmoothTTRoles, detect, smooth_tt_role_flags

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
                   log=print, sessions: set | None = None,
                   roles: SmoothTTRoles | None = None) -> Dict[str, list]:
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
            for ev in detect(task, data, roles):
                events[task].append({**finfo, "event": ev,
                                     "event_time": _event_time(task, ev)})
        if (i + 1) % 200 == 0:
            log(f"  {i + 1}/{len(files)}  " +
                " ".join(f"{t}:{len(events[t])}" for t in tasks))
    return events


_BOUNDARY_KEYS = {"pause": ("start", "end"), "smooth_tt": ("a_end", "b_start"),
                  "bc": ("a_utt_start", "a_utt_end")}


def clip_event_boundaries(task: str, sliced: List[dict], audio, sr: int,
                          roles: SmoothTTRoles | None = None) -> List[dict]:
    """Events of one extracted clip with transcript and audio boundaries.

    Written to the manifest so that a distributed evaluation set records which
    window each event is judged on. Scoring recomputes them from the clip
    (``detection.boundaries: audio``), so this is documentation, not input.
    With ``roles.enabled`` every Smooth TT event is listed with its
    failed role checks (``role_flags``, text and audio); scoring drops the
    events whose list is not empty.
    """
    from fdbja.data.event_boundaries import DEFAULT_VAD, refine_events, voice_activity
    use_roles = task == "smooth_tt" and roles is not None and roles.enabled
    events = DETECTORS[task](sliced)
    if use_roles:
        events = [{**ev, "role_flags": smooth_tt_role_flags(ev, sliced, roles)}
                  for ev in events]
    va_a = voice_activity(audio[:, 0], sr, DEFAULT_VAD)
    va_b = voice_activity(audio[:, 1], sr, DEFAULT_VAD) if audio.shape[1] > 1 else None
    out = []
    for ev in refine_events(task, events, va_a, va_b, DEFAULT_VAD, roles=roles):
        k0, k1 = _BOUNDARY_KEYS[task]
        rec = {"transcript": {k0: round(ev["boundary"]["asr"][k0], 3),
                              k1: round(ev["boundary"]["asr"][k1], 3)},
               "audio": {k0: ev[k0], k1: ev[k1]},
               "flags": ev["boundary"]["flags"]}
        if use_roles:
            rec["role_flags"] = ev["role_flags"]
        out.append(rec)
    return out


def _sample_extract(events, task, n_samples, audio_dir, out_dir, seed, log,
                    boundaries: str = "transcript",
                    roles: SmoothTTRoles | None = None):
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
        entry = {"clip": stem, "session_id": sid, "key": key,
                 "event_time": round(cand["event_time"], 3),
                 "clip_start": round(cs, 3), "clip_end": round(ce, 3)}
        if boundaries == "audio" and getattr(audio, "ndim", 1) == 2:
            entry["events"] = clip_event_boundaries(task, sliced, audio, sr, roles)
        manifest.append(entry)
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
    boundaries = config.get("detection", "boundaries", default="transcript")
    roles = SmoothTTRoles.from_config(
        config.get("detection", "smooth_tt_roles", default=None))

    sessions = load_session_list(sessions_file) if sessions_file else None
    events = collect_events(text_dir, tasks, limit_sessions, log,
                            sessions=sessions, roles=roles)
    log("totals: " + ", ".join(f"{t}={len(events[t])}" for t in tasks))
    for task in tasks:
        _sample_extract(events, task, n, audio_dir,
                        os.path.join(eval_dir, task), seed, log,
                        boundaries=boundaries, roles=roles)


def _clip_start_for(task: str, jchat: List[dict], duration: float,
                    approx_start: float, tol: float = 0.051,
                    clip_duration: float = CLIP_DURATION):
    """Exact start of a published clip. Clip names carry the start rounded to
    0.1 s; the exact value is ``event_time - clip_duration / 2`` (clamped to
    the recording) of the event the clip was sampled for, so the event of
    ``task`` whose clip start lies within ``tol`` of ``approx_start`` is
    searched in the full recording. Returns (start, n_matches); ``None`` when
    no event matches."""
    best, n = None, 0
    for ev in DETECTORS[task](jchat):
        cs = _event_time(task, ev) - clip_duration / 2
        cs = max(0.0, min(cs, duration - clip_duration))
        if abs(cs - approx_start) <= tol:
            n += 1
            if best is None or abs(cs - approx_start) < abs(best - approx_start):
                best = cs
    return best, n


def reconstruct_from_list(config, clips_csv: str, log=print) -> Dict[str, int]:
    """Rebuild a published evaluation set from its clip list.

    ``clips_csv`` needs the columns ``task``, ``clip_id``, ``recording_key``
    (file stem of the LLM-jp-Zoom1 recording) and ``clip_start_s``. Each clip
    is cut again from the corpus under ``corpus.eval_dir/{task}/{clip_id}``
    with the same slicing as ``reconstruct``; events are then detected from
    the clip as usual. Clips whose exact start cannot be located fall back to
    ``clip_start_s`` and are reported.
    """
    import csv

    import soundfile as sf
    zoom1 = config.get("corpus", "zoom1_root")
    text_dir = config.get("corpus", "text_dir", default=os.path.join(zoom1, "text"))
    audio_dir = config.get("corpus", "audio_dir",
                           default=os.path.join(zoom1, "audio_merged"))
    eval_dir = config.eval_dir
    index = {}
    for root, _dirs, fs in os.walk(text_dir):
        for f in fs:
            if f.endswith(".json"):
                index[f[:-5]] = (os.path.basename(root), os.path.join(root, f))
    with open(clips_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    tasks = set(config.tasks)
    done = {t: 0 for t in tasks}
    missing, approx = [], []
    manifests: Dict[str, list] = {t: [] for t in tasks}
    for r in rows:
        task = r["task"]
        if task not in tasks:
            continue
        key = r["recording_key"]
        if key not in index:
            missing.append(r["clip_id"])
            continue
        sid, jpath = index[key]
        audio_path = os.path.join(audio_dir, sid, f"{key}.wav")
        if not os.path.exists(audio_path):
            missing.append(r["clip_id"])
            continue
        with open(jpath, encoding="utf-8") as f:
            jchat = json.load(f)
        info = sf.info(audio_path)
        cs, n = _clip_start_for(task, jchat, info.duration, float(r["clip_start_s"]))
        if cs is None:
            cs = float(r["clip_start_s"])
            approx.append(r["clip_id"])
        ce = cs + CLIP_DURATION
        s0, s1 = int(cs * info.samplerate), int(ce * info.samplerate)
        audio, sr = sf.read(audio_path, start=s0, stop=s1)
        out_dir = os.path.join(eval_dir, task)
        os.makedirs(out_dir, exist_ok=True)
        sf.write(os.path.join(out_dir, f"{r['clip_id']}.wav"), audio, sr)
        with open(os.path.join(out_dir, f"{r['clip_id']}.json"), "w",
                  encoding="utf-8") as f:
            json.dump(slice_jchat(jchat, cs, ce), f, ensure_ascii=False, indent=2)
        manifests[task].append({"clip": r["clip_id"], "session_id": sid, "key": key,
                                "clip_start": round(cs, 3), "clip_end": round(ce, 3),
                                "start_matched": n})
        done[task] += 1
    for task, man in manifests.items():
        if not man:
            continue
        with open(os.path.join(eval_dir, task, "manifest.jsonl"), "w",
                  encoding="utf-8") as f:
            for m in man:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
    log("rebuilt: " + ", ".join(f"{t}={n}" for t, n in sorted(done.items())))
    if missing:
        log(f"recordings not found in the corpus: {len(missing)} clips (e.g. {missing[:3]})")
    if approx:
        log(f"exact start not located, used the rounded start: {len(approx)} clips "
            f"(e.g. {approx[:3]})")
    return done
