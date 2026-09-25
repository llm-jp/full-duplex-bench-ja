"""Offline scoring runner: score precomputed model outputs.

Given, per task and model:
  - GT stereo wav + GT jchat (the evaluation clip and its human reference)
  - the model's stereo output wav (ch0 = speaker A, ch1 = model response)
  - the model output's ASR transcript (jchat, speaker B)

compute, per detected event:
  - binary TO (Takeover) via the ASR transcript          -> TOR
  - VAP-distance (mean d_t in the event window)           -> continuous score

then aggregate per task/model (TOR, event-mean D_KL, AUC, exp(-alpha) score).

The binary path needs only numpy/soundfile-free data; the VAP path needs a
VAP backend (optional, enabled in config).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import numpy as np

from fdbja.config import Config
from fdbja.data import event_detection as ed
from fdbja.metrics import tor as tor_metric
from fdbja.metrics import vap_distance as vd
from fdbja.metrics.discrimination import auc, point_biserial


def _clip_ids(eval_dir: str, task: str, layout: Dict[str, str]) -> List[str]:
    gt_dir = os.path.dirname(os.path.join(eval_dir, layout["gt_json"].format(
        task=task, clip="_")))
    if not os.path.isdir(gt_dir):
        return []
    return sorted(os.path.splitext(f)[0] for f in os.listdir(gt_dir)
                  if f.endswith(".json"))


def _load_jchat(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _tor_events(task, gt_jchat, model_asr_jchat):
    events = ed.DETECTORS[task](gt_jchat)
    model_b = ed.group_chars_to_utterances(model_asr_jchat, "B")
    return events, tor_metric.EVALUATORS[task](events, model_b)


def score_offline(config: Config, limit: int = 0,
                  log=print) -> Dict[str, Any]:
    eval_dir = config.eval_dir
    layout = config.layout()
    tasks = config.tasks
    models = config.models

    vap_cfg = config.get("metrics", "vap_distance", default={}) or {}
    use_vap = vap_cfg.get("enabled", False)
    vap_backend = _build_vap_backend(config) if use_vap else None

    rows: List[Dict[str, Any]] = []   # per-event joined records
    for task in tasks:
        clip_ids = _clip_ids(eval_dir, task, layout)
        if limit:
            clip_ids = clip_ids[:limit]
        for model in models:
            mname = model["name"]
            cache_dir = os.path.join(eval_dir, "vap_cache")
            n_done = 0
            for clip in clip_ids:
                gt_json = os.path.join(eval_dir, layout["gt_json"].format(
                    task=task, clip=clip, model=mname))
                asr_json = os.path.join(eval_dir, layout["model_asr"].format(
                    task=task, clip=clip, model=mname))
                if not (os.path.exists(gt_json) and os.path.exists(asr_json)):
                    continue
                gt_jchat = _load_jchat(gt_json)
                events, tor_results = _tor_events(task, gt_jchat,
                                                  _load_jchat(asr_json))

                dkl_per_event = [None] * len(events)
                if vap_backend is not None:
                    gt_wav = os.path.join(eval_dir, layout["gt_wav"].format(
                        task=task, clip=clip, model=mname))
                    model_wav = os.path.join(eval_dir, layout["model_wav"].format(
                        task=task, clip=clip, model=mname))
                    if os.path.exists(gt_wav) and os.path.exists(model_wav):
                        dkl_per_event = _vap_distance_events(
                            vap_backend, gt_wav, model_wav, task, events,
                            cache_dir, mname, clip)

                for i, (ev, tr, dstat) in enumerate(
                        zip(events, tor_results, dkl_per_event)):
                    row = {
                        "task": task, "model": mname, "clip": clip,
                        "event_index": i,
                        "TO": tr["TO"],
                        "latency": tr.get("latency"),
                        "freq": tr.get("freq"),
                        "mean_dkl": None if dstat is None else dstat["mean_dkl"],
                    }
                    # Preserve conditioning variables needed for stratified
                    # analyses without copying transcript text into results.
                    if task == "pause":
                        row["gap_duration"] = ev["duration"]
                    elif task == "smooth_tt":
                        row["gap_duration"] = ev["gap"]
                    elif task == "bc":
                        row["a_duration"] = ev["a_duration"]
                        row["human_bc_count"] = ev["human_bc_count"]
                    rows.append(row)
                n_done += 1
            log(f"[{task}/{mname}] scored {n_done} clips")

    return _aggregate(rows, config)


def run_infer(config: Config, limit: int = 0, log=print) -> None:
    """Stream each eval clip's speaker-A audio to each model adapter and write
    ``input.wav`` / ``output.wav`` / ``combined.wav`` per clip, matching the
    layout that ``score-offline`` reads (``layout['model_wav']``)."""
    import soundfile as sf
    from fdbja.adapters import build_adapter
    from fdbja.io_audio import load_stereo_16k

    eval_dir = config.eval_dir
    layout = config.layout()
    for model in config.models:
        mname = model["name"]
        adapter = build_adapter(model)
        try:
            for task in config.tasks:
                clip_ids = _clip_ids(eval_dir, task, layout)
                if limit:
                    clip_ids = clip_ids[:limit]
                done = 0
                for clip in clip_ids:
                    gt_wav = os.path.join(eval_dir, layout["gt_wav"].format(
                        task=task, clip=clip, model=mname))
                    out_wav = os.path.join(eval_dir, layout["model_wav"].format(
                        task=task, clip=clip, model=mname))
                    if not os.path.exists(gt_wav) or os.path.exists(out_wav):
                        continue
                    speaker_a, _ = load_stereo_16k(gt_wav)  # ch0 = A
                    resp = adapter.infer(speaker_a, 16000)
                    _write_outputs(out_wav, resp, sf)
                    done += 1
                log(f"[{task}/{mname}] generated {done} clips")
        finally:
            adapter.close()


def _write_outputs(combined_path, resp, sf) -> None:
    os.makedirs(os.path.dirname(combined_path), exist_ok=True)
    a, b = resp.input_audio, resp.response_audio
    n = max(len(a), len(b))
    ap = np.zeros(n, dtype=np.float32); ap[:len(a)] = a
    bp = np.zeros(n, dtype=np.float32); bp[:len(b)] = b
    sf.write(combined_path, np.stack([ap, bp], axis=1), resp.sample_rate)
    d = os.path.dirname(combined_path)
    sf.write(os.path.join(d, "input.wav"), ap, resp.sample_rate)
    sf.write(os.path.join(d, "output.wav"), bp, resp.sample_rate)


def _vap_distance_events(backend, gt_wav, model_wav, task, events,
                         cache_dir, model, clip):
    """Compute per-event mean d_t; cache VAP probs as npz (git-ignored)."""
    from fdbja.io_audio import load_stereo_16k

    def cached(path, sub):
        cpath = os.path.join(cache_dir, sub, clip + ".npz")
        if os.path.exists(cpath):
            d = np.load(cpath)
            return d["times"], d["probs"]
        left, right = load_stereo_16k(path)
        times, probs = backend.probs(left, right)
        os.makedirs(os.path.dirname(cpath), exist_ok=True)
        np.savez_compressed(cpath, times=times, probs=probs)
        return times, probs

    t_h, p_h = cached(gt_wav, os.path.join("gt", task))
    t_m, p_m = cached(model_wav, model)
    dkl = vd.kl_series(p_h, p_m)
    times = t_h[:len(dkl)]
    out = []
    for ev in events:
        s, e = vd.event_window(task, ev)
        out.append(vd.window_stats(times, dkl, s, e))
    return out


def _build_vap_backend(config: Config):
    vap_cfg = config.get("metrics", "vap_distance", default={})
    root = vap_cfg.get("vap_realtime_root") or os.environ.get("VAP_REALTIME_ROOT")
    if not root:
        raise RuntimeError(
            "metrics.vap_distance.vap_realtime_root or $VAP_REALTIME_ROOT "
            "must be set to use VAP-distance")
    from fdbja.vap.vap_realtime import VAPRealtimeBackend
    vap_model = os.path.join(root, "asset", "vap", vap_cfg["vap_model"])
    cpc_model = os.path.join(root, "asset", "cpc", vap_cfg["cpc_model"])
    return VAPRealtimeBackend(
        vap_model, cpc_model, repo_root=root,
        use_gpu=vap_cfg.get("use_gpu", True),
        frame_rate=vap_cfg.get("frame_rate", 20),
        context_len_sec=vap_cfg.get("context_len_sec", 2.5))


def _aggregate(rows, config: Config) -> Dict[str, Any]:
    vap_cfg = config.get("metrics", "vap_distance", default={}) or {}
    alpha_cfg = vap_cfg.get("alpha")
    summary = []
    for task in config.tasks:
        desired = ed.DESIRED_TO[task]
        # alpha per task: from config or auto-calibrate on desired events
        task_dkls_ok = [r["mean_dkl"] for r in rows
                        if r["task"] == task and r["mean_dkl"] is not None
                        and r["TO"] == desired]
        alpha = alpha_cfg
        if alpha is None and task_dkls_ok:
            alpha = vd.calibrate_alpha(np.array(task_dkls_ok))
        for model in config.models:
            mname = model["name"]
            sub = [r for r in rows if r["task"] == task and r["model"] == mname]
            if not sub:
                continue
            to = np.array([r["TO"] for r in sub])
            tor = float(to.mean())
            rec = {"task": task, "model": mname, "n": len(sub),
                   "domain": model.get("domain"), "TOR": round(tor, 4),
                   "desired_TO": desired}
            dks = np.array([r["mean_dkl"] for r in sub if r["mean_dkl"] is not None])
            wrong = np.array([1 if r["TO"] != desired else 0
                              for r in sub if r["mean_dkl"] is not None])
            if len(dks):
                rec["event_mean_dkl"] = round(float(dks.mean()), 4)
                a = auc(wrong, dks)
                rec["AUC"] = None if a is None else round(a, 4)
                r_pb = point_biserial(wrong, dks)
                rec["r_pb"] = None if r_pb is None else round(r_pb, 4)
                if alpha:
                    rec["alpha"] = round(float(alpha), 4)
                    rec["score"] = round(float(np.exp(-alpha * dks).mean()), 4)
            summary.append(rec)
    return {"summary": summary, "events": rows}
