"""Offline scoring runner: score precomputed model outputs.

Given, per task and model:
  - GT stereo wav + GT jchat (the evaluation clip and its human reference)
  - the model's stereo output wav (ch0 = speaker A, ch1 = model response)
  - the model output's ASR transcript (jchat, speaker B)

compute, per detected event:
  - binary TO (Takeover) via the ASR transcript          -> TOR
    (``detection.system_activity: audio``: the model's speech intervals from
    the output wav, the ASR only for the characters)
  - VAP-distance (mean d_t in the event window)           -> continuous score

then aggregate per task/model (TOR, event-mean D_KL, AUC, exp(-alpha) score).

The binary path needs only numpy/soundfile-free data; the VAP path needs a
VAP backend (optional, enabled in config). ``run_transcribe`` produces the ASR
transcript from the model output wav with the backend chosen in ``asr.backend``.
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


def _detect_events(task, gt_jchat, gt_wav=None, boundaries="transcript",
                   roles=None, params=None):
    """Events of a GT clip. ``boundaries="audio"`` keeps the
    events found in the transcript but moves their boundaries onto the human
    speech of the GT wav (``data/event_boundaries.py``); ``"transcript"`` is
    the transcript-boundary definition (character timestamps). ``roles``
    (``ed.SmoothTTRoles``) drops Smooth TT events that are not a
    hand-over from A to B (text checks; with audio boundaries also B's voice
    activity before ``a_end``)."""
    events = ed.detect(task, gt_jchat, roles, params)
    if boundaries == "audio":
        if gt_wav is None or not os.path.exists(gt_wav):
            raise FileNotFoundError(
                f"detection.boundaries=audio needs the GT wav: {gt_wav}")
        from fdbja.data.event_boundaries import drop_role_flagged, refine_from_wav
        events = refine_from_wav(task, events, gt_wav, roles=roles)
        if task == "smooth_tt" and roles is not None and roles.enabled:
            events = drop_role_flagged(events)
    elif boundaries != "transcript":
        raise ValueError(f"detection.boundaries must be transcript|audio, got {boundaries!r}")
    return events


def _tor_events(task, gt_jchat, model_asr_jchat, events=None,
                system_activity="transcript", model_wav=None, sys_cfg=None,
                out_utts=None, overlap_tolerance=0.0):
    """Events and their TO results. ``system_activity="audio"`` takes the
    model's speech intervals from the output wav (``data/system_activity.py``)
    and uses the ASR only for the characters; ``"transcript"`` takes both from
    the ASR character timestamps (the published definition). The third value
    is ``None`` or the per-clip statistics of the audio path. ``out_utts``
    (a list) receives the model utterances the decisions were made on."""
    if events is None:
        events = ed.DETECTORS[task](gt_jchat)
    stats = None
    if system_activity == "audio":
        if model_wav is None or not os.path.exists(model_wav):
            raise FileNotFoundError(
                f"detection.system_activity=audio needs the model wav: {model_wav}")
        from fdbja.data import system_activity as sa
        model_b, stats = sa.utterances_from_wav(
            model_asr_jchat, model_wav, "B", sys_cfg or sa.DEFAULT_SYSTEM_ACTIVITY)
    elif system_activity == "transcript":
        model_b = ed.group_chars_to_utterances(model_asr_jchat, "B")
    else:
        raise ValueError("detection.system_activity must be transcript|audio, "
                         f"got {system_activity!r}")
    if out_utts is not None:
        out_utts.extend(model_b)
    kw = {"overlap_tolerance": overlap_tolerance} if task == "pause" and overlap_tolerance else {}
    return events, tor_metric.EVALUATORS[task](events, model_b, **kw), stats


#: the window each task judges the model in (clip seconds), as in metrics/tor.py
JUDGE_WINDOW = {
    "pause": lambda ev: (ev["start"], ev["end"]),
    "smooth_tt": lambda ev: (ev["a_end"] - 0.5, ev["a_end"] + 2.0),
}


def score_offline(config: Config, limit: int = 0,
                  log=print, models=None) -> Dict[str, Any]:
    """Score every model entry whose outputs are in place (any adapter,
    including ``offline`` and adapters this installation cannot build:
    scoring reads only files). ``models`` restricts it to these names."""
    from fdbja.summary import read_asr_meta

    eval_dir = config.eval_dir
    layout = config.layout()
    tasks = config.tasks
    wanted = set(models) if models else None
    models = [m for m in config.models if wanted is None or m["name"] in wanted]
    if wanted is not None and len(models) < len(wanted):
        known = {m["name"] for m in config.models}
        log(f"[fdbja] not in the config: {', '.join(sorted(wanted - known))}")
    asr_backends: Dict[str, Any] = {}

    vap_cfg = config.get("metrics", "vap_distance", default={}) or {}
    use_vap = vap_cfg.get("enabled", False)
    vap_backend = _build_vap_backend(config) if use_vap else None
    vap_path = _vap_model_path(vap_cfg) if use_vap else None
    cache_tag = _vap_cache_tag(vap_cfg, vap_path) if use_vap else None
    boundary = _boundary_settings(vap_cfg) if use_vap else None
    pr_raw = config.get("metrics", "pause_response", default={}) or {}
    pr_cfg = None
    if pr_raw.get("enabled", False):
        from fdbja.metrics.pause_response import PauseResponseConfig
        pr_cfg = PauseResponseConfig.from_config(pr_raw)
    boundaries = config.get("detection", "boundaries", default="transcript")
    roles = ed.SmoothTTRoles.from_config(
        config.get("detection", "smooth_tt_roles", default=None))
    system_activity = config.get("detection", "system_activity", default="transcript")
    sys_cfg = None
    if system_activity == "audio":
        from fdbja.data.system_activity import SystemActivityConfig
        sys_cfg = SystemActivityConfig.from_config(
            config.get("detection", "system_vad", default=None))
    elif system_activity != "transcript":
        raise ValueError("detection.system_activity must be transcript|audio, "
                         f"got {system_activity!r}")
    event_cache: Dict[Any, list] = {}
    det_params = {t: config.detector_params(t) for t in tasks}

    rows: List[Dict[str, Any]] = []   # per-event joined records
    for task in tasks:
        clip_ids = _clip_ids(eval_dir, task, layout)
        if limit:
            clip_ids = clip_ids[:limit]
        for model in models:
            mname = model["name"]
            # Cached VAP probabilities are only valid for the checkpoint that
            # produced them, so key the cache by the checkpoint file name.
            vap_cfg = config.get("metrics", "vap_distance", default={}) or {}
            ckpt_stem = os.path.splitext(os.path.basename(
                str(vap_cfg.get("vap_model", "vap"))))[0]
            cache_dir = os.path.join(eval_dir, "vap_cache", ckpt_stem)
            n_done = 0
            n_wav_only = 0
            for clip in clip_ids:
                gt_json = os.path.join(eval_dir, layout["gt_json"].format(
                    task=task, clip=clip, model=mname))
                asr_json = os.path.join(eval_dir, layout["model_asr"].format(
                    task=task, clip=clip, model=mname))
                if not (os.path.exists(gt_json) and os.path.exists(asr_json)):
                    if os.path.exists(os.path.join(eval_dir, layout["model_wav"].format(
                            task=task, clip=clip, model=mname))):
                        n_wav_only += 1
                    continue
                if n_done == 0:
                    meta = read_asr_meta(os.path.dirname(asr_json))
                    asr_backends[f"{task}/{mname}"] = meta and meta.get("backend")
                gt_jchat = _load_jchat(gt_json)
                key = (task, clip)
                if key not in event_cache:
                    event_cache[key] = _detect_events(
                        task, gt_jchat,
                        os.path.join(eval_dir, layout["gt_wav"].format(
                            task=task, clip=clip, model=mname)),
                        boundaries, roles, det_params[task])
                model_wav = os.path.join(eval_dir, layout["model_wav"].format(
                    task=task, clip=clip, model=mname))
                model_utts: list = []
                events, tor_results, sys_stats = _tor_events(
                    task, gt_jchat, _load_jchat(asr_json), events=event_cache[key],
                    system_activity=system_activity, model_wav=model_wav,
                    sys_cfg=sys_cfg, out_utts=model_utts,
                    overlap_tolerance=float(
                        det_params.get(task, {}).get("overlap_tolerance", 0.0)))

                dkl_per_event = [None] * len(events)
                bdkl_per_event = [None] * len(events)
                gt_wav = os.path.join(eval_dir, layout["gt_wav"].format(
                    task=task, clip=clip, model=mname))
                have_wavs = os.path.exists(gt_wav) and os.path.exists(model_wav)
                if vap_backend is not None and have_wavs:
                    task_boundary = (boundary if boundary is not None
                                     and task in boundary["tasks"] else None)
                    dkl_per_event, bdkl_per_event = _vap_distance_events(
                        vap_backend, gt_wav, model_wav, task, events,
                        cache_dir, mname, clip, boundary=task_boundary,
                        cache_tag=cache_tag)
                pr_per_event = [None] * len(events)
                if pr_cfg is not None and task == "pause" and have_wavs:
                    from fdbja.metrics.pause_response import judge_events
                    pr_per_event = judge_events(gt_wav, model_wav, events, pr_cfg)

                for i, (ev, tr, dstat, bstat, prj) in enumerate(
                        zip(events, tor_results, dkl_per_event, bdkl_per_event,
                            pr_per_event)):
                    row = {
                        "task": task, "model": mname, "clip": clip,
                        # the role checks drop Smooth TT events; keep the unrestricted index
                        "event_index": ev.get("source_index", i),
                        "TO": tr["TO"],
                        "latency": tr.get("latency"),
                        "onset": tr.get("onset"),
                        "freq": tr.get("freq"),
                        "mean_dkl": None if dstat is None else dstat["mean_dkl"],
                    }
                    if boundary is not None and task in boundary["tasks"]:
                        row["vapd_boundary"] = None if bstat is None else bstat["mean_dkl"]
                    if pr_cfg is not None and task == "pause":
                        for k in ("no_response", "interrupt", "pause_response"):
                            row[k] = None if prj is None else prj[k]
                    # Preserve conditioning variables needed for stratified
                    # analyses without copying transcript text into results.
                    if sys_stats is not None:
                        # per-clip: ASR characters that no speech segment took
                        row["sys_unassigned_chars"] = sys_stats["unassigned_chars"]
                    if "boundary" in ev:
                        row["boundary_flags"] = "|".join(ev["boundary"]["flags"])
                    if task == "pause":
                        row["gap_duration"] = ev["duration"]
                        row["t_start"], row["t_end"] = ev["start"], ev["end"]
                    elif task == "smooth_tt":
                        row["gap_duration"] = ev["gap"]
                        row["t_start"], row["t_end"] = ev["a_end"], ev["b_start"]
                    if task in JUDGE_WINDOW:
                        # model utterances near the judgement window, for
                        # event-level reports (analyze-profile)
                        w0, w1 = JUDGE_WINDOW[task](ev)
                        row["sys_segments"] = [
                            [round(u["start"], 3), round(u["end"], 3), int(u["num_chars"])]
                            for u in model_utts
                            if u["end"] > w0 - 1.0 and u["start"] < w1 + 1.0]
                    elif task == "bc":
                        row["a_duration"] = ev["a_duration"]
                        row["human_bc_count"] = ev["human_bc_count"]
                        row["human_freq"] = ev["human_freq"]
                        row["bc_count"] = tr.get("bc_count")
                        row["jsd"] = tr.get("jsd")
                    rows.append(row)
                n_done += 1
            msg = f"[{task}/{mname}] scored {n_done} clips"
            if n_wav_only:
                msg += (f" ({n_wav_only} with combined.wav but no transcript: "
                        "run fdbja transcribe)")
            elif n_done == 0:
                msg += " (no outputs: run infer + transcribe, or place combined.wav)"
            log(msg)

    return _aggregate(rows, config, models=models, asr_backends=asr_backends,
                      vap_path=vap_path)


def run_infer(config: Config, limit: int = 0, log=print, models=None) -> int:
    """Stream each eval clip's speaker-A audio to each model adapter and write
    ``input.wav`` / ``output.wav`` / ``combined.wav`` per clip, matching the
    layout that ``score-offline`` reads (``layout['model_wav']``).

    ``models`` restricts the run to these model names. Entries with
    ``adapter: offline`` (or no adapter) are not run: their ``combined.wav``
    files are expected to be placed by hand, and the number still missing is
    reported. An entry whose adapter cannot be built is reported and skipped,
    so the other models still run. Returns the number of skipped entries
    (0 = every requested adapter ran).
    """
    import soundfile as sf

    from fdbja.adapters import build_adapter, is_offline
    from fdbja.io_audio import load_stereo_16k

    eval_dir = config.eval_dir
    layout = config.layout()
    wanted = set(models) if models else None
    failed = 0
    for model in config.models:
        mname = model["name"]
        if wanted is not None and mname not in wanted:
            continue
        if is_offline(model):
            for task in config.tasks:
                clip_ids = _clip_ids(eval_dir, task, layout)
                if limit:
                    clip_ids = clip_ids[:limit]
                missing = sum(
                    1 for clip in clip_ids
                    if not os.path.exists(os.path.join(eval_dir, layout["model_wav"].format(
                        task=task, clip=clip, model=mname))))
                log(f"[{task}/{mname}] offline: nothing to run, "
                    f"{len(clip_ids) - missing}/{len(clip_ids)} combined.wav in place")
            continue
        try:
            adapter = build_adapter(model)
        except (ValueError, ImportError, TypeError) as e:
            log(f"[{mname}] skipped: {e}")
            failed += 1
            continue
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
    return failed


def run_transcribe(config: Config, limit: int = 0, models=None, tasks=None,
                   backend: str = None, overwrite: bool = False,
                   device: str = None, log=print) -> int:
    """Transcribe each model response (``layout['model_wav']``, ch1) into the
    jchat file that ``score_offline`` reads (``layout['model_asr']``).

    The backend comes from ``--backend``, else ``asr.backend`` in the config;
    ``asr.options`` are passed to its constructor. Existing transcripts are
    kept unless ``overwrite``. Returns the number of clips transcribed.
    """
    from fdbja.asr import DEFAULT_BACKEND, load_asr
    from fdbja.summary import write_asr_meta

    asr_cfg = config.get("asr", default={}) or {}
    name = backend or asr_cfg.get("backend") or DEFAULT_BACKEND
    options = dict(asr_cfg.get("options", {}) or {})
    if device:
        options["device"] = device
    gate = dict(asr_cfg.get("silence_gate", {}) or {})
    asr = load_asr(name, **options)
    log(f"[asr] backend={name} options={options} silence_gate={gate or 'default'}")

    eval_dir = config.eval_dir
    layout = config.layout()
    wanted = set(models) if models else None
    total = 0
    for model in config.models:
        mname = model["name"]
        if wanted is not None and mname not in wanted:
            continue
        for task in (tasks or config.tasks):
            clip_ids = _clip_ids(eval_dir, task, layout)
            if limit:
                clip_ids = clip_ids[:limit]
            done = skipped = 0
            for clip in clip_ids:
                model_wav = os.path.join(eval_dir, layout["model_wav"].format(
                    task=task, clip=clip, model=mname))
                asr_json = os.path.join(eval_dir, layout["model_asr"].format(
                    task=task, clip=clip, model=mname))
                if not os.path.exists(model_wav):
                    continue
                if os.path.exists(asr_json) and not overwrite:
                    skipped += 1
                    continue
                transcribe_file(asr, model_wav, asr_json, gate=gate)
                if done == 0:
                    write_asr_meta(os.path.dirname(asr_json), name, options, gate)
                done += 1
            log(f"[{task}/{mname}] transcribed {done} clips"
                + (f" (kept {skipped} existing)" if skipped else ""))
            total += done
    return total


def transcribe_file(asr, wav_path: str, out_json: str, channel: int = 1,
                    speaker: str = "B", gate: Dict[str, Any] = None
                    ) -> List[Dict[str, Any]]:
    """Transcribe one response channel of ``wav_path`` and write jchat JSON.

    ``channel`` selects the response channel of a stereo file (ch1 = model
    response in ``combined.wav``); a mono file is used as is. ``gate``
    overrides keys of :data:`fdbja.asr.base.DEFAULT_GATE` (config
    ``asr.silence_gate``): a silent channel gets ``[]`` without calling the ASR.
    """
    import soundfile as sf

    from fdbja.asr import validate_jchat
    from fdbja.asr.base import DEFAULT_GATE, silence_gate

    data, sr = sf.read(wav_path, dtype="float32")
    if data.ndim == 2:
        data = data[:, min(channel, data.shape[1] - 1)]
    data = np.ascontiguousarray(data)
    opts = {**DEFAULT_GATE, **(gate or {})}
    items = validate_jchat(silence_gate(
        data, sr, lambda: asr.transcribe(data, sr, speaker=speaker), **opts))
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    return items


def transcribe_channels(asr, wav_path: str, out_json: str, channels, speakers,
                        gate: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Transcribe several channels of ``wav_path`` into one jchat file.

    ``speakers[i]`` labels the characters of ``channels[i]``. With
    ``channels=[0, 1]`` and ``speakers=["A", "B"]`` this turns a two-channel
    dialogue recording into the transcript ``reconstruct`` reads (speaker
    labels ``A`` / ``B`` are required by the event detectors). One channel
    gives the same file as :func:`transcribe_file`.
    """
    import tempfile

    if len(channels) != len(speakers):
        raise ValueError("one speaker label per channel")
    if len(channels) == 1:
        return transcribe_file(asr, wav_path, out_json, channel=channels[0],
                               speaker=speakers[0], gate=gate)
    items: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        for ch, spk in zip(channels, speakers):
            items += transcribe_file(asr, wav_path, os.path.join(tmp, f"{ch}.json"),
                                     channel=ch, speaker=spk, gate=gate)
    items.sort(key=lambda c: (c["start"], c["speaker"]))
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    return items


def _write_outputs(combined_path, resp, sf) -> None:
    os.makedirs(os.path.dirname(combined_path), exist_ok=True)
    a, b = resp.input_audio, resp.response_audio
    n = max(len(a), len(b))
    ap = np.zeros(n, dtype=np.float32)
    ap[:len(a)] = a
    bp = np.zeros(n, dtype=np.float32)
    bp[:len(b)] = b
    sf.write(combined_path, np.stack([ap, bp], axis=1), resp.sample_rate)
    d = os.path.dirname(combined_path)
    sf.write(os.path.join(d, "input.wav"), ap, resp.sample_rate)
    sf.write(os.path.join(d, "output.wav"), bp, resp.sample_rate)


def _boundary_settings(vap_cfg):
    """Boundary weighting of ``metrics.vap_distance`` or ``None`` (uniform)."""
    weighting = vap_cfg.get("weighting", "uniform") or "uniform"
    if weighting == "uniform":
        return None
    if weighting != "boundary":
        raise ValueError("metrics.vap_distance.weighting must be uniform|boundary, "
                         f"got {weighting!r}")
    tau = float(vap_cfg.get("tau_seconds", vd.DEFAULT_TAU))
    if tau <= 0:
        raise ValueError(f"metrics.vap_distance.tau_seconds must be positive, got {tau}")
    tasks = list(vap_cfg.get("boundary_tasks", ["smooth_tt"]) or [])
    bad = set(tasks) - {"pause", "smooth_tt"}
    if bad:
        raise ValueError(f"metrics.vap_distance.boundary_tasks: {sorted(bad)} "
                         "(boundary weighting is defined for pause / smooth_tt)")
    return {"tau": tau, "tasks": tasks,
            "pre": float(vap_cfg.get("boundary_pre_seconds", vd.BOUNDARY_PRE)),
            "post": float(vap_cfg.get("boundary_post_seconds", vd.BOUNDARY_POST))}


def _vap_model_path(vap_cfg) -> str:
    root = (vap_cfg.get("vap_realtime_root")
            or os.environ.get("VAP_REALTIME_ROOT") or "")
    return os.path.join(root, "asset", "vap", str(vap_cfg.get("vap_model", "")))


def _vap_cache_tag(vap_cfg, vap_path=None):
    """What a cached VAP probability depends on besides the input audio: the
    checkpoint (SHA-256 of the file when it is readable, else its name; with
    ``vap_revision`` the Hub revision too) and the frame settings.
    ``metrics.vap_distance.cache: trust`` returns ``None`` (reuse any cached
    file, the behaviour before 0.2.0)."""
    from fdbja.summary import file_sha256

    mode = vap_cfg.get("cache", "validate") or "validate"
    if mode == "trust":
        return None
    if mode != "validate":
        raise ValueError(f"metrics.vap_distance.cache must be validate|trust, got {mode!r}")
    ckpt = file_sha256(vap_path) if vap_path else None
    cpc = vap_cfg.get("cpc_model")
    return "|".join(str(x) for x in (
        ckpt or os.path.basename(str(vap_cfg.get("vap_model", ""))),
        vap_cfg.get("vap_revision", ""), cpc,
        vap_cfg.get("frame_rate", 20), vap_cfg.get("context_len_sec", 2.5)))


def vap_cache_key(wav_path: str, cache_tag: str) -> str:
    """Cache key of one VAP probability file: SHA-256 over the input wav's
    bytes and ``cache_tag`` (checkpoint and frame settings)."""
    import hashlib

    from fdbja.summary import file_sha256

    return hashlib.sha256(
        f"{file_sha256(wav_path)}|{cache_tag}".encode("utf-8")).hexdigest()


def _vap_distance_events(backend, gt_wav, model_wav, task, events,
                         cache_dir, model, clip, boundary=None, cache_tag=None):
    """Per-event mean d_t and (with ``boundary``) the boundary-weighted mean;
    cache VAP probs as npz (git-ignored). Returns ``(stats, boundary_stats)``.

    With ``cache_tag`` (the default ``metrics.vap_distance.cache: validate``)
    each file stores :func:`vap_cache_key` of its input wav and is recomputed
    when the wav, the checkpoint or the frame settings changed, or when it
    carries no key (written before 0.2.0). Model outputs are cached per task
    (``<model>/<task>/<clip>.npz``) so clip names may repeat across tasks."""
    from fdbja.io_audio import load_stereo_16k

    def cached(path, sub):
        cpath = os.path.join(cache_dir, sub, clip + ".npz")
        key = None if cache_tag is None else vap_cache_key(path, cache_tag)
        if os.path.exists(cpath):
            with np.load(cpath) as d:
                if key is None or ("key" in d.files and str(d["key"]) == key):
                    return d["times"], d["probs"]
        left, right = load_stereo_16k(path)
        times, probs = backend.probs(left, right)
        os.makedirs(os.path.dirname(cpath), exist_ok=True)
        extra = {} if key is None else {"key": np.array(key)}
        np.savez_compressed(cpath, times=times, probs=probs, **extra)
        return times, probs

    t_h, p_h = cached(gt_wav, os.path.join("gt", task))
    t_m, p_m = cached(model_wav, model if cache_tag is None else os.path.join(model, task))
    dkl = vd.kl_series(p_h, p_m)
    times = t_h[:len(dkl)]
    out = []
    bout = []
    for ev in events:
        s, e = vd.event_window(task, ev)
        out.append(vd.window_stats(times, dkl, s, e))
        if boundary is None:
            bout.append(None)
        else:
            lo, hi, t0 = vd.boundary_window(task, ev, boundary["pre"], boundary["post"])
            bout.append(vd.weighted_window_stats(times, dkl, lo, hi, t0, boundary["tau"]))
    return out, bout


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


def _aggregate(rows, config: Config, models=None, asr_backends=None,
               vap_path=None) -> Dict[str, Any]:
    from fdbja import summary as sm

    vap_cfg = config.get("metrics", "vap_distance", default={}) or {}
    alpha_cfg = vap_cfg.get("alpha")
    if alpha_cfg is not None and not float(alpha_cfg) > 0:
        raise ValueError("metrics.vap_distance.alpha must be positive or null "
                         f"(auto-calibrate), got {alpha_cfg!r}")
    # the trivial baselines (silence / always_talk) stay out of the alpha
    # calibration when asked: silence is "desired" on every Pause event
    calib_skip = set()
    if vap_cfg.get("calibration_exclude_baselines", False):
        from fdbja.adapters import BASELINE_ADAPTERS
        calib_skip = {m["name"] for m in (config.models or [])
                      if m.get("adapter") in BASELINE_ADAPTERS}
    boot = {**sm.DEFAULT_BOOTSTRAP,
            **(config.get("output", "bootstrap", default={}) or {})}
    unit = sm.bootstrap_unit(config)

    def ci(vals, recs, s):
        clusters = [r["clip"] for r in recs] if unit == "clip" else None
        return sm.bootstrap_ci(vals, boot["n"], s, clusters=clusters)
    models = config.models if models is None else models
    summary = []
    for task in config.tasks:
        desired = ed.DESIRED_TO[task]
        # alpha per task: from config or auto-calibrate on desired events
        task_dkls_ok = [r["mean_dkl"] for r in rows
                        if r["task"] == task and r["mean_dkl"] is not None
                        and r["TO"] == desired and r["model"] not in calib_skip]
        alpha = alpha_cfg
        if alpha is None and task_dkls_ok:
            alpha = vd.calibrate_alpha(np.array(task_dkls_ok))
            if alpha is None:
                import warnings
                warnings.warn(f"[{task}] alpha calibration skipped: the median D_KL of the "
                              "desired events is not positive (no score column)",
                              RuntimeWarning, stacklevel=2)
        for model in models:
            mname = model["name"]
            sub = [r for r in rows if r["task"] == task and r["model"] == mname]
            if not sub:
                continue
            to = np.array([r["TO"] for r in sub])
            tor = float(to.mean())
            seed = sm.stream_seed(boot["seed"], task, mname)
            # keys and order follow docs/results.json; n = events
            rec = {"task": task, "model": mname, "n": len(sub),
                   "n_clips": len({r["clip"] for r in sub}),
                   "domain": model.get("domain"), "desired_TO": desired,
                   "TOR": round(tor, 4),
                   "TOR_ci": ci(to, sub, seed)}
            if task == "bc":
                # backchannel frequency (per second of A) and timing JSD,
                # as in the original Full-Duplex-Bench Backchannel task
                freqs = [r["freq"] for r in sub if r.get("freq") is not None]
                hfreqs = [r["human_freq"] for r in sub if r.get("human_freq") is not None]
                jsds = [r["jsd"] for r in sub if r.get("jsd") is not None]
                if freqs:
                    rec["bc_freq"] = round(float(np.mean(freqs)), 4)
                if hfreqs:
                    rec["human_freq"] = round(float(np.mean(hfreqs)), 4)
                if jsds:
                    rec["jsd"] = round(float(np.mean(jsds)), 4)
                # JSD is 1.0 whenever the model never backchannels, so also
                # report how often it did and the JSD restricted to those events
                anyb = [r for r in sub if r.get("bc_count")]
                rec["bc_any_rate"] = round(len(anyb) / len(sub), 4)
                jsd_c = [r["jsd"] for r in anyb if r.get("jsd") is not None]
                if jsd_c:
                    rec["jsd_cond"] = round(float(np.mean(jsd_c)), 4)
            dks = np.array([r["mean_dkl"] for r in sub if r["mean_dkl"] is not None])
            wrong = np.array([1 if r["TO"] != desired else 0
                              for r in sub if r["mean_dkl"] is not None])
            if len(dks):
                rec["event_mean_dkl"] = round(float(dks.mean()), 4)
                rec["dkl_ci"] = ci(dks, [r for r in sub if r["mean_dkl"] is not None],
                                   seed + 1)
                a = auc(wrong, dks)
                rec["AUC"] = None if a is None else round(a, 4)
                r_pb = point_biserial(wrong, dks)
                rec["r_pb"] = None if r_pb is None else round(r_pb, 4)
                if alpha is not None:
                    rec["alpha"] = round(float(alpha), 4)
                    rec["score"] = round(float(np.exp(-alpha * dks).mean()), 4)
            b_rows = [r for r in sub if r.get("vapd_boundary") is not None]
            bks = [r["vapd_boundary"] for r in b_rows]
            if bks:
                rec["vapd_boundary"] = round(float(np.mean(bks)), 4)
                rec["vapd_boundary_ci"] = ci(bks, b_rows, seed + 2)
            prs = [r for r in sub if r.get("pause_response") is not None]
            if prs:
                rec.update(sm.pause_response_stats(prs, boot["n"], seed + 3, unit=unit))
            if task == "smooth_tt":
                rec.update(sm.timing_stats(sub))
            else:
                rec["lat_median"] = None
            summary.append(rec)
    conditions = sm.scoring_conditions(config, asr_backends or {}, vap_path)
    return {"summary": summary, **conditions, "events": rows}
