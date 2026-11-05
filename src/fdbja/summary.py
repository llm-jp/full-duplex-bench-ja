"""Summary statistics shared by ``score-offline`` and the leaderboard.

* 95% intervals: percentile bootstrap over events (the leaderboard's
  ``"ci": "95% bootstrap intervals over events"``), one random stream per
  (task, model) so that adding a model does not change the others.
* Response timing on Smooth TT: ``lat_*`` are computed from ``latency``
  (clipped at 0, the leaderboard's ``lat_median``); ``onset_*`` from the signed
  start of the response relative to the end of A's utterance, so responses
  that overlap A (negative values, down to -0.5 s) stay visible.
* Scoring conditions (ASR backend, event boundaries, model speech intervals,
  VAP checkpoint) are recorded next to the numbers, under the same keys as
  ``docs/results.json``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import zlib
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

DEFAULT_BOOTSTRAP = {"n": 10000, "seed": 0}
CI_NOTE = "95% bootstrap intervals over events"


def stream_seed(base: int, *keys: str) -> int:
    """Seed of the bootstrap stream of one (task, model)."""
    return (int(base) + zlib.crc32("/".join(keys).encode("utf-8"))) % (2 ** 32)


def bootstrap_ci(values: Sequence[float], n_boot: int = 10000, seed: int = 0,
                 level: float = 0.95, stat=np.mean) -> Optional[List[float]]:
    """Percentile bootstrap interval of ``stat`` over ``values`` (events).

    Returns ``[low, high]`` or ``None`` when there are no values.
    """
    x = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if len(x) == 0:
        return None
    if n_boot <= 0:
        return None
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    chunk = max(1, 2_000_000 // max(1, len(x)))
    for i in range(0, n_boot, chunk):
        k = min(chunk, n_boot - i)
        idx = rng.integers(0, len(x), size=(k, len(x)))
        stats[i:i + k] = stat(x[idx], axis=1)
    lo, hi = np.percentile(stats, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return [round(float(lo), 4), round(float(hi), 4)]


def _q(x: np.ndarray, q: float) -> Optional[float]:
    return round(float(np.percentile(x, q)), 4) if len(x) else None


def timing_stats(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Response timing of the taken turns (Smooth TT rows with TO = 1)."""
    taken = [r for r in rows if r.get("TO") == 1]
    lat = np.array([r["latency"] for r in taken if r.get("latency") is not None])
    on = np.array([r["onset"] for r in taken if r.get("onset") is not None])
    return {
        "lat_n": int(len(lat)),
        "lat_median": _q(lat, 50),
        "lat_q1": _q(lat, 25),
        "lat_q3": _q(lat, 75),
        "onset_median": _q(on, 50),
        "onset_q1": _q(on, 25),
        "onset_q3": _q(on, 75),
        # share of the taken turns that started before A had finished
        "overlap_rate": round(float((on < 0).mean()), 4) if len(on) else None,
    }


def file_sha256(path: str) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# hidden and without a .json suffix so that globs over the transcripts skip it
ASR_META = ".fdbja-asr-meta"


def write_asr_meta(asr_dir: str, backend: str, options: Dict[str, Any],
                   gate: Dict[str, Any]) -> None:
    """Record which ASR wrote the transcripts of one ``<task>_<model>_asr``."""
    os.makedirs(asr_dir, exist_ok=True)
    meta = {"backend": backend,
            "options": {k: v for k, v in (options or {}).items() if k != "device"},
            "silence_gate": gate or {}}
    with open(os.path.join(asr_dir, ASR_META), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)


def read_asr_meta(asr_dir: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(asr_dir, ASR_META)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def scoring_conditions(config, asr_backends: Dict[str, Any],
                       vap_path: Optional[str] = None) -> Dict[str, Any]:
    """Scoring conditions under the keys of ``docs/results.json``.

    ``asr_backends`` maps ``task/model`` to the backend recorded by
    ``transcribe`` (``asr_meta.json``), or ``None`` when the transcripts were
    placed by hand.
    """
    from fdbja import __version__

    vap_cfg = config.get("metrics", "vap_distance", default={}) or {}
    roles = config.get("detection", "smooth_tt_roles", default={}) or {}
    names = sorted({str(v) for v in asr_backends.values() if v})
    unknown = sorted(k for k, v in asr_backends.items() if not v)
    asr = ", ".join(names) if names else None
    if unknown:
        asr = (asr + "; " if asr else "") + (
            f"unknown for {len(unknown)} task/model pairs (no {ASR_META}: "
            "transcripts not written by fdbja transcribe)")
    vap = None
    if vap_cfg.get("enabled"):
        sha = file_sha256(vap_path) if vap_path else None
        vap = os.path.basename(str(vap_cfg.get("vap_model", "")))
        vap += f" (sha256 {sha[:12]})" if sha else " (file not found)"
    return {
        "boundaries": config.get("detection", "boundaries", default="transcript"),
        "system_activity": config.get("detection", "system_activity", default="transcript"),
        "smooth_tt_roles": bool(roles.get("enabled", False)),
        "asr": asr,
        "asr_by_model": {k: v for k, v in sorted(asr_backends.items())},
        "vap_checkpoint": vap,
        "ci": CI_NOTE,
        "toolkit_version": __version__,
    }


_CSV_FIRST = ["task", "model", "domain", "n", "n_clips", "desired_TO", "TOR",
              "TOR_ci_low", "TOR_ci_high", "lat_n", "lat_median", "lat_q1", "lat_q3",
              "onset_median", "onset_q1", "onset_q3", "overlap_rate",
              "event_mean_dkl", "dkl_ci_low", "dkl_ci_high"]


def write_summary(results: Dict[str, Any], out_dir: str) -> List[str]:
    """``summary.json`` (summary + conditions, no per-event rows; the shape of
    ``docs/results.json``) and ``summary.csv`` (one row per task x model,
    intervals split into ``*_low`` / ``*_high``, conditions repeated)."""
    os.makedirs(out_dir, exist_ok=True)
    jpath = os.path.join(out_dir, "summary.json")
    payload = {k: v for k, v in results.items() if k != "events"}
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)

    flat = []
    for r in results.get("summary", []):
        row = {}
        for k, v in r.items():
            if isinstance(v, list) and len(v) == 2:
                row[f"{k}_low"], row[f"{k}_high"] = v
            else:
                row[k] = v
        for k in ("boundaries", "system_activity", "smooth_tt_roles", "asr",
                  "vap_checkpoint"):
            row[k] = results.get(k)
        flat.append(row)
    cols = [c for c in _CSV_FIRST if any(c in r for r in flat)]
    cols += sorted({k for r in flat for k in r} - set(cols))
    cpath = os.path.join(out_dir, "summary.csv")
    with open(cpath, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in flat:
            w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in cols})
    return [jpath, cpath]
