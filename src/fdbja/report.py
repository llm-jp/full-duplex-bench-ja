"""Result exporters: JSON file and a printed summary table
(``summary.json`` / ``summary.csv`` are written by :mod:`fdbja.summary`)."""

from __future__ import annotations

import json
import os
from typing import Any, Dict


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
