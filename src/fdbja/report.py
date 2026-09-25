"""Result exporters: JSON file and a printed summary table."""

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
    cols = ["task", "model", "n", "domain", "TOR",
            "event_mean_dkl", "AUC", "score"]
    header = " | ".join(f"{c:>14}" if c not in ("task", "model") else f"{c:<16}"
                        for c in cols)
    lines = [header, "-" * len(header)]
    for r in results.get("summary", []):
        cells = []
        for c in cols:
            v = r.get(c, "")
            v = "" if v is None else v
            cells.append(f"{str(v):<16}" if c in ("task", "model")
                         else f"{str(v):>14}")
        lines.append(" | ".join(cells))
    return "\n".join(lines)
