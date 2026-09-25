"""Gap-stratified analysis for the Pause Handling task.

This reuses existing ASR outputs and cached VAP probabilities, so no model
API calls or GPU inference are required.
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from fdbja.data import event_detection as ed
from fdbja.metrics import tor as tor_metric
from fdbja.metrics import vap_distance as vd


BINS = (
    ("0.4-0.6", 0.4, 0.6),
    ("0.6-0.8", 0.6, 0.8),
    ("0.8-1.0", 0.8, 1.000001),
)


def gap_bin(duration: float) -> str | None:
    for label, low, high in BINS:
        if low <= duration < high:
            return label
    return None


def _load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _discover_models(eval_dir: Path) -> list[str]:
    prefix, suffix = "pause_", "_asr"
    return sorted(
        p.name[len(prefix):-len(suffix)]
        for p in eval_dir.glob(f"{prefix}*{suffix}")
        if p.is_dir() and not p.name.startswith("pause_baseline_")
    )


def _cached_dkl(eval_dir: Path, model: str, clip: str,
                events: list[dict]) -> list[float | None]:
    gt_path = eval_dir / "vap_cache" / "gt" / "pause" / f"{clip}.npz"
    model_path = eval_dir / "vap_cache" / model / f"{clip}.npz"
    if not gt_path.exists() or not model_path.exists():
        return [None] * len(events)
    with np.load(gt_path) as gt, np.load(model_path) as pred:
        dkl = vd.kl_series(gt["probs"], pred["probs"])
        times = gt["times"][:len(dkl)]
        return [vd.window_stats(times, dkl, *vd.event_window("pause", ev))
                ["mean_dkl"] for ev in events]


def collect_rows(eval_dir: str | Path,
                 models: Iterable[str] | None = None) -> list[dict]:
    root = Path(eval_dir)
    selected = list(models) if models else _discover_models(root)
    rows = []
    for gt_path in sorted((root / "pause").glob("*.json")):
        clip = gt_path.stem
        events = ed.detect_pauses(_load_json(gt_path))
        for model in selected:
            asr_path = root / f"pause_{model}_asr" / "text" / f"{clip}.json"
            if not asr_path.exists():
                continue
            model_b = ed.group_chars_to_utterances(_load_json(asr_path), "B")
            tor = tor_metric.eval_pause(events, model_b)
            dkls = _cached_dkl(root, model, clip, events)
            for index, (event, to_result, mean_dkl) in enumerate(
                    zip(events, tor, dkls)):
                duration = float(event["duration"])
                label = gap_bin(duration)
                if label is None:
                    continue
                rows.append({
                    "model": model,
                    "clip": clip,
                    "event_index": index,
                    "gap_duration": duration,
                    "gap_bin": label,
                    "TO": int(to_result["TO"]),
                    "mean_dkl": mean_dkl,
                })
    return rows


def _cluster_bootstrap(rows: list[dict], field: str, n_boot: int,
                       rng: np.random.Generator) -> tuple[float, float]:
    by_clip = defaultdict(list)
    for row in rows:
        value = row[field]
        if value is not None:
            by_clip[row["clip"]].append(float(value))
    clips = list(by_clip)
    if not clips:
        return float("nan"), float("nan")
    samples = np.empty(n_boot)
    for i in range(n_boot):
        chosen = rng.choice(clips, size=len(clips), replace=True)
        values = [v for clip in chosen for v in by_clip[clip]]
        samples[i] = np.mean(values)
    return tuple(float(x) for x in np.quantile(samples, [0.025, 0.975]))


def aggregate(rows: list[dict], n_boot: int = 10000,
              seed: int = 42) -> list[dict]:
    rng = np.random.default_rng(seed)
    output = []
    models = sorted({r["model"] for r in rows})
    for model in models:
        bin_records = []
        for label, _, _ in BINS:
            sub = [r for r in rows
                   if r["model"] == model and r["gap_bin"] == label]
            if not sub:
                continue
            tor = float(np.mean([r["TO"] for r in sub]))
            tor_lo, tor_hi = _cluster_bootstrap(sub, "TO", n_boot, rng)
            dkl_values = [r["mean_dkl"] for r in sub
                          if r["mean_dkl"] is not None]
            dkl = float(np.mean(dkl_values)) if dkl_values else None
            dkl_ci = (_cluster_bootstrap(sub, "mean_dkl", n_boot, rng)
                      if dkl_values else (None, None))
            rec = {
                "model": model, "gap_bin": label,
                "n_events": len(sub),
                "n_clips": len({r["clip"] for r in sub}),
                "TOR": tor, "TOR_ci_low": tor_lo, "TOR_ci_high": tor_hi,
                "mean_dkl": dkl,
                "mean_dkl_ci_low": dkl_ci[0],
                "mean_dkl_ci_high": dkl_ci[1],
            }
            output.append(rec)
            bin_records.append(rec)
        if len(bin_records) == len(BINS):
            output.append({
                "model": model, "gap_bin": "macro",
                "n_events": sum(r["n_events"] for r in bin_records),
                "n_clips": len({r["clip"] for r in rows
                                if r["model"] == model}),
                "TOR": float(np.mean([r["TOR"] for r in bin_records])),
                "TOR_ci_low": None, "TOR_ci_high": None,
                "mean_dkl": (float(np.mean([r["mean_dkl"] for r in bin_records]))
                             if all(r["mean_dkl"] is not None
                                    for r in bin_records) else None),
                "mean_dkl_ci_low": None, "mean_dkl_ci_high": None,
            })
    return output


def trend_summary(rows: list[dict], n_boot: int = 10000,
                  seed: int = 42) -> list[dict]:
    """Estimate the long-minus-short TOR difference by clip bootstrap."""
    rng = np.random.default_rng(seed)
    output = []
    for model in sorted({r["model"] for r in rows}):
        sub = [r for r in rows if r["model"] == model]
        by_clip = defaultdict(list)
        for row in sub:
            by_clip[row["clip"]].append(row)
        clips = list(by_clip)

        def difference(sampled_clips):
            sampled = [r for clip in sampled_clips for r in by_clip[clip]]
            short = [r["TO"] for r in sampled if r["gap_bin"] == "0.4-0.6"]
            long = [r["TO"] for r in sampled if r["gap_bin"] == "0.8-1.0"]
            if not short or not long:
                return None
            return float(np.mean(long) - np.mean(short))

        observed = difference(clips)
        boot = []
        for _ in range(n_boot):
            sampled = rng.choice(clips, size=len(clips), replace=True)
            value = difference(sampled)
            if value is not None:
                boot.append(value)
        lo, hi = np.quantile(boot, [0.025, 0.975])
        output.append({
            "model": model,
            "TOR_long_minus_short": observed,
            "ci_low": float(lo),
            "ci_high": float(hi),
            "bootstrap_probability_positive": float(np.mean(np.array(boot) > 0)),
        })
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, summary: list[dict], trends: list[dict]) -> None:
    lines = [
        "# Pause gap-stratified analysis", "",
        "Clip-level bootstrap 95% confidence intervals are shown in brackets.",
        "Macro is the unweighted mean of the three gap bins.", "",
        "| Model | Gap (s) | Events | Clips | TOR (95% CI) | Mean D_KL |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        if r["gap_bin"] == "macro":
            tor_text = f'{r["TOR"]:.3f}'
        else:
            tor_text = (f'{r["TOR"]:.3f} '
                        f'[{r["TOR_ci_low"]:.3f}, {r["TOR_ci_high"]:.3f}]')
        dkl = "-" if r["mean_dkl"] is None else f'{r["mean_dkl"]:.3f}'
        lines.append(f'| {r["model"]} | {r["gap_bin"]} | '
                     f'{r["n_events"]} | {r["n_clips"]} | {tor_text} | {dkl} |')
    lines.extend([
        "", "## Long-vs-short pause effect", "",
        "Positive values mean more erroneous takeovers for 0.8-1.0 s pauses.",
        "", "| Model | TOR difference | 95% CI | P(positive) |",
        "|---|---:|---:|---:|",
    ])
    for r in trends:
        lines.append(
            f'| {r["model"]} | {r["TOR_long_minus_short"]:.3f} | '
            f'[{r["ci_low"]:.3f}, {r["ci_high"]:.3f}] | '
            f'{r["bootstrap_probability_positive"]:.3f} |')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(path: Path, summary: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    labels = [b[0] for b in BINS]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for model in sorted({r["model"] for r in summary}):
        recs = {r["gap_bin"]: r for r in summary
                if r["model"] == model and r["gap_bin"] != "macro"}
        if not all(label in recs for label in labels):
            continue
        y = np.array([recs[label]["TOR"] for label in labels])
        lo = np.array([recs[label]["TOR_ci_low"] for label in labels])
        hi = np.array([recs[label]["TOR_ci_high"] for label in labels])
        ax.errorbar(labels, y, yerr=[y - lo, hi - y], marker="o",
                    capsize=3, label=model)
    ax.set_xlabel("Intra-turn gap duration (s)")
    ax.set_ylabel("Erroneous takeover rate")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(eval_dir: str, out_dir: str, models: Iterable[str] | None = None,
        n_boot: int = 10000, seed: int = 42) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(eval_dir, models)
    if not rows:
        raise RuntimeError("No Pause events joined with model ASR outputs")
    summary = aggregate(rows, n_boot=n_boot, seed=seed)
    trends = trend_summary(rows, n_boot=n_boot, seed=seed)
    _write_csv(out / "pause_events.csv", rows)
    _write_csv(out / "pause_summary.csv", summary)
    (out / "pause_analysis.json").write_text(
        json.dumps({"bins": [b[0] for b in BINS], "summary": summary,
                    "trends": trends},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_markdown(out / "pause_summary.md", summary, trends)
    _plot(out / "pause_tor_by_gap.png", summary)
    print(f"[pause-analysis] {len(rows)} model-event rows -> {out}")
    return {"events": rows, "summary": summary, "trends": trends}
