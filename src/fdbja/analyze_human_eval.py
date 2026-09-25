"""Aggregate crowdsourced human-evaluation responses and validate the metrics.

Reads the ``responses_*.json`` files exported by the web UI plus the bundle's
``study_a.csv`` / ``study_b.csv`` (and, optionally, a ``score-offline``
``results.json`` to join VAP-distance), then reports:

  Study A (ground-truth validity, model-free)
    - agreement of the human-majority choice with the automatic label
    - Fleiss kappa / mean pairwise agreement, per task and overall
    - rate of "can't tell" responses

  Study B (metric validity)
    - ICC(1) of the timing-naturalness Likert
    - Spearman(VAP-distance, mean Likert) with bootstrap CI  (expect < 0)
    - binary TO-correct vs Likert (does the binary call track humans?)
    - CORE: the same Spearman *within* the binary-correct group (TO==desired)
      -- evidence that VAP-distance captures graded quality the binary drops
    - partial Spearman controlling for binary TO and latency

Outputs ``human_eval_analysis.json`` and a readable ``human_eval_summary.md``.
"""

from __future__ import annotations

import csv
import glob
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

from fdbja.metrics import agreement as ag
from fdbja.metrics.discrimination import auc

N_STUDY_A_CHOICES = 4   # incl. "can't tell"
UNCERTAIN_CHOICE = 3


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _read_csv(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_responses(responses_dir: str) -> List[dict]:
    """Flatten every responses_*.json into per-(worker, item) records."""
    out = []
    for path in sorted(glob.glob(os.path.join(responses_dir, "responses_*.json"))):
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        worker = doc.get("worker_id", os.path.basename(path))
        for r in doc.get("responses", []):
            out.append({"worker": worker, **r})
    return out


def _results_index(results_path: Optional[str]) -> Dict[tuple, dict]:
    """(task, clip, event_index, model) -> {mean_dkl, latency}."""
    if not results_path or not os.path.exists(results_path):
        return {}
    with open(results_path, encoding="utf-8") as f:
        doc = json.load(f)
    idx = {}
    for e in doc.get("events", []):
        key = (e["task"], e["clip"], e.get("event_index"), e["model"])
        idx[key] = {"mean_dkl": e.get("mean_dkl"), "latency": e.get("latency")}
    return idx


# ---------------------------------------------------------------------------
# Study A
# ---------------------------------------------------------------------------

def analyze_study_a(responses, study_a_csv) -> dict:
    meta = {r["item_id"]: r for r in study_a_csv}
    by_item = defaultdict(list)
    for r in responses:
        if r.get("study") != "A":
            continue
        ch = r.get("answer", {}).get("choice")
        if ch is not None:
            by_item[r["item_id"]].append(int(ch))

    per_task = defaultdict(lambda: {"ratings": [], "match": 0, "n": 0,
                                    "uncertain": 0, "judgments": 0})
    overall = {"ratings": [], "match": 0, "n": 0,
               "uncertain": 0, "judgments": 0}
    for item_id, choices in by_item.items():
        m = meta.get(item_id)
        if not m:
            continue
        task = m["task"]
        expected = int(m["expected_choice"])
        # majority vote (ties -> lowest index); ignore uncertain in the vote
        votes = [c for c in choices if c != UNCERTAIN_CHOICE] or choices
        maj = max(set(votes), key=lambda c: (votes.count(c), -c))
        for bucket in (per_task[task], overall):
            bucket["ratings"].append(choices)
            bucket["n"] += 1
            bucket["match"] += int(maj == expected)
            bucket["judgments"] += len(choices)
            bucket["uncertain"] += sum(1 for c in choices
                                       if c == UNCERTAIN_CHOICE)

    def summarize(b):
        table, _ = ag.to_count_table(b["ratings"], N_STUDY_A_CHOICES)
        return {
            "n_items": b["n"],
            "majority_accuracy": round(b["match"] / b["n"], 4) if b["n"] else None,
            "fleiss_kappa": _r(ag.fleiss_kappa(table)),
            "pct_agreement": _r(ag.pct_agreement(b["ratings"])),
            "uncertain_rate": round(b["uncertain"] / b["judgments"], 4)
                              if b["judgments"] else None,
            "mean_raters": round(b["judgments"] / b["n"], 2) if b["n"] else None,
        }

    return {"overall": summarize(overall),
            "by_task": {t: summarize(b) for t, b in sorted(per_task.items())}}


# ---------------------------------------------------------------------------
# Study B
# ---------------------------------------------------------------------------

def analyze_study_b(responses, study_b_csv, results_idx) -> dict:
    meta = {r["item_id"]: r for r in study_b_csv}
    main_by_item = defaultdict(list)
    for r in responses:
        if r.get("study") != "B":
            continue
        v = r.get("answer", {}).get("main")
        if v is not None:
            main_by_item[r["item_id"]].append(float(v))

    # per-item joined record
    items = []
    for item_id, ratings in main_by_item.items():
        m = meta.get(item_id)
        if not m:
            continue
        ev = results_idx.get((m["task"], m["clip"],
                              _int(m.get("event_index")), m["model"]), {})
        desired = _int(m.get("desired_TO"))
        to = _int(m.get("TO"))
        items.append({
            "item_id": item_id, "task": m["task"], "model": m["model"],
            "ratings": ratings, "mean_likert": float(np.mean(ratings)),
            "to": to, "desired_to": desired,
            "to_correct": None if to is None or desired is None
                          else int(to == desired),
            "mean_dkl": ev.get("mean_dkl"), "latency": ev.get("latency"),
            "models_disagree": _int(m.get("models_disagree")),
        })

    icc = ag.icc1([it["ratings"] for it in items])

    def corr_block(rows):
        d = [r["mean_dkl"] for r in rows if r["mean_dkl"] is not None]
        y = [r["mean_likert"] for r in rows if r["mean_dkl"] is not None]
        if len(d) < 3:
            return {"n": len(d), "spearman": None, "ci95": None}
        rho = ag.spearman(d, y)
        ci = ag.bootstrap_ci(d, y)
        return {"n": len(d), "spearman": _r(rho),
                "ci95": [round(c, 3) for c in ci] if ci else None}

    # binary call vs humans: does TO-correct separate higher Likert?
    corr_items = [r for r in items if r["to_correct"] is not None]
    likert = [r["mean_likert"] for r in corr_items]
    correct = [r["to_correct"] for r in corr_items]
    auc_to = auc(correct, likert) if corr_items else None

    # CORE: within the binary-correct group, does VAP-distance still rank humans?
    within_ok = corr_block([r for r in items if r["to_correct"] == 1])
    within_bad = corr_block([r for r in items if r["to_correct"] == 0])

    # partial Spearman: VAP-distance vs Likert | TO-correct, latency
    pr = [r for r in items if r["mean_dkl"] is not None]
    have_lat = [r["latency"] for r in pr if r["latency"] is not None]
    covars = [[r["to_correct"] for r in pr]]
    if len(have_lat) == len(pr) and pr:
        covars.append([r["latency"] for r in pr])
    partial = ag.partial_spearman([r["mean_dkl"] for r in pr],
                                  [r["mean_likert"] for r in pr],
                                  covars) if len(pr) >= 4 else None

    by_task = {}
    for t in sorted(set(r["task"] for r in items)):
        sub = [r for r in items if r["task"] == t]
        by_task[t] = {"n_items": len(sub), "overall": corr_block(sub),
                      "within_to_correct": corr_block(
                          [r for r in sub if r["to_correct"] == 1])}

    return {
        "n_items": len(items),
        "icc1_likert": _r(icc),
        "spearman_dkl_likert": corr_block(items),
        "auc_to_correct_vs_likert": _r(auc_to),
        "within_binary_class": {
            "to_correct": within_ok, "to_incorrect": within_bad},
        "partial_spearman_dkl_likert_given_to_latency": _r(partial),
        "covariates": ["to_correct"] + (["latency"]
                       if len(have_lat) == len(pr) and pr else []),
        "by_task": by_task,
        "_items": items,
    }


# ---------------------------------------------------------------------------
# Orchestration / reporting
# ---------------------------------------------------------------------------

def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _r(v, nd=4):
    return None if v is None else round(float(v), nd)


def run(responses_dir: str, bundle_dir: str, out_dir: str,
        results_path: Optional[str] = None, log=print) -> dict:
    responses = load_responses(responses_dir)
    study_a = _read_csv(os.path.join(bundle_dir, "study_a.csv"))
    study_b = _read_csv(os.path.join(bundle_dir, "study_b.csv"))
    results_idx = _results_index(results_path)

    n_workers = len(set(r["worker"] for r in responses))
    log(f"loaded {len(responses)} judgments from {n_workers} workers")
    if results_path and not results_idx:
        log(f"WARNING: no events joined from {results_path} "
            "(VAP-distance correlations will be empty)")

    out = {
        "n_judgments": len(responses),
        "n_workers": n_workers,
        "study_a": analyze_study_a(responses, study_a),
        "study_b": analyze_study_b(responses, study_b, results_idx),
    }
    os.makedirs(out_dir, exist_ok=True)
    items = out["study_b"].pop("_items")
    with open(os.path.join(out_dir, "human_eval_analysis.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    _write_item_csv(os.path.join(out_dir, "study_b_items.csv"), items)
    md = format_markdown(out)
    with open(os.path.join(out_dir, "human_eval_summary.md"), "w",
              encoding="utf-8") as f:
        f.write(md)
    print(md)
    log(f"wrote {out_dir}/human_eval_analysis.json + human_eval_summary.md")
    return out


def _write_item_csv(path, items):
    if not items:
        return
    cols = ["item_id", "task", "model", "mean_likert", "n_ratings",
            "to", "desired_to", "to_correct", "mean_dkl", "latency",
            "models_disagree"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for it in items:
            w.writerow({k: it.get(k) for k in cols if k != "n_ratings"} |
                       {"n_ratings": len(it["ratings"])})


def format_markdown(out: dict) -> str:
    a, b = out["study_a"], out["study_b"]
    L = ["# Full-Duplex-Bench-JA 人手評価 集計",
         "",
         f"判定数 {out['n_judgments']} / 作業者 {out['n_workers']} 名", ""]

    L += ["## Study A: 正解設計の妥当性", "",
          "| 範囲 | 項目 | 多数決一致率 | Fleiss κ | 一致率(pairwise) | "
          "判断困難率 | 平均評価者 |",
          "|---|--:|--:|--:|--:|--:|--:|"]

    def row_a(name, s):
        return (f"| {name} | {s['n_items']} | {_pct(s['majority_accuracy'])} | "
                f"{_f(s['fleiss_kappa'])} | {_pct(s['pct_agreement'])} | "
                f"{_pct(s['uncertain_rate'])} | {s['mean_raters']} |")

    L.append(row_a("全体", a["overall"]))
    for t, s in a["by_task"].items():
        L.append(row_a(t, s))

    sb = b["spearman_dkl_likert"]
    wc = b["within_binary_class"]
    L += ["", "## Study B: 指標の妥当性", "",
          f"- ICC(1) Likert(評価者間信頼性): **{_f(b['icc1_likert'])}**",
          f"- Spearman(VAP-distance, 人手平均 Likert): "
          f"**{_f(sb['spearman'])}** "
          f"(95%CI {sb['ci95']}, n={sb['n']}) — 負相関を期待",
          f"- AUC(二値 TO 正誤 → 人手 Likert): **{_f(b['auc_to_correct_vs_likert'])}**",
          "",
          "### 核心: 二値が同点とする群の内部での連続性",
          f"- 二値 **正解**群内の Spearman(dkl, Likert): "
          f"**{_f(wc['to_correct']['spearman'])}** "
          f"(95%CI {wc['to_correct']['ci95']}, n={wc['to_correct']['n']})",
          f"- 二値 **不正解**群内: {_f(wc['to_incorrect']['spearman'])} "
          f"(n={wc['to_incorrect']['n']})",
          f"- 偏相関(二値・遅延を統制): "
          f"**{_f(b['partial_spearman_dkl_likert_given_to_latency'])}** "
          f"(共変量: {', '.join(b['covariates'])})",
          "",
          "### タスク別",
          "| タスク | n | Spearman(全体) | Spearman(二値正解群内) |",
          "|---|--:|--:|--:|"]
    for t, s in b["by_task"].items():
        L.append(f"| {t} | {s['n_items']} | {_f(s['overall']['spearman'])} | "
                 f"{_f(s['within_to_correct']['spearman'])} |")
    L += ["",
          "> 二値正解群の内部でも負相関が出れば、VAP-distance は二値が捨てた"
          "品質差を人間と同じ向きで捉えている(= VAP が反応しているだけ、ではない)。",
          ""]
    return "\n".join(L)


def _f(v):
    return "—" if v is None else f"{v:.3f}"


def _pct(v):
    return "—" if v is None else f"{v * 100:.1f}%"
