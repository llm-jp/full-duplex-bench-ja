"""Human-evaluation sampling and manifest generation.

Builds crowdsourcing manifests (CSV) for the two validation studies described
in docs/human_eval_design.md:

  Study A (ground-truth validity): per detected event, a third party judges
    the appropriate partner behavior, compared against the human-B-as-annotator
    label. Stimulus = speaker-A audio up to the decision point.

  Study B (metric validity): per (event x model), an annotator rates the
    model's timing naturalness. Items are stratified so events where models
    agree and events where they disagree on binary TO are both represented
    (the disagreement items are the most informative for showing VAP-distance
    captures graded quality the binary metric misses). Stimulus = the model's
    combined.wav (speaker A + model response).

Outputs CSV manifests (+ optional rendered Study-A stimuli). Audio is read
from the eval_dir; nothing corpus-derived is committed.
"""

from __future__ import annotations

import csv
import json
import os
import random
from typing import Dict, List

from fdbja.config import Config
from fdbja.data import event_detection as ed
from fdbja.metrics import tor as tor_metric

# ---------------------------------------------------------------------------
# Finalized survey questions (Japanese)
# ---------------------------------------------------------------------------

# global instructions shown on the start screen (Higashinaka-lab style
# breakdown: what to do / what not to penalize / absolute rating)
INTRO_NOTES = [
    "実際の日本語会話から切り出した短い音声(20秒程度)を聴いて回答します。"
    "項目はランダムな順に並びます。",
    "各音声は途中で止めずに最後まで聴いてから回答してください"
    "(最後まで再生すると回答欄が押せるようになります)。聞き直しは自由です。",
    "音声は左右で話者が分かれている場合があります(左=相手の話者、右=応答)。"
    "イヤホンまたはヘッドホンを必ず使用してください。",
    "短い相槌(「うん」「はい」など)や軽い声の重なりは、人間同士の会話では"
    "自然に起こります。それ自体を減点しないでください。",
    "会話の一部を切り出しているため、冒頭や末尾が急に始まる/切れることが"
    "あります。その点は評価に含めないでください。",
    "応答の声は合成音声(機械の声)のことがあります。声質や機械っぽさは"
    "評価対象ではありません。各設問の説明に従ってください。",
    "各音声はほかの音声と比べず、それぞれ単独で(絶対評価で)判断してください。",
]

STUDY_A_BREAKDOWN = (
    "会話の途中まで(相手の話者がひと区切りするところまで)を聴きます。"
    "あなたがこの会話の聞き手だったら、この直後にどう振る舞うのが最も自然かを"
    "選んでください。正解を当てるクイズではありません。あなた自身の感覚で"
    "判断してください。")
STUDY_A_QUESTION = "今聞いた発話の直後、対話の相手として最も自然な振る舞いはどれですか?"
STUDY_A_CHOICES = [
    "すぐに話し始める(ターンを取る)",
    "黙って相手の発話を待つ/聞き続ける",
    "短い相槌だけ打つ(「うん」「はい」など)",
    "どちらとも言えない/判断が難しい",
]
# maps the auto label (desired TO + task) to the "correct" Study-A choice index
#   smooth_tt desired TO=1 -> choice 0 (take turn)
#   pause     desired TO=0 -> choice 1 (stay silent)
#   bc        desired TO=0 -> choice 2 (backchannel only)
STUDY_A_EXPECTED = {"smooth_tt": 0, "pause": 1, "bc": 2}

STUDY_B_BREAKDOWN = (
    "相手の話者(左)に対するシステムの応答(右)の「間の取り方・タイミング」"
    "だけを評価してください。"
    "【見るところ】話し始めるタイミング、沈黙の長さ、相槌の入れ方、"
    "声の重なり方が、人間の聞き手として自然かどうか。"
    "【見ないところ】声質や合成音声らしさ、話している内容の良し悪し、"
    "聞き取りやすさは評価に含めないでください。声が機械的でも、タイミングが"
    "自然なら高く評価して構いません。"
    "【低く評価する例】相手の発話に不必要に割り込む/相手が少し間を置いた"
    "だけなのに話し始める/話すべきところで黙り続ける/相槌が過剰・場違い/"
    "返答までの間が不自然に長い。"
    "自然な相槌や軽い重なりは減点しないでください。")
STUDY_B_Q_MAIN = ("このシステムの応答の「間の取り方・発話のタイミング」は、"
                  "人間同士の会話として自然に聞こえましたか?")
STUDY_B_MAIN_SCALE = "1: まったく自然ではない 〜 5: とても自然である"
STUDY_B_MAIN_SCALE_LABELS = [
    "1. まったく自然ではない",
    "2. あまり自然ではない",
    "3. どちらともいえない",
    "4. やや自然である",
    "5. とても自然である",
]
STUDY_B_Q_CAT = "このシステムは実際に何をしましたか?"
STUDY_B_CATEGORIES = [
    "適切なタイミングで話し始めた",
    "話すべきでない所で割り込んだ/喋りすぎた",
    "話すべき所で黙ってしまった",
    "適切に相槌を打った",
    "相槌が過剰/不自然だった",
    "その他",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decision_time(task: str, ev: dict) -> float:
    """Time at which the partner must decide what to do (Study-A truncation)."""
    if task == "pause":
        return ev["start"]          # onset of the intra-turn gap
    if task == "smooth_tt":
        return ev["a_end"]          # end of A's turn
    if task == "bc":
        return ev["a_utt_end"]      # end of A's (backchannel-eliciting) span
    raise ValueError(task)


def _clip_ids(eval_dir: str, task: str, layout: Dict[str, str]) -> List[str]:
    gt_dir = os.path.dirname(os.path.join(eval_dir, layout["gt_json"].format(
        task=task, clip="_")))
    if not os.path.isdir(gt_dir):
        return []
    return sorted(os.path.splitext(f)[0] for f in os.listdir(gt_dir)
                  if f.endswith(".json"))


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _per_event_to(task, gt_jchat, asr_jchat):
    events = ed.DETECTORS[task](gt_jchat)
    b = ed.group_chars_to_utterances(asr_jchat, "B")
    return events, [r["TO"] for r in tor_metric.EVALUATORS[task](events, b)]


# ---------------------------------------------------------------------------
# Study A
# ---------------------------------------------------------------------------

def build_study_a(config: Config, n_per_task: int, seed: int,
                  render_audio: bool, out_dir: str, log=print) -> List[dict]:
    eval_dir = config.eval_dir
    layout = config.layout()
    rows = []
    for task in config.tasks:
        # enumerate all events across clips
        pool = []
        for clip in _clip_ids(eval_dir, task, layout):
            gt = _load(os.path.join(eval_dir, layout["gt_json"].format(
                task=task, clip=clip)))
            for i, ev in enumerate(ed.DETECTORS[task](gt, **config.detector_params(task))):
                pool.append((clip, i, ev))
        rng = random.Random(seed)
        rng.shuffle(pool)
        # session-diverse: prefer distinct clips first
        seen, picked = set(), []
        for clip, i, ev in pool:
            if len(picked) >= n_per_task:
                break
            if clip in seen and len(picked) < n_per_task * 0.8:
                continue
            seen.add(clip)
            picked.append((clip, i, ev))

        for k, (clip, i, ev) in enumerate(picked):
            item_id = f"A_{task}_{k:03d}"
            dt = _decision_time(task, ev)
            if render_audio:
                _render_study_a_stimulus(eval_dir, layout, task, clip, dt,
                                         os.path.join(out_dir, "audio_a",
                                                      item_id + ".wav"))
            rows.append({
                "item_id": item_id, "study": "A", "task": task, "clip": clip,
                "event_index": i, "decision_time": round(dt, 3),
                "expected_choice": STUDY_A_EXPECTED[task],
                "question": STUDY_A_QUESTION,
                "choices": " | ".join(STUDY_A_CHOICES),
                "audio": f"audio_a/{item_id}.wav" if render_audio else "",
            })
    _write_csv(os.path.join(out_dir, "study_a.csv"), rows)
    log(f"Study A: {len(rows)} items -> {out_dir}/study_a.csv")
    return rows


def _render_study_a_stimulus(eval_dir, layout, task, clip, decision_time,
                             out_path, lead_out=0.2):
    """Mono speaker-A audio from clip start to the decision point (+lead_out)."""
    import soundfile as sf
    gt_wav = os.path.join(eval_dir, layout["gt_wav"].format(task=task, clip=clip))
    data, sr = sf.read(gt_wav, dtype="float32")
    a = data[:, 0] if data.ndim == 2 else data
    end = min(len(a), int((decision_time + lead_out) * sr))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sf.write(out_path, a[:end], sr)


# ---------------------------------------------------------------------------
# Study B
# ---------------------------------------------------------------------------

def build_study_b(config: Config, n_events_per_task: int, seed: int,
                  out_dir: str, render_audio: bool, log=print) -> List[dict]:
    import shutil
    eval_dir = config.eval_dir
    layout = config.layout()
    models = [m["name"] for m in config.models]
    rows = []
    for task in config.tasks:
        clips = _clip_ids(eval_dir, task, layout)
        # per clip: events + each model's per-event TO (None if ASR missing)
        per_event = []  # (clip, event_index, ev, {model: TO})
        for clip in clips:
            gt = _load(os.path.join(eval_dir, layout["gt_json"].format(
                task=task, clip=clip)))
            events = ed.DETECTORS[task](gt, **config.detector_params(task))
            to_by_model = {}
            for m in models:
                asr_p = os.path.join(eval_dir, layout["model_asr"].format(
                    task=task, clip=clip, model=m))
                if os.path.exists(asr_p):
                    _, tos = _per_event_to(task, gt, _load(asr_p))
                    to_by_model[m] = tos
            for i, ev in enumerate(events):
                tos = {m: to_by_model[m][i] for m in to_by_model
                       if i < len(to_by_model[m])}
                per_event.append((clip, i, ev, tos))

        # stratify: events where models DISAGREE on TO are most informative
        def disagree(tos):
            return len(set(tos.values())) > 1 if tos else False
        diverse = [e for e in per_event if disagree(e[3])]
        rest = [e for e in per_event if not disagree(e[3])]
        rng = random.Random(seed)
        rng.shuffle(diverse)
        rng.shuffle(rest)
        n_div = min(len(diverse), n_events_per_task // 2)
        chosen = diverse[:n_div] + rest[:n_events_per_task - n_div]
        chosen = chosen[:n_events_per_task]

        for k, (clip, i, ev, tos) in enumerate(chosen):
            for m in models:
                mc = os.path.join(eval_dir, layout["model_wav"].format(
                    task=task, clip=clip, model=m))
                if not os.path.exists(mc):
                    continue
                item_id = f"B_{task}_{k:03d}_{m}"
                audio_rel = os.path.relpath(mc, eval_dir)
                if render_audio:  # copy into bundle so it can be served
                    dst = os.path.join(out_dir, "audio_b", item_id + ".wav")
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copyfile(mc, dst)
                    audio_rel = f"audio_b/{item_id}.wav"
                rows.append({
                    "item_id": item_id,
                    "study": "B", "task": task, "clip": clip,
                    "event_index": i, "model": m,
                    "TO": tos.get(m, ""),
                    "desired_TO": ed.DESIRED_TO[task],
                    "models_disagree": int(disagree(tos)),
                    "audio": audio_rel,
                    "q_main": STUDY_B_Q_MAIN, "main_scale": STUDY_B_MAIN_SCALE,
                    "q_cat": STUDY_B_Q_CAT,
                    "categories": " | ".join(STUDY_B_CATEGORIES),
                })
    _write_csv(os.path.join(out_dir, "study_b.csv"), rows)
    n_dis = sum(1 for r in rows if r["models_disagree"] == 1)
    log(f"Study B: {len(rows)} items ({n_dis} from disagreement events) "
        f"-> {out_dir}/study_b.csv")
    return rows


def _write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        open(path, "w").close()
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def sample_human_eval(config: Config, out_dir: str, n_a: int = 60,
                      n_b: int = 50, seed: int = 42, per_worker: int = 30,
                      render_audio: bool = True, log=print) -> None:
    a_rows = build_study_a(config, n_a, seed, render_audio, out_dir, log)
    b_rows = build_study_b(config, n_b, seed, out_dir, render_audio, log)

    manifest = {
        "seed": seed,
        "max_per_worker": per_worker,
        "intro_notes": INTRO_NOTES,
        "study_a": {
            "breakdown": STUDY_A_BREAKDOWN,
            "question": STUDY_A_QUESTION,
            "choices": STUDY_A_CHOICES,
            "items": [{"item_id": r["item_id"], "task": r["task"],
                       "audio": r["audio"],
                       "expected_choice": r["expected_choice"]}
                      for r in a_rows],
        },
        "study_b": {
            "breakdown": STUDY_B_BREAKDOWN,
            "q_main": STUDY_B_Q_MAIN, "main_scale": STUDY_B_MAIN_SCALE,
            "main_scale_labels": STUDY_B_MAIN_SCALE_LABELS,
            "q_cat": STUDY_B_Q_CAT, "categories": STUDY_B_CATEGORIES,
            "items": [{"item_id": r["item_id"], "task": r["task"],
                       "model": r["model"], "audio": r["audio"],
                       "TO": r["TO"], "models_disagree": r["models_disagree"]}
                      for r in b_rows],
        },
    }
    with open(os.path.join(out_dir, "manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)

    # copy the static web UI into the bundle so the out dir is self-servable
    webui_src = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "webui")
    if os.path.isdir(webui_src):
        import shutil
        for fn in os.listdir(webui_src):
            shutil.copyfile(os.path.join(webui_src, fn),
                            os.path.join(out_dir, fn))
        log(f"web UI copied; serve {out_dir}/ and open index.html")
    else:
        log(f"manifest.json written; webui/ not found at {webui_src}")

