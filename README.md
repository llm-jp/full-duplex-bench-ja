# Full-Duplex-Bench-JA

**日本語フルデュプレックス音声対話モデル**のターンテイキング評価ツールキット。
[Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench) の
日本語版で、**人手注釈を必要としない**のが特徴です。実対話コーパスにおける
対話相手(人間 B)の振る舞いを正解として用います。

📊 **リーダーボード**: <https://llm-jp.github.io/full-duplex-bench-ja/>
(結果は `docs/results.json`。`score-offline` の出力を差し替えて更新。
各タスクのサンプル音声を波形付きで試聴可能 — 刺激音声との同時再生対応、
素材は `docs/samples/`、定義は `docs/meta.json` の `samples`)

> 状態(2026-07 時点): **private / 開発中。** 評価パイプライン(`reconstruct` → `infer` →
> `score-offline`)は LLM-jp-Zoom1 実データで 6 モデル
> (J-Moshi / J-Moshi-ext / LLM-jp-Moshi-v1 / Gemini-Live / GPT-Realtime /
> Grok-Voice)を完走済み。人手評価用の `sample-human-eval` /
> `analyze-human-eval` も実装済み(実施はこれから)。
> 下記「コマンドの状態」「既知の限界」参照。

## 何を評価するか

対話コーパスから 3 種類のターンテイキング事象を自動検出し、モデルの応答が
どれだけ人間らしいタイミングかを採点します。

| タスク | 事象 | モデルの望ましい挙動 |
|--------|------|----------------------|
| **Pause Handling** | ターン内の間(0.4–1.0 秒)で人間 B が沈黙 | 沈黙する(TO=0) |
| **Smooth Turn-Taking** | A(≥4 秒)→ 間(<0.4 秒)→ B(≥4 秒) | ターンを取る(TO=1) |
| **Backchanneling** | A(≥3 秒)中に人間 B が相槌のみ | 相槌に留める(TO=0) |

2 つの指標:

- **TOR**(二値 Takeover Rate)— 本家 FDB の指標。モデルがターンを取ったか否か。
- **VAP-distance**(連続)— 同じ話者 A の音声に対し、**人間**の応答と
  **モデル**の応答をそれぞれ凍結
  [VAP](https://github.com/inokoj/VAP-Realtime) に通し、次話者予測分布間の
  KL 距離を測る。二値が捨てる「タイミングの良し悪し」を連続値で捉える。

比較用に、同じ 256 クラス VAP 定式化を使う **TurnNat 系の尤度スコア**
(`metrics/turnnat.py`)も実装しています。参照応答を必要としない絶対尺度で、
VAP-distance の対照尺度としての位置づけは
[docs/related_metrics.md](docs/related_metrics.md) を参照。

いずれの指標にも退化解があるため(沈黙し続ければ Pause / BC は「正解」、
喋り続ければ Smooth TT は「正解」)、`silence_baseline` と
`always_talk_baseline`(`adapters/baselines.py`)を全タスクで併記します。

## アーキテクチャ

```
src/fdbja/
  data/event_detection.py   # pause / smooth_tt / bc 検出器(jchat 入力)
  metrics/tor.py            # 二値 Takeover Rate
  metrics/vap_distance.py   # 連続 VAP-distance(バックエンド非依存)
  metrics/turnnat.py        # TurnNat 系 NLL スコア(比較用ベースライン)
  metrics/discrimination.py # AUC・点双列相関
  adapters/                 # モデル adapter: 話者A音声を流し応答音声を録る
    base.py  moshi_ws.py  gemini_live.py  openai_realtime.py  _oggopus.py
    baselines.py            # 退化ベースライン(常時沈黙 / 常時発話)
  vap/                      # VAP バックエンド(参照: inokoj/VAP-Realtime)
  asr/                      # ASR バックエンド(参照: ReazonSpeech + WhisperX)
  config.py  runner.py  report.py  io_audio.py
  human_eval.py             # 人手評価バンドル生成(クラウドソーシング用)
  analyze_human_eval.py     # 回収した回答の集計・指標との照合
  cli.py                    # fdbja reconstruct | infer | score-offline
                            #       | sample-human-eval | analyze-human-eval
configs/default.yaml        # config 駆動の評価設定
webui/                      # 人手評価用の静的 Web UI(依存ライブラリなし)
```

config 駆動で、**データはコードと別レイヤ**として扱います(リポジトリには
同梱しません。[DATASET.md](DATASET.md) 参照)。自分のモデルを評価するには
`ModelAdapter`(`adapters/base.py`)を実装します。

## インストール(uv)

```bash
uv venv --python 3.10
uv pip install -e .                # コア(TOR のみ)
uv pip install -e '.[vap]'         # + VAP-distance バックエンド(torch)
uv pip install -e '.[infer]'       # + Moshi WebSocket 推論(opuslib, websockets)
uv pip install -e '.[gemini]'      # + Gemini Live 推論(google-genai)
uv pip install -e '.[realtime]'    # + GPT-Realtime / Grok-Voice 推論(websockets)
uv pip install -e '.[analysis]'    # + 図表
```

Blackwell GPU(sm_120)では torch を cu128 以上で入れる:

```bash
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## 使い方

```bash
# モデル出力を採点(実装・検証済み)。--limit でクリップ数を絞れる
uv run fdbja score-offline --config configs/default.yaml [--limit N]

# モデル出力を生成(要ライブ Moshi サーバ / GOOGLE_API_KEY /
# OPENAI_API_KEY / XAI_API_KEY — 使うモデルに応じて)
uv run fdbja infer --config configs/default.yaml [--limit N]

# コーパスから評価クリップを生成
uv run fdbja reconstruct --config configs/default.yaml

# 人手評価バンドルを生成(Web UI・刺激音声込み。docs/human_eval.md 参照)
uv run fdbja sample-human-eval --config configs/default.yaml --out ./human_eval

# 回収した回答(responses_*.json)を集計し自動指標と照合
uv run fdbja analyze-human-eval --responses <dir> --bundle ./human_eval \
    --results <results.json> --out ./human_eval_analysis

# 既存 Pause 出力をギャップ長別に再集計(API/GPU 推論は不要)
uv run fdbja analyze-pause --eval-dir /path/to/v4_eval \
    --out ./results/pause_gap_analysis \
    --models jmoshi jmoshi_ext llmjp gemini gpt_realtime grok_voice
```

`analyze-pause` は 0.4--0.6 秒、0.6--0.8 秒、0.8--1.0 秒の層別 TOR・
VAP-distance と、クリップ単位 bootstrap 信頼区間を出力する。

Backchannel は capability-specific な任意タスクとして扱う。サーバ側の
ターン検出によりユーザ発話中の出力が制約される API は、同時入出力可能な
モデルと同一ランキングで比較しない。

VAP-distance を使う場合は `VAP_REALTIME_ROOT`(または config の
`metrics.vap_distance.vap_realtime_root`)に
[VAP-Realtime](https://github.com/inokoj/VAP-Realtime) のパスを指定する。

## コマンドの状態

| コマンド | 状態 |
|----------|------|
| `reconstruct` | ✅ 実装・**検証済み**(LLM-jp-Zoom1 から 20 秒クリップ + manifest を生成、再検出で事象が保持されることを確認) |
| `infer` | ✅ 実装・**実データで完走済み**(Moshi 系 3 モデルは Moshi-WS、Gemini-Live、GPT-Realtime、Grok-Voice の計 6 モデルで全 600 クリップの応答生成を完了)。ライブサーバ/API キーが必要 |
| `score-offline` | ✅ 実装・**検証済み**(リファレンス評価セットで二値 TOR が既知値と完全一致。上記 6 モデルの単一実行での一括採点がリーダーボードの元データ。α はモデル集合に依存するため、モデル追加時は全モデルを 1 回の実行で採点し直すこと) |
| `sample-human-eval` | ✅ 実装済み(クラウドソーシング用バンドル生成。[docs/human_eval.md](docs/human_eval.md) 参照) |
| `analyze-human-eval` | ✅ 実装済み(回答の集計と自動指標との照合。人手評価の実施はこれから) |
| `analyze-pause` | ✅ 実装・検証済み(既存 6 モデルのキャッシュからギャップ長別に再集計) |

## 既知の限界

二値 TOR は **ASR 転写ベース**の判定(本家 Full-Duplex-Bench 準拠)のため、
多段 ASR パイプラインの誤りを継承する。実データで確認済みの 3 パターン:

1. **整列ずれによる偽陰性** — WhisperX の文字整列が発話開始を遅らせ、
   イベント窓との重なりを取りこぼす(開始時刻ずれ中央値 0.14 秒だが
   p90 は 0.94 秒。espnet セグメント時刻と比較すると TO 判定の 5〜31% が
   モデル×タスクにより不一致。Moshi 系ほど不一致が大きい)
2. **正解音声の欠落** — Zoom 話者別トラックの無音ゲーティングで人間 B の
   短い相槌が録音から消え、転写とのみ残る(転写された窓内相槌の約 1/3 が
   音響的に無音)。VAP-distance の正解参照にも影響し得る
3. **幽霊転写による偽陽性** — ほぼ無音のモデル出力に ASR が短語を
   長い区間で転写し、相槌免除(≤6 文字 かつ ≤1.5 秒)を外れて
   ターン奪取扱いになる

VAP-distance は生波形から計算するため 1・3 の影響を受けない(2 は要注意)。
これらは人手評価による指標の妥当性検証(実施予定)の動機でもある。

## ライセンス

コード: **Apache-2.0**(`LICENSE` 参照)。データおよびモデルチェックポイントは
対象外で、それぞれ独自の規約に従います([DATASET.md](DATASET.md) 参照)。

## 謝辞

[Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench)
(Lin et al., 2025) および
[VAP-Realtime](https://github.com/inokoj/VAP-Realtime)(Inoue et al.)に基づきます。
