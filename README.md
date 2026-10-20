# Full-Duplex-Bench-JA

**日本語フルデュプレックス音声対話モデル**のターンテイキング評価ツールキット。
[Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench)(Lin et al., 2025)の
評価設計を日本語に持ち込んだもので、**人手注釈を必要としない**のが特徴です。
実対話コーパス(LLM-jp-Zoom1)における対話相手(人間 B)の振る舞いを正解として用い、
モデルの応答が「人間らしいタイミング」かどうかを採点します。

- リーダーボード: <https://llm-jp.github.io/full-duplex-bench-ja/>
  (`docs/` を GitHub Pages で配信。結果は `docs/results.json`、サンプル音声は `docs/samples/`)。
  順位は **TOR の適合率**(本家 Full-Duplex-Bench と同じ二値指標)で付けています。
  **VAP-distance は実験的指標**です。人手判断との相関や既存指標との比較による検証が済んでいないため参考値として併記し、
  既定の config では無効にしています(`metrics.vap_distance.enabled: true` で有効化)。
  掲載中の VAP-distance は既定の MIT チェックポイント(`maai-kyoto/vap_jp_kyoto`、下記)で 2026-09-28 に採点し直した値で、
  **チェックポイントが違う値は比較できません**(`docs/results.json` の `vap_checkpoint` に採点に使った重みを記録しています)。
  掲載は Pause Handling と Smooth Turn-Taking の 2 タスクです。bc(相槌)は転写の閾値に依存する実験的タスクとしてコードにだけ残し、既定では採点しません。
- 英語版 README: [README.en.md](README.en.md)

## 前提条件

| 用途 | 必要なもの |
|------|-----------|
| 採点(二値 TOR) | Python 3.10 以上、[uv](https://docs.astral.sh/uv/) |
| 採点(VAP-distance) | 上に加えて `.[vap]`(torch)、[VAP-Realtime](https://github.com/inokoj/VAP-Realtime) のローカル clone(CPC エンコーダ同梱)、日本語 VAP チェックポイント [`maai-kyoto/vap_jp_kyoto`](https://huggingface.co/maai-kyoto/vap_jp_kyoto) の `vap_state_dict_jp_kyoto_20hz_2500msec.pt`(MIT。リビジョン固定で取得)を `<VAP_REALTIME_ROOT>/asset/vap/` に置く。手順は下の「VAP-distance の準備」 |
| モデル出力の ASR 転写(`transcribe`) | いずれか 1 つの extra: `.[asr-reazon-whisperx]`(既定。ReazonSpeech-ESPnet + WhisperX。論文の数値を出した経路)/ `.[asr-nemo]`(ReazonSpeech NeMo v2、整列器なしで高速)/ `.[asr-k2]`(ReazonSpeech k2 v2、CPU 可)。モデルの重みは初回実行時に Hugging Face Hub から取得 |
| 評価クリップの生成 | LLM-jp-Zoom1 のローカルコピー([DATASET.md](DATASET.md)) |
| モデル推論(`infer`) | 評価対象に応じて Moshi 系 WebSocket サーバ、`GOOGLE_API_KEY`、`OPENAI_API_KEY`、`XAI_API_KEY` |

## クイックスタート

### 1. 動作確認(合成データ)

LLM-jp-Zoom1 が手元に無くても、合成の小さなコーパスでパイプライン全体
(`reconstruct` → `infer` → `transcribe` → `score-offline`)を通せます。
モデルサーバ・API キー・音声認識モデルは不要で、CPU で 1 分かかりません。
音声は振幅変調した雑音、文字は仮の記号なので、**出てくる数値に意味はありません**(インストールと配線の確認用)。

```bash
git clone https://github.com/llm-jp/full-duplex-bench-ja.git && cd full-duplex-bench-ja
uv sync                                                                          # コア + 開発ツール
uv run python examples/dummy/make_dummy_corpus.py --out data/dummy_zoom1          # 合成コーパス(10 セッション)
uv run fdbja reconstruct --config examples/dummy/config.yaml                     # 事象検出 → 20 秒クリップ + manifest
uv run fdbja infer --config examples/dummy/config.yaml                           # 自明なベースライン 2 本の応答を生成
uv run python examples/dummy/energy_asr.py --config examples/dummy/config.yaml   # 応答を jchat に(音量ベースの代用品)
uv run fdbja score-offline --config examples/dummy/config.yaml                   # 採点 → results/dummy/results.json
```

常時沈黙(`silence_baseline`)が両タスクで TOR 0、常時発話(`always_talk_baseline`)が Pause で TOR 1 になれば正常です
(代用品の転写は常時発話を 1 つの長い発話として返すので、Smooth TT の判定窓に発話の開始が無く TOR 0 になります)。
`examples/dummy/energy_asr.py` は、自前の ASR を `fdbja.asr.register_backend` で登録して `transcribe` に使う例にもなっています。
VAP-distance も確かめる場合は、下の「VAP-distance の準備」を済ませ、`examples/dummy/config.yaml` の
`metrics.vap_distance.enabled` を `true` にして `score-offline` を再実行します(CPU で数分)。

### 2. 実データで評価する

```bash
uv sync --extra asr-reazon-whisperx                         # ASR バックエンド(1 つ選ぶ。下の表を参照)
# configs/default.yaml の corpus.zoom1_root を LLM-jp-Zoom1 の場所に、models を評価対象に合わせて書き換える
uv run fdbja reconstruct --config configs/default.yaml      # LLM-jp-Zoom1 → 事象検出 → 20 秒クリップ + manifest
uv run fdbja infer --config configs/default.yaml            # 話者 A の音声をモデルに流し応答を録る
uv run fdbja transcribe --config configs/default.yaml       # モデル応答を文字単位の jchat に転写
uv run fdbja score-offline --config configs/default.yaml    # 評価ディレクトリの出力を採点
```

`score-offline` は `corpus.eval_dir`(既定 `./data/eval`)配下の次のファイルを読みます
(パスの雛形は config の `layout` で変更できます)。

```
data/eval/
  pause/<clip>.wav                       # 評価クリップ(ステレオ 16 kHz、ch0 = 話者 A、ch1 = 人間 B)
  pause/<clip>.json                      # 同クリップの jchat 転写(正解事象の検出に使う)
  pause_<model>_output/<clip>/combined.wav   # モデル出力(ch0 = 話者 A、ch1 = モデル応答)
  pause_<model>_asr/text/<clip>.json         # モデル応答の jchat 転写(TOR の判定に使う)
  smooth_tt/... , bc/...                 # 他タスクも同じ形
```

自分のモデルを採点する最短経路は、(1) `reconstruct` で評価クリップを作る、
(2) 各クリップの `combined.wav` を上の場所に置く(または `ModelAdapter` を実装して `infer` で生成する)、
(3) `transcribe` で応答側チャンネル(ch1)を jchat に転写する、(4) `score-offline` を走らせる、の 4 段です。
config の `models` に `name` と `domain: in|out` を追加してください。

### VAP-distance の準備(実験的指標を使う場合のみ)

```bash
uv sync --extra vap                                   # ASR の extra と併用するなら --extra を並べる
git clone https://github.com/inokoj/VAP-Realtime.git ../VAP-Realtime   # CPC エンコーダ(asset/cpc/)を同梱
curl -L -o ../VAP-Realtime/asset/vap/vap_state_dict_jp_kyoto_20hz_2500msec.pt \
  https://huggingface.co/maai-kyoto/vap_jp_kyoto/resolve/fe24ac60d8fcc80463edde97ed90e3ceca5e5b88/vap_state_dict_jp_kyoto_20hz_2500msec.pt
shasum -a 256 ../VAP-Realtime/asset/vap/vap_state_dict_jp_kyoto_20hz_2500msec.pt
# -> 7ad0bf0b59e17a30ee55cee2aa6db89b84d64b670881ccd3f74b52ccede05d29
export VAP_REALTIME_ROOT=$(cd ../VAP-Realtime && pwd)
```

このチェックポイントは 2026-10-01 に `maai-kyoto/vap_jp_kyoto` の `main` から外されたため、
上のようにリビジョン(`fe24ac6`)を固定して取得します。リーダーボードの VAP-distance はこのファイル
(SHA-256 が上と一致するもの)で採点しました。GPU が無い環境では config の `metrics.vap_distance.use_gpu: false` で CPU を使います。

## 主要コマンド

```bash
uv run fdbja reconstruct --config configs/default.yaml           # LLM-jp-Zoom1 → 事象検出 → 20 秒クリップ + manifest
uv run fdbja infer --config configs/default.yaml [--limit N]     # 話者 A の音声をモデルに流し応答を録る
uv run fdbja transcribe --config configs/default.yaml [--model NAME] [--task T] [--backend B]   # 応答を ASR して jchat を書く
uv run fdbja score-offline --config configs/default.yaml [--limit N]   # TOR + VAP-distance を採点し results.json を出力
uv run fdbja analyze-pause --eval-dir data/eval --out results/pause_gap   # Pause をギャップ長別に再集計(bootstrap CI 付き)
uv run fdbja sample-human-eval --config configs/default.yaml --out ./human_eval   # 人手評価バンドル(Web UI 込み)
uv run fdbja analyze-human-eval --responses <dir> --bundle ./human_eval --out <dir>   # 回収した回答の集計と自動指標との照合
```

各コマンドの引数は `uv run fdbja <command> --help` を参照。人手評価の設計と手順は
[docs/human_eval.md](docs/human_eval.md)、Pause の層別分析は
[docs/pause_gap_analysis.md](docs/pause_gap_analysis.md) にあります。

### ASR バックエンド

`transcribe` は `infer` が書いた `combined.wav` の ch1(モデル応答)を文字単位の時刻付き転写(jchat)にします。
config の `asr.backend` で選び、それぞれ専用の extra を入れます(依存する torch / numpy の範囲が違うため 1 環境に 1 つ)。

| バックエンド | 経路 | 文字の時刻 | 追加依存 |
|------------|------|-----------|---------|
| `reazonspeech_whisperx`(既定) | [ReazonSpeech-ESPnet v2](https://huggingface.co/reazon-research/reazonspeech-espnet-v2) の区間転写 → [WhisperX](https://github.com/m-bain/whisperX) で wav2vec2 CTC 整列 | 整列器が文字ごとに推定 | `.[asr-reazon-whisperx]`(espnet + whisperx、GPU で 20 秒クリップ約 2.5 秒、VRAM 約 3 GB) |
| `reazonspeech_nemo` | [ReazonSpeech NeMo v2](https://huggingface.co/reazon-research/reazonspeech-nemo-v2)(FastConformer-RNNT、日本語) | トークンの発火時刻。**サブワード内は文字数で均等分割**し、0.5 秒以内の間は次トークンまで橋渡し(近似。`asr/_subword.py`) | `.[asr-nemo]`(NeMo、GPU 推奨。20 秒クリップ約 0.5 秒、VRAM 約 5 GB) |
| `reazonspeech_k2` | [ReazonSpeech k2 v2](https://huggingface.co/reazon-research/reazonspeech-k2-v2)(Zipformer、sherpa-onnx) | 同上 | `.[asr-k2]`(ONNX、CPU で 20 秒クリップ約 1 秒。最も軽い) |

リーダーボードと論文の TOR は `reazonspeech_whisperx` の経路で計算しました。同じ音声に対して
ESPnet の転写文は完全に再現され、文字の開始時刻の差は中央値 0.04 秒でした(6 クリップ)。
`reazonspeech_nemo` / `reazonspeech_k2` は整列器を使わない分だけ速い一方、発話の切れ目の数が多く出るため
(同じ 6 クリップで発話数 24 / 22、WhisperX 経路は 21)、**TOR の絶対値はリーダーボードと直接比較できません**。
既存の結果と比べるときは既定のバックエンドを使い、報告時にはバックエンド名を添えてください。
任意の音声を単発で転写するには
`uv run fdbja transcribe --backend reazonspeech_nemo --wav response.wav --out ./text` が使えます。
自前の ASR を使う場合は `fdbja.asr.ASRModel` を実装して `fdbja.asr.register_backend` で登録します。
自前の ASR に差し替えた場合は、既定バックエンドの結果とは別の条件として報告してください。

### モデルアダプタ

| アダプタ | 対象 | 追加依存 |
|---------|------|---------|
| `moshi_ws` | J-Moshi / J-Moshi-ext / LLM-jp-Moshi-v1 など Moshi 系の WebSocket サーバ | `.[infer]` |
| `gemini_live` | Gemini Live API | `.[gemini]` |
| `openai_realtime` | OpenAI Realtime プロトコル(GPT-Realtime、Grok Voice) | `.[realtime]` |
| `silence` / `always_talk` | 自明なベースライン(常時沈黙 / 常時発話)。常時発話は初回に Open JTalk(声質 Mei、CC BY 3.0、帰属表示は [src/fdbja/assets/README.md](src/fdbja/assets/README.md))で 60 秒の音声ループを生成してキャッシュ | 常時発話のみ `pyopenjtalk`(無い場合は `source_wav` を指定) |

商用 API のアダプタは **2026-08 時点**のプロトコルとモデル名
(`gemini-live-2.5-flash-preview`、`gpt-realtime`、`grok-voice-latest`)で動作確認しています。
API 側の変更への追随は保証しません。動かなくなった場合は Issue で報告してください。

## 何を評価するか

対話コーパスから 3 種類のターンテイキング事象を自動検出し、
その場面でのモデルの振る舞いを人間 B の振る舞いと比べます。

| タスク | 事象 | モデルの望ましい挙動 |
|--------|------|----------------------|
| **Pause Handling** | ターン内の間(0.4–1.0 秒)で人間 B が沈黙 | 沈黙する(TO=0) |
| **Smooth Turn-Taking** | A(≥4 秒)→ 間(<0.4 秒)→ B(≥4 秒) | ターンを取る(TO=1) |
| **Backchanneling** | A(≥3 秒)中に人間 B が相槌のみ | 相槌に留める(TO=0)。あわせて相槌の頻度(`bc_freq`、A の発話 1 秒あたり。人間は `human_freq`)と、相槌の開始時刻を 0.2 秒ビンにした人間との JS ダイバージェンス(`jsd`、0〜1、相槌なしは 1.0)、相槌を 1 回以上出した事象の割合(`bc_any_rate`)とその事象に限った JSD(`jsd_cond`)を出す (実験的。既定の `tasks` に含めない) |

主指標は 1 つ、実験的指標が 1 つです。

- **TOR**(二値 Takeover Rate、主指標)— Full-Duplex-Bench と同じ定義。モデルがターンを取ったか否か。
  モデル応答の ASR 転写から判定します。リーダーボードの順位はこの指標の適合率で付けます。
- **VAP-distance**(連続、**実験的**)— 同じ話者 A の音声に対し、**人間**の応答と**モデル**の応答をそれぞれ凍結
  [VAP](https://github.com/inokoj/VAP-Realtime) に通し、次話者予測分布間の KL 距離を測ります。
  二値が捨てる「タイミングの良し悪し」を連続値で捉える狙いですが、人手判断との相関・既存の自動指標との比較・
  チェックポイントへの頑健性の検証がまだ終わっていません(別途、論文として検証中)。
  既定の config では無効で、`metrics.vap_distance.enabled: true` にした場合だけ参考値として併記されます。
  報告するときは「実験的指標」と添えてください。

比較用に、同じ 256 クラス VAP 定式化を使う **TurnNat 系の尤度スコア**(`metrics/turnnat.py`)も
実装しています。参照応答を必要としない絶対尺度で、位置づけは
[docs/related_metrics.md](docs/related_metrics.md) を参照。

いずれの指標にも自明な解があるため(沈黙し続ければ Pause / BC は「正解」、喋り続ければ Smooth TT は
「正解」)、`silence_baseline` と `always_talk_baseline` を全タスクで併記します。

Backchannel はモデルの能力に依存する任意タスクとして扱います。サーバ側のターン検出により
ユーザ発話中の出力が制約される API は、同時入出力可能なモデルと同一ランキングで比較しません。

### 既知の限界

二値 TOR は ASR 転写ベースの判定のため、ASR パイプラインの誤りを継承します。実データで確認した 3 パターン:

1. **整列ずれによる偽陰性** — WhisperX の文字整列が発話開始を遅らせ、イベント窓との重なりを取りこぼす
   (開始時刻ずれの中央値 0.14 秒、p90 は 0.94 秒)。
2. **正解音声の欠落** — Zoom の話者別トラックの無音ゲーティングで人間 B の短い相槌が録音から消え、
   転写にのみ残る(転写された窓内相槌の約 1/3 が音響的に無音)。VAP-distance の正解参照にも影響し得る。
3. **幽霊転写による偽陽性** — ほぼ無音のモデル出力に ASR が短語を長い区間で転写し、
   相槌免除(≤6 文字かつ ≤1.5 秒)を外れてターン奪取扱いになる。

VAP-distance は生波形から計算するため 1・3 の影響を受けません(2 は要注意)。
`sample-human-eval` / `analyze-human-eval` はこれらの妥当性を人手で検証するためのものです。

## 構成

```
src/fdbja/
  cli.py                    # fdbja コマンド(reconstruct | infer | transcribe | score-offline
                            #              | analyze-pause | sample-human-eval | analyze-human-eval)
  config.py  runner.py  report.py  io_audio.py
  data/event_detection.py   # pause / smooth_tt / bc 検出器(jchat 入力)
  data/reconstruct.py       # コーパス → 評価クリップ + manifest
  metrics/tor.py            # 二値 Takeover Rate
  metrics/vap_distance.py   # 連続 VAP-distance(バックエンド非依存)
  metrics/turnnat.py        # TurnNat 系 NLL スコア(比較用)
  metrics/discrimination.py # AUC・点双列相関
  metrics/agreement.py      # 人手評価の一致率・κ・ICC
  adapters/                 # ModelAdapter 実装(moshi_ws / gemini_live / openai_realtime / baselines)
  vap/                      # VAP バックエンド(VAP-Realtime をラップ)
  asr/                      # ASR バックエンド(reazonspeech_nemo / reazonspeech_k2 / reazonspeech_whisperx)
  human_eval.py  analyze_human_eval.py  pause_analysis.py
configs/default.yaml        # 評価設定(検出閾値・モデル一覧・レイアウト)
examples/dummy/             # 合成コーパスの生成・音量ベースの代用 ASR・動作確認用 config
webui/                      # 人手評価用の静的 Web UI(依存ライブラリなし)
docs/                       # リーダーボード(GitHub Pages)と設計メモ
tests/                      # pytest(コアのみ、外部依存なし。合成コーパスでの通し試験を含む)
```

config 駆動で、**データはコードと別レイヤ**として扱います。自分のモデルを評価するには
`ModelAdapter`(`adapters/base.py`)を実装するか、上記レイアウトに出力を置きます。

## データ

リポジトリには音声・コーパスを同梱しません(`docs/samples/` の試聴用抜粋 — 3 事象 × 刺激・人間 B・モデル 2 つ — を除く)。
評価クリップは LLM-jp-Zoom1 から `reconstruct` でローカル生成します。
入手方法、in-domain / out-of-domain の報告義務、held-out の方針は [DATASET.md](DATASET.md) を参照。

## ライセンス

- **コード**: Apache-2.0([LICENSE](LICENSE))。
- **バックエンドとデータは対象外**で、それぞれの条件に従ってください。

| 構成要素 | 条件 | 出典 |
|---------|------|------|
| VAP-Realtime(コード) | MIT | [inokoj/VAP-Realtime](https://github.com/inokoj/VAP-Realtime) |
| 日本語 VAP チェックポイント `vap_state_dict_jp_kyoto_20hz_2500msec.pt`(既定) | MIT。本リポジトリには含めず、Hugging Face からリビジョン `fe24ac6` を指定して取得 | [maai-kyoto/vap_jp_kyoto](https://huggingface.co/maai-kyoto/vap_jp_kyoto) |
| 日本語 VAP チェックポイント `vap_state_dict_jp_20hz_2500msec.pt`(2026-07-08 版のリーダーボードで使用。現行版は MIT チェックポイントで再採点済み) | **学術目的のみ**(academic purposes only)。同じコード経路で読み込める | VAP-Realtime README「Lisence」節 |
| CPC エンコーダ `60k_epoch4-d0f474de.pt` | MIT(facebookresearch/CPC_audio) | 同上 |
| ReazonSpeech(コード、`reazonspeech-{nemo,k2,espnet}-asr`) | Apache-2.0 | [reazon-research/ReazonSpeech](https://github.com/reazon-research/ReazonSpeech) |
| ReazonSpeech モデル(`reazonspeech-nemo-v2` / `reazonspeech-k2-v2` / `reazonspeech-espnet-v2`) | Apache-2.0(各モデルカード) | [Hugging Face: reazon-research](https://huggingface.co/reazon-research) |
| NVIDIA NeMo(`nemo_toolkit`、`.[asr-nemo]` の依存) | Apache-2.0 | [NVIDIA/NeMo](https://github.com/NVIDIA/NeMo) |
| sherpa-onnx(`.[asr-k2]` の依存) | Apache-2.0 | [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) |
| ESPnet(`.[asr-reazon-whisperx]` の依存) | Apache-2.0 | [espnet/espnet](https://github.com/espnet/espnet) |
| WhisperX(整列) | BSD-2-Clause | [m-bain/whisperX](https://github.com/m-bain/whisperX) |
| WhisperX が日本語整列に使う wav2vec2 モデル(`jonatasgrosman/wav2vec2-large-xlsr-53-japanese`) | Apache-2.0(モデルカード) | [Hugging Face](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-japanese) |
| Full-Duplex-Bench(設計の出典) | CC BY-NC 4.0。本リポジトリはコードを流用せず、事象定義と TOR の定義を独立に再実装 | [DanielLin94144/Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench) |
| LLM-jp-Zoom1(評価コーパス) | コーパスの利用条件に従う。`docs/samples/` の音声は同コーパスからの抜粋で、同じ条件が及ぶ | [DATASET.md](DATASET.md) |
| 評価対象モデルの出力(`docs/samples/`) | LLM-jp-Moshi-v1 と J-Moshi-ext(CC BY-NC 4.0 のモデル)の出力のみ収録。商用 API の出力は収録していない | 各モデルカード |

既定の構成(コード Apache-2.0、VAP-Realtime MIT、VAP チェックポイント MIT、CPC MIT、ReazonSpeech / ESPnet Apache-2.0、
WhisperX BSD-2-Clause)であれば、採点結果の利用に商用・非商用の制限はありません。
学術目的限定のチェックポイントを使った場合はその条件が結果にも及びます。

## 引用

本ツールキットを使った場合は、日本音響学会 2026 年秋季研究発表会の講演(2-7-12)を引用してください
([CITATION.cff](CITATION.cff) にも同じ書誌情報があります)。

```bibtex
@inproceedings{abe2026fdbja_asj,
  title     = {Full-Duplex-Bench-JA: 日本語フルデュプレックス音声対話モデル評価のためのベンチマーク構築},
  author    = {阿部 雄斗 and 佐伯 真於 and 高道 慎之介 and 藤江 真也 and 小川 哲司 and 東中 竜一郎},
  booktitle = {日本音響学会 2026 年秋季研究発表会 講演論文集},
  note      = {2-7-12},
  year      = {2026},
  month     = sep
}
```

## バージョン / 変更履歴

- **(準備中)** VAP-distance を実験的指標に位置づけ直し、既定で無効に。リーダーボードの順位を TOR の適合率に変更(2026-10-12)。
- **0.1.0(公開初版、準備中)** — `reconstruct` → `infer` → `transcribe` → `score-offline` の評価パイプライン、
  ASR バックエンド 3 種(ReazonSpeech NeMo / k2 / ESPnet+WhisperX)、
  TOR と VAP-distance、TurnNat 系スコア、自明なベースライン、Pause のギャップ長別分析、
  人手評価バンドル生成と集計、リーダーボード(6 モデル: J-Moshi / J-Moshi-ext / LLM-jp-Moshi-v1 /
  Gemini Live / GPT-Realtime / Grok Voice)。
  VAP チェックポイントの既定を MIT の `maai-kyoto/vap_jp_kyoto`(20 Hz / 2.5 s)に変更(ビット順の再検証済み)。
  リーダーボードの VAP-distance も同じチェックポイントで再採点(2026-09-28。TOR と事象数は 2026-07-08 版と同一)。
  試聴サンプルは 3 事象 × 2 モデルに絞り、D_KL の表示値は MIT チェックポイントで再計算。

バージョンは `pyproject.toml` の `version` が正です。

## トラブルシュート

| 症状 | 対処 |
|------|------|
| `score-offline` が何も採点しない(`scored 0 clips`) | `corpus.eval_dir` 配下に `<task>/<clip>.json` と `<task>_<model>_asr/text/<clip>.json` の両方があるか確認。片方が無いクリップは黙ってスキップされます。後者は `fdbja transcribe` が書きます |
| `ASR backend '...' is missing a dependency` | バックエンドの extra が入っていません。`uv sync --extra asr-reazon-whisperx`(NeMo は `asr-nemo`、k2 は `asr-k2`)。extra 同士は排他なので環境を分けてください |
| `.[asr-reazon-whisperx]` の import で `numpy.core.multiarray failed to import` | `ctc-segmentation` の wheel が numpy 1.x 向けにビルドされています。`uv pip install cython setuptools` の後、`uv pip install --no-build-isolation --no-cache --reinstall --no-binary ctc-segmentation ctc-segmentation` で手元の numpy に対して再ビルド |
| Blackwell 世代の GPU(sm_120)で ASR が CPU に落ちる / CUDA エラー | `uv pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128` を extra より先に入れる(下の torch の行も参照) |
| `metrics.vap_distance.vap_realtime_root or $VAP_REALTIME_ROOT must be set` | VAP-Realtime を clone し、`VAP_REALTIME_ROOT=/path/to/VAP-Realtime` を設定。`asset/cpc/` は clone に含まれる。`asset/vap/` には「VAP-distance の準備」の手順で `vap_state_dict_jp_kyoto_20hz_2500msec.pt` を置く |
| Hugging Face から `vap_state_dict_jp_kyoto_20hz_2500msec.pt` が見つからない(404) | 2026-10-01 に `main` から外されました。URL の `main` をリビジョン `fe24ac60d8fcc80463edde97ed90e3ceca5e5b88` に置き換える(「VAP-distance の準備」) |
| VAP チェックポイントを替えたら D_KL の値が変わった | 仕様どおりです。VAP は凍結した計測器で、チェックポイントが違えば値は比較できません。比較する全モデルを同じチェックポイントで採点し直し、`results.json` の `vap_checkpoint` を確認してください |
| torch が Blackwell GPU(sm_120)で動かない | `uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128` で cu128 以上を入れる |
| α(`exp(-α·D_KL)` のスケール)がモデル追加のたびに変わる | α はモデル集合から自動較正されます。モデルを追加したら全モデルを 1 回の `score-offline` で採点し直してください |
| `sample-human-eval` が `webui/ not found` と言う | Web UI はリポジトリ直下の `webui/` を参照します。`git clone` した作業ツリーから(`uv sync` の editable install で)実行してください |
| 商用 API アダプタが接続できない | モデル名・プロトコルは 2026-08 時点のもの。各 API のドキュメントで現行のモデル名を確認し、config の `model` を差し替えてください |

## 謝辞

[Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench)(Lin et al., 2025)の評価設計と、
[VAP-Realtime](https://github.com/inokoj/VAP-Realtime) / [MaAI](https://huggingface.co/maai-kyoto)(Inoue et al.、京都大学)の日本語 VAP モデルに基づいています。
