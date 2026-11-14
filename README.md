# Full-Duplex-Bench-JA

**日本語フルデュプレックス音声対話モデル**のターンテイキング評価ツールキット。
[Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench)(Lin et al., 2025、[arXiv:2503.04721](https://arxiv.org/abs/2503.04721))の
評価設計を日本語に持ち込んだもので、**人手注釈を必要としない**のが特徴です。
実対話コーパス(LLM-jp-Zoom1)における対話相手(人間 B)の振る舞いを正解として用い、
モデルの応答が「人間らしいタイミング」かどうかを採点します。

**範囲**: Full-Duplex-Bench v1 の 4 課題(Pause Handling / Backchanneling / Smooth Turn-Taking / User Interruption)のうち、相手(話者 A)の音声を固定して測れる 2 課題(Pause Handling、Smooth Turn-Taking)を日本語の実対話で実装しています。
Backchanneling(システムの相槌)は事象の定義だけを公開しており、採点していません。User Interruption(ユーザの割り込み)と、v1.5 が扱う重なりの場面(ユーザの相槌、第三者・背景の発話)には未対応です(下の「ロードマップ」)。

- リーダーボード: <https://llm-jp.github.io/full-duplex-bench-ja/>
  (`docs/` を GitHub Pages で配信。結果は `docs/results.json`、サンプル音声は `docs/samples/`)。
  順位は **TOR で望ましい振る舞いをした割合**(Full-Duplex-Bench と同じ二値指標)で付けています。
  ただし Pause のこの値は「黙るほど高い」定義で、聴取実験では人の判断と逆を向きました(リーダーボードの「人の判断との検証」)。
  **VAP-distance は実験的指標**です。人の判断との照合は聴取実験 1 回(2026-11)だけで、Pause では人の選好と合わなかったため参考値として併記し、
  既定の config では無効にしています(`metrics.vap_distance.enabled: true` で有効化)。
  掲載中の値(2026-11-04、9 システム)は、人間側の発話境界を音声で付け直す設定(`detection.boundaries: audio`)、
  ASJ の講演論文(下の「引用」。以下「論文」)と同じ書き起こし(ReazonSpeech-ESPnet で書き起こし、WhisperX で文字の時刻を整列)、既定の MIT チェックポイント(`maai-kyoto/vap_jp_kyoto`、下記)で採点したもので、
  **境界・書き起こし・チェックポイントが違う値は比較できません**(`docs/results.json` の `boundaries` / `asr` / `vap_checkpoint` に記録しています)。
  掲載は Pause Handling と Smooth Turn-Taking の 2 タスクです。bc(相槌)は転写の閾値に依存する実験的タスクとしてコードにだけ残し、既定では採点しません。
  人の判断との検証(2026-11、クラウドソーシング 150 人)の要点もリーダーボードに載せています。
- 英語版 README: [README.en.md](README.en.md)

**Quick start**(LLM-jp-Zoom1 が手元にある場合。合成データでの 1 分の動作確認は下の「[クイックスタート](#クイックスタート)」1.)

```bash
uv sync --extra asr-reazon-whisperx                        # 1. インストール(ASR バックエンドを 1 つ)
uv run fdbja reconstruct --config configs/default.yaml     # 2. 評価クリップを作る(config の corpus.zoom1_root を設定)
# 応答を用意: `fdbja infer` + `fdbja transcribe`、または combined.wav を置く(「[モデルアダプタ](#モデルアダプタ)」)
uv run fdbja score-offline --config configs/default.yaml   # 3. 採点 → results/summary.json
uv run fdbja leaderboard --reference docs/results.json     # 4. リーダーボードと同じ順位で掲載値と並べる
```

**目次**: [前提条件](#前提条件) / [クイックスタート](#クイックスタート) / [主要コマンド](#主要コマンド) /
[何を評価するか](#何を評価するか)([既知の限界](#既知の限界)) / [構成](#構成) / [データ](#データ) / [ライセンス](#ライセンス) /
[引用](#引用) / [ロードマップ](#ロードマップ) / [変更履歴](#バージョン--変更履歴) / [トラブルシュート](#トラブルシュート) / [謝辞](#謝辞)

## 前提条件

| 用途 | 必要なもの |
|------|-----------|
| 採点(二値 TOR) | Python 3.10 以上、[uv](https://docs.astral.sh/uv/) |
| 採点(VAP-distance) | 上に加えて `.[vap]`(torch)、[VAP-Realtime](https://github.com/inokoj/VAP-Realtime) のローカル clone(CPC エンコーダ同梱)、日本語 VAP チェックポイント [`maai-kyoto/vap_jp_kyoto`](https://huggingface.co/maai-kyoto/vap_jp_kyoto) の `vap_state_dict_jp_kyoto_20hz_2500msec.pt`(MIT。リビジョン固定で取得)を `<VAP_REALTIME_ROOT>/asset/vap/` に置く。手順は下の「VAP-distance の準備」 |
| モデル出力の ASR 転写(`transcribe`) | いずれか 1 つの extra: `.[asr-reazon-whisperx]`(既定。ReazonSpeech-ESPnet + WhisperX。論文の数値を出した経路)/ `.[asr-nemo]`(ReazonSpeech NeMo v2、整列器なしで高速)/ `.[asr-k2]`(ReazonSpeech k2 v2、CPU 可)。モデルの重みは初回実行時に Hugging Face Hub から取得 |
| 評価クリップの生成 | LLM-jp-Zoom1 のローカルコピー([DATASET.md](DATASET.md))、または自前の 2 チャネル対話の録音(下の「3. 自前の 2 チャネル対話から評価セットを作る」) |
| モデル推論(`infer`) | 評価対象に応じて Moshi 系 WebSocket サーバ、`GOOGLE_API_KEY`、`OPENAI_API_KEY`、`XAI_API_KEY`。自作モデルは `ModelAdapter` を実装する(「モデルアダプタ」) |
| 常時発話のベースライン(`always_talk`、既定の `source: auto`) | `.[always-talk]`(pyopenjtalk。初回に Open JTalk で 60 秒の音声ループを合成)。入れない場合は `source_wav` に音声ファイルを指定 |
| GPU | ロックファイルの torch は PyPI の Linux 版(CUDA 13.0 ビルド)で、NVIDIA ドライバ 580 以上が必要。古いドライバでは下の「GPU ドライバが古い場合」の手順で入れ替える |

### GPU ドライバが古い場合(torch の入れ替えと `--no-sync`)

`nvidia-smi` の右上の `CUDA Version` が 12.x(例: ドライバ 550 系 = CUDA 12.4)の環境では、ロックされた torch は GPU を使えません
(CPU に落ちる、または `CUDA driver version is insufficient` などのエラー)。extra を入れたあと、ドライバに合う CUDA の版の torch に入れ替えます。

```bash
nvidia-smi | head -4                                        # 右上の CUDA Version を確認
uv sync --extra asr-reazon-whisperx                         # まず通常どおり入れる
uv pip install --reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu126   # CUDA 12.x のドライバ向け
uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
uv run --no-sync fdbja transcribe --config configs/default.yaml
```

**入れ替えたあとは `uv run --no-sync`(または環境変数 `UV_NO_SYNC=1`)で実行してください。**
`uv run` と `uv sync` は既定で環境をロックファイルに合わせ直すため、`--no-sync` を付けないと torch が CUDA 13.0 版に戻ります。
index の URL の末尾(`cu126` など)は、[PyTorch の配布一覧](https://pytorch.org/get-started/locally/)でドライバが対応するものを選んでください。

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
(2) 各クリップの `combined.wav` を上の場所に置く(config では `adapter: offline`)か、`ModelAdapter` を実装して `infer` で生成する(下の「モデルアダプタ」)、
(3) `transcribe` で応答側チャンネル(ch1)を jchat に転写する、(4) `score-offline` を走らせる、の 4 段です。
config の `models` に `name` と `domain: in|out` を追加してください。

### 3. 自前の 2 チャネル対話から評価セットを作る

LLM-jp-Zoom1 の評価データは公開準備中です([DATASET.md](DATASET.md))。それまでの間、または別の領域で評価したい場合は、
手元の 2 人対話の録音から同じ手順で評価セットを作れます。必要なのは次の 2 つです。

- **ステレオの音声**(wav / flac、サンプリング周波数は任意): ch0 = 話者 A(モデルに聞かせる側)、ch1 = 話者 B(正解の振る舞いをする人間)。
  話者ごとに別に録ったもの(オンライン会議の話者別録音など)を 2 チャネルにまとめてください。1 チャネルの混合音声は使えません。
- **文字単位の時刻付き転写(jchat)**: 録音ごとに 1 つの JSON。1 文字 1 要素の配列で、各要素は次の 4 つのキーを持ちます。

```json
[
  {"speaker": "A", "word": "こ", "start": 0.52, "end": 0.61},
  {"speaker": "A", "word": "ん", "start": 0.61, "end": 0.70},
  {"speaker": "B", "word": "は", "start": 3.10, "end": 3.18}
]
```

`speaker` は `A`(ch0)か `B`(ch1)のどちらかで、**両方が必要**です(事象の検出は A と B の発話の並びを見ます)。
`word` は 1 文字、`start` / `end` は録音の先頭からの秒です。手元に転写が無ければ `transcribe` の `--wav` で作れます。
`--channel` と `--speaker` を並べると、2 つのチャンネルを別々に書き起こして 1 つの jchat にまとめます
(`--speaker` を省くと全文字が `B` になります。モデル応答を書き起こすための既定です)。

```bash
# my_dialogues/audio/<録音名>.wav(ステレオ、ch0 = A、ch1 = B)から my_dialogues/text/<録音名>.json を作る
uv run fdbja transcribe --wav my_dialogues/audio/*.wav --channel 0 1 --speaker A B --out my_dialogues/text
```

次に、config の `corpus` を `layout: generic` にして `reconstruct` を走らせます。`generic` では `text_dir` 配下の
すべての `*.json` を 1 録音とみなし、`audio_dir` 配下で**ファイル名(拡張子を除く)が同じ**音声と組にします(サブディレクトリは自由)。

```yaml
corpus:
  layout: generic              # zoom1(既定。<text_dir>/<セッション>/<録音>.json と <audio_dir>/<セッション>/<録音>.wav)| generic
  text_dir: ./my_dialogues/text
  audio_dir: ./my_dialogues/audio
  eval_dir: ./data/my_eval
```

`reconstruct --layout generic` でも指定できます。以降の `infer` → `transcribe` → `score-offline` は実データの場合と同じです。
合成データで一通り試すには次のとおりです(数値に意味はありません)。

```bash
uv run python examples/dummy/make_dummy_corpus.py --out data/my_dialogues --layout generic --audio-only   # ステレオ音声だけを作る
uv run python examples/dummy/energy_asr.py --backend energy --wav data/my_dialogues/audio/*.wav \
    --channel 0 1 --speaker A B --out data/my_dialogues/text                                     # 音量ベースの代用品で jchat を作る
uv run fdbja reconstruct --config examples/dummy/config_own_dialogues.yaml
uv run fdbja infer --config examples/dummy/config_own_dialogues.yaml
uv run python examples/dummy/energy_asr.py --config examples/dummy/config_own_dialogues.yaml
uv run fdbja score-offline --config examples/dummy/config_own_dialogues.yaml
```

注意点:

- 事象の検出は文字の時刻だけで行います(Pause = A のターン内の 0.4〜1.0 秒の間で B が黙っている、Smooth TT = A が 4 秒以上話し、0.4 秒未満の間をおいて B が 4 秒以上話す)。
  文字の時刻の精度がそのまま事象の境界の精度になるので、`detection.boundaries: audio`(下の節)で人間側の境界を音声に付け直すことを勧めます。
- クリップは 20 秒で、20 秒より短い録音からは作れません。
- 自前の評価セットの数値は LLM-jp-Zoom1 の評価セット(リーダーボード)とは比較できません。データの出典・事象数・設定を添えて報告してください。

### 4. 自分の 2 チャネル音声を測る(`fdbja measure`)

評価セットを使わずに、手元の対話の録音(ユーザとシステムが別々のチャネルに入った wav)からタイミングの統計を出します。
転写も注釈も要らず、基本依存(numpy / soundfile / pyyaml)だけで動きます(`uv sync` のみ、extra は不要)。

```bash
uv run fdbja measure rec.wav --condition model=my-model      # ch0 = ユーザ、ch1 = システム(逆なら --user-channel 1)
uv run fdbja measure recordings/ --json out.json             # ディレクトリ: 録音ごとの行と全体の集計
uv run fdbja make-profile human_dialogues/ --language ja --relationship strangers \
    --situation chat --medium "video call" -o profile.json   # 人どうしの録音から基準プロファイルを作る
uv run fdbja measure rec.wav --profile profile.json          # プロファイルに照らしたスコアを追加
```

```python
from fdbja import measure, load_profile, make_profile
m = measure("rec.wav", conditions={"model": "my-model"})    # mono 2 本なら measure("user.wav", "system.wav")
print(m.stats["turn_end"]["latency_median"], m.signature)    # numpy 配列なら measure(x, sample_rate=16000)
```

出力は次の記述統計です(音量ベースの検出器で、音声だけから事象を検出します)。

| 事象 | 出力 |
|---|---|
| ユーザのターンの終わり | 応答の遅延(符号つき。負 = ユーザが話し終える前に話し始めた)の中央値・四分位・件数、応答なしの割合 |
| ユーザのターン内の休止(0.4〜1.0 秒) | システムが声を出した割合。1.5 秒以下を「短い発声」、それより長いものを「長い発声」に分ける |
| ユーザの発話中にシステムが話し始めた箇所 | 回数(ユーザの発話 1 分あたり)、両者が同時に声を出していた時間の割合 |
| 全体 | 話者ごとの発話時間の割合 |

- **リーダーボードの数値とは比較できません**。事象を録音そのものから検出しており(評価セットの固定の事象ではない)、転写も使いません。
  取り違えないよう、キー名も `results.json` とは別にしてあります。
- 休止の中の「短い発声 / 長い発声」は**長さだけで判定**しています。相槌かどうか、ターンを取ったかどうかは見ていません。
- 数値には出力の `signature`(バージョン・検出の設定・プロファイル・チャネルの割り当てを 1 行にしたもの。例:
  `fdbja:0.2.0|measure|vad:energy(p56)|events:default|profile:none|ch:user=0,sys=1`)と、録音の条件を添えて報告してください。
  録音の条件は `--condition key=value` で出力に残せます(モデル、声、プロンプト、回線など。key は自由)。
  プロンプトの中身を出したくないときは `--condition-file prompt=prompt.txt` でハッシュだけを記録できます。
- 閾値は `--set turn_min_seconds=4`(Python では `config={"turn_min_seconds": 4}`)のように変えられ、変えた項目は `signature` に出ます。
  既定値はベンチマークのルール定数です(応答の窓 −0.5〜+2.0 秒、短い発声の上限 1.5 秒、0.1 秒未満の重なりは無視 など)。
- **基準プロファイル**(`--profile`)を渡すと、応答の遅延がプロファイルの許容範囲に入った割合、許容範囲からの外れ(秒)、
  休止に対する無反応と割り込みの割合(`metrics/pause_response.py` のルールを、ここで検出した休止に適用したもの)を加えます。
  同梱のプロファイルは `generic` だけです。これはルール定数だけから作ったもので、**人の対話のデータに基づく規範ではありません**
  (許容範囲 = 応答の窓なので、許容範囲に入った割合は応答ありの割合と同じ値になります)。
  タイミングの規範は言語・関係性・場面・媒体で変わるので、規範として使うときは対象に合った人どうしの録音から `make-profile` で作ってください。
  プロファイルには言語・関係性・場面・媒体・出典・録音数・事象数・検出の設定が記録されます。
- 音量ベースなので、雑音・笑い・息と発話を区別しません。チャネル間の音漏れがある録音では重なりが過大になります。

設計(どの変数をどこに記録するか)と限界は [docs/design/measure-entry.md](docs/design/measure-entry.md) にあります。

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
uv run fdbja score-offline --config configs/default.yaml [--limit N] [--model NAME]   # TOR(有効にした場合は VAP-distance も)を採点し results.json / summary.json / summary.csv を出力
uv run fdbja leaderboard --results results/summary.json [--reference docs/results.json]   # リーダーボードと同じ順位(望ましい振る舞いの割合)で表示。採点条件が違えば警告
uv run fdbja analyze-profile --results results/results.json --out results/profile   # システムごとの傾向(黙る / 被せる / 遅い)と事象単位のレポート
uv run fdbja analyze-pause --eval-dir data/eval --out results/pause_gap   # Pause をギャップ長別に再集計(bootstrap CI 付き)
uv run fdbja sample-human-eval --config configs/default.yaml --out ./human_eval   # 人手評価バンドル(Web UI 込み)
uv run fdbja analyze-human-eval --responses <dir> --bundle ./human_eval --out <dir>   # 回収した回答の集計と自動指標との照合
uv run fdbja measure rec.wav [--profile generic] [--condition key=value]   # 任意の 2 チャネル録音のタイミング統計(転写なし、リーダーボードとは比較不可)
uv run fdbja make-profile <dir> --language ja --relationship R --situation S --medium M -o profile.json   # 人どうしの録音から measure 用の基準プロファイル
```

各コマンドの引数は `uv run fdbja <command> --help` を参照。人手評価の設計と手順は
[docs/human_eval.md](docs/human_eval.md)、Pause の層別分析は
[docs/pause_gap_analysis.md](docs/pause_gap_analysis.md) にあります。

### 採点結果の出力(`score-offline` / `analyze-profile`)

`score-offline` は `output.dir` に 3 つのファイルを書きます。

| ファイル | 中身 |
|---------|------|
| `results.json` | `summary`(タスク × システム)、採点条件、`events`(事象 × システムの判定。事象の時刻 `t_start` / `t_end`、判定窓の近くのシステムの発話 `sys_segments` = `[開始, 終了, 文字数]` を含む) |
| `summary.json` | `results.json` から `events` を除いたもの。`docs/results.json` と同じ形 |
| `summary.csv` | `summary` を 1 行 1 システム × タスクに展開したもの(区間は `*_low` / `*_high`、採点条件を各行に付記) |

`summary` の主な列(キー名は `docs/results.json` と同じ):

| キー | 内容 |
|-----|------|
| `n` / `n_clips` | 採点した事象の数 / クリップの数 |
| `TOR` / `TOR_ci` | 二値 TOR と 95% 区間(事象を単位とする bootstrap、10000 回、百分位。`output.bootstrap: {n, seed}` で変更可。リーダーボードに掲載した区間は 2,000 回で計算したもの) |
| `lat_median` / `lat_q1` / `lat_q3` / `lat_n` | Smooth TT でターンを取った事象の応答の遅れ(A の発話終了から応答の開始まで、秒)。リーダーボードと同じく負の値は 0 に丸めたもの |
| `onset_median` / `onset_q1` / `onset_q3` / `overlap_rate` | 同じ遅れを丸めずに保持したもの(-0.5〜+2.0 秒)と、そのうち A が話し終える前に話し始めた(被せた)割合 |
| `event_mean_dkl` / `dkl_ci` | VAP-distance(有効にした場合のみ)とその 95% 区間 |

採点条件はトップレベルの `boundaries`(`detection.boundaries`)、`system_activity`、`smooth_tt_roles`、`asr`
(`transcribe` がモデルごとに記録したバックエンド名。転写を手で置いた場合は不明と書く)、`vap_checkpoint`(ファイル名と SHA-256 の先頭 12 桁)、`ci` に入ります。
`events` の各行には丸めない遅れ `onset` も入ります。

`analyze-profile` は `results.json` から、システムごとの傾向を 1 枚の表(`profile.md` / `profile.csv`)と散布図(`profile.png` / `profile.svg`。`.[analysis]` の matplotlib が必要)に、
事象ごとの判定とその理由を `events.csv` にまとめます。

| 列 | 内容 |
|----|------|
| Pause: 話し続けた / 相槌だけ / 黙った | ポーズに 0.1 秒以上重なる発話が相槌の範囲(6 文字以下かつ 1.5 秒以下)を超えた(= TOR)/ 相槌の範囲に収まった / 重なる発話が無い |
| Smooth TT: ターンを取った / うち被せた / 無反応 | A の発話終了の -0.5〜+2.0 秒に 3 文字以上の発話を始めた(= TOR)/ そのうち A の終了より前に始めた / どちらも無い |
| 遅れの四分位 | 丸めない応答の遅れ(秒) |
| 傾向(目安) | 被せる型 = Pause で話し続けた割合が 0.5 以上、黙る型 = Smooth TT の無反応が 0.5 以上、遅い型 = 遅れの中央値が 1.0 秒以上(閾値は引数で変更可) |

散布図は横軸が Pause で話し続けた割合、縦軸が Smooth TT で無反応だった割合で、左下が望ましい振る舞い、左上が黙る型、右下が被せる型です。
`events.csv` には事象の時刻、判定窓、システムの発話区間、判定(`takeover` / `short` / `silent` / `taken` / `overlap` / `no_response`)とその理由が並びます。

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

### TOR の判定に使う発話区間(`detection.system_activity`)

二値の TOR は「モデルがいつ話しているか」(Pause の無音に掛かったか、Smooth TT で相手の発話終了直後に話し始めたか)と
「何を話したか」(文字数で相槌と発話権の取得を区別する)の両方を見ます。

| 設定 | いつ話しているか | 何を話したか |
|------|----------------|-------------|
| `transcript`(既定) | ASR の文字の開始・終了時刻を 0.3 秒の間で発話にまとめたもの | ASR の文字数 |
| `audio` | モデル応答チャンネル(ch1)の音声区間。`detection.boundaries: audio` と同じ音量ベースの検出器(20 ms フレーム、ITU-T P.56 型の閾値)で取り、0.3 秒以下の無音はつなぐ | ASR の発話の文字数を、その発話と重なる音声区間に重なりの長さで按分 |

`transcript` では時刻が ASR の実装に依存します。整列器が区間末の文字を後ろの無音まで伸ばすと
(例: 「ん」が 2.08〜5.30 秒)、モデルが Pause の無音に話し掛けたと判定されます。
評価セットの 240 事象(Pause)で、同じ音声を 2 種類の ASR で書き起こした場合に判定が食い違った事象は
`transcript` で 17〜73 件、`audio` で 1〜18 件でした(4 システム)。2 種類の ASR の TOR の差は `audio` で最大 0.03 です(Smooth TT を話者交替の事象に絞った 72 事象では最大 0.07)。

```yaml
detection:
  system_activity: audio      # transcript(既定)| audio
  system_vad: {}              # 任意。merge_gap(0.3)/ max_offset(0.5)/ min_chars(0.5)/ 検出器の設定(margin_db など)
```

- `audio` は `combined.wav` を読みます(`transcript` は ASR の jchat だけ)。ASR は文字数を得るためだけに使います。
  VAP-distance は ASR を使わないので、どちらの設定でも変わりません。
- **既存の数値(リーダーボード・論文)は `system_activity: transcript` と論文時の書き起こしの組み合わせでのみ再現します**
  (論文は `boundaries: transcript`、2026-11-04 のリーダーボードは `boundaries: audio`。下の節)。
  既定の `reazonspeech_whisperx` で書き起こし直すと、文字の終端の付け方が違うため Pause の TOR が上がります
  (J-Moshi-ext 0.242 → 0.425)。`audio` の値は ASR に依存しない代わりに既存の数値とも一致しない
  (同じ J-Moshi-ext で Pause 0.175 / Smooth TT 0.558)ので、別の条件として報告してください。
- 0.3 秒のつなぎ幅を大きくすると、話し続けている途中の間で発話が切れにくくなり、Smooth TT の TOR が下がり
  Pause の TOR が上がります。0.3 秒は `transcript` の文字のまとめ方と同じ値で、人間の話し手(評価クリップの ch1)を
  同じ手順で採点したときに `transcript` の判定に最も近くなります。

### 事象の境界(`detection.boundaries`)と話者交替への限定(`detection.smooth_tt_roles`)

| 設定 | 既定 | 内容 |
|------|------|------|
| `boundaries: transcript` | ✓ | 人間側(A・B)の発話境界を書き起こしの文字の時刻で置く(論文の定義) |
| `boundaries: audio` | | 同じ事象のまま、人間側の境界を正解クリップの音声区間に付け直す(`data/event_boundaries.py`)。LLM-jp-Zoom1 では文字の時刻が音声より 0.15〜0.3 秒ほど早いため。2026-11-04 のリーダーボードはこの設定 |
| `smooth_tt_roles.enabled: true` | 無効 | Smooth TT を A から B へ本当にターンが渡る事象に限る((a) A の発話が相槌主体でない (b) A の終わりの直前 2 秒に B が 0.5 秒を超えて話していない (c) B の発話が相槌の連なりでない)。評価セットでは 197 事象 → 72 事象 |

```yaml
detection:
  boundaries: audio           # transcript(既定)| audio
  smooth_tt_roles:
    enabled: true             # 既定は false
```

`reconstruct` と `score-offline` の両方が同じ設定を読みます。既定を変えると事象の時刻(と `smooth_tt_roles` では事象の数)が変わるので、
結果には設定を添えて報告してください。

### モデルアダプタ

| アダプタ | 対象 | 追加依存 |
|---------|------|---------|
| `moshi_ws` | J-Moshi / J-Moshi-ext / LLM-jp-Moshi-v1 など Moshi 系の WebSocket サーバ | `.[infer]` |
| `gemini_live` | Gemini Live API | `.[gemini]` |
| `openai_realtime` | OpenAI Realtime プロトコル(GPT-Realtime、Grok Voice) | `.[realtime]` |
| `silence` / `always_talk` | 自明なベースライン(常時沈黙 / 常時発話)。常時発話は初回に Open JTalk(声質 Mei、CC BY 3.0、帰属表示は [src/fdbja/assets/README.md](src/fdbja/assets/README.md))で 60 秒の音声ループを生成してキャッシュ | 常時発話のみ `pyopenjtalk`(無い場合は `source_wav` を指定) |

#### 自作モデルをつなぐ

config の `adapter` には、上の名前のほかに次の 2 つを書けます。

| 書き方 | 動作 |
|-------|------|
| `adapter: "パッケージ.モジュール:クラス名"` | 自作の `ModelAdapter`(`adapters/base.py`)を import して使う。`name` / `domain` 以外のキーはコンストラクタの引数になる。モジュールは import できる場所に置く(インストールするか `PYTHONPATH` を通す) |
| `adapter: offline` | 何も実行しない。各クリップの `combined.wav`(ch0 = 話者 A、ch1 = システム)を自分で置く。`infer` は置かれている数を表示するだけ |

`ModelAdapter` は `infer(input_audio, sample_rate) -> DialogueResponse` を 1 つ実装します。
`input_audio` は話者 A のモノラル音声(16 kHz)、返り値の `response_audio` はシステムの応答音声で、`infer` が `combined.wav` に書きます。
コードから登録する場合は `fdbja.adapters.register_adapter("名前", クラス)` の後に `fdbja.cli.main([...])` を呼びます。
`examples/adapters/reply_after_silence.py` は「相手が 0.6 秒黙ったら固定の音声を返す」最小の例です。

```bash
# 合成データのクイックスタート(上の 1.)の後で
PYTHONPATH=examples/adapters uv run fdbja infer --config examples/adapters/config.yaml   # adapter: "reply_after_silence:ReplyAfterSilence"
uv run python examples/dummy/energy_asr.py --config examples/adapters/config.yaml
uv run fdbja score-offline --config examples/adapters/config.yaml
```

アダプタを作れない項目(未知の名前、依存が入っていない、引数が違う)は、`infer` がその項目だけを飛ばして理由を表示し、残りのシステムは続けます(終了コードは 1)。
`score-offline` はアダプタを作らずファイルだけを読むので、推論用と採点用で config を分ける必要はありません。出力がそろっているシステムだけを採点し、
`--model` で対象を絞れます(`infer` / `transcribe` も同じ)。

リーダーボードの商用 API 3 システムの出力は、次のモデル ID と時期で取得しました。

| システム | アダプタ | モデル ID | 声 | 取得時期 |
|----------|----------|-----------|----|----------|
| Gemini Live | `gemini_live` | `gemini-3.1-flash-live-preview` | Aoede | 2026-04 |
| GPT-Realtime | `openai_realtime` | `gpt-realtime` | marin | 2026-07 |
| Grok Voice | `openai_realtime` | `grok-voice-latest` | eve | 2026-07 |

3 システムとも、発話区間の検出はサーバ側の既定値のままです。Gemini Live の出力は、`gemini_live` アダプタの移植元のスクリプト(設定は同じ)で取得しました。
`gpt-realtime` と `grok-voice-latest` は別名で、指す先のモデルは提供元の更新で替わります
(Grok Voice は取得後に新しい世代が公開されています)。いま同じ ID で推論しても、リーダーボードと同じモデルになるとは限りません。
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
  モデル応答の ASR 転写から判定します。リーダーボードの順位はこの指標で望ましい振る舞いをした割合で付けます。
- **VAP-distance**(連続、**実験的**)— 同じ話者 A の音声に対し、**人間**の応答と**モデル**の応答をそれぞれ凍結
  [VAP](https://github.com/inokoj/VAP-Realtime) に通し、次話者予測分布間の KL 距離を測ります。
  二値が捨てる「タイミングの良し悪し」を連続値で捉える狙いですが、人の判断との照合は聴取実験 1 回(2026-11)だけで、
  Pause では人の選好と合いませんでした(リーダーボードの「人の判断との検証」)。別のデータでの確認と、チェックポイントへの頑健性の検証は済んでいません。
  既定の config では無効で、`metrics.vap_distance.enabled: true` にした場合だけ参考値として併記されます。
  報告するときは「実験的指標」と添えてください。

比較用に、同じ 256 クラス VAP 定式化を使う **TurnNat 系の尤度スコア**(`metrics/turnnat.py`)も
実装しています。参照応答を必要としない絶対尺度で、位置づけは
[docs/related_metrics.md](docs/related_metrics.md) を参照。

いずれの指標にも自明な解があるため(沈黙し続ければ Pause / BC は「正解」、喋り続ければ Smooth TT は
「正解」)、`silence_baseline` と `always_talk_baseline` を全タスクで併記します。

Backchannel はモデルの能力に依存する任意タスクとして扱います。サーバ側のターン検出により
ユーザ発話中の出力が制約される API は、同時入出力可能なモデルと同一ランキングで比較しません。

**指標の位置づけ**: ターンテイキングの指標は「人が自然と感じる範囲(許容範囲)の外に出ているか」を測るもので、
許容範囲の中の個性(相槌の多寡、間の長短)は採点しません。Smooth TT では「録音の人間 B に近い」が許容範囲の中心として成り立ちましたが、
Pause では成り立たず(人の判断との検証で、間に短く声を出したシステムが選ばれた)、検証中の Pause 指標は
「間で短く反応したか(相槌の有無)と、相手に被せて続けなかったか」を測ります。
相手の発話中の相槌(Backchanneling)は別の課題として定義のみ公開しています(未採点)。

### 提案指標(既定では無効)

人の判断との検証([リーダーボードの「人の判断との検証」](https://llm-jp.github.io/full-duplex-bench-ja/#validation-sec)、聴取実験 2026-11)で、
TOR・元の VAP-distance より人の選好と合った 2 つの指標を実装しています。どちらも既定の採点(TOR、元の VAP-distance)は変えず、
有効にしたときだけ `score-offline` の要約(`summary.json` / `summary.csv`)と事象単位の結果に列が加わります。
既定で無効にしているのは、検証が 1 コーパス(LLM-jp-Zoom1)・9 システムの聴取実験 1 回だけで、別のデータでの確認が済んでいないためです。

- **境界重み付き VAP-distance**(Smooth TT 用、`metrics.vap_distance.weighting: boundary`、`vap_distance.enabled: true` が前提)—
  元の VAP-distance が判定窓の D_KL(P_t‖Q_t) を等重みで平均するのに対し、境界 t0(A の話し終わり)からの重み
  w_t = exp(−|t − t0| / τ)(`tau_seconds`、既定 2.0 秒)で加重平均します。平均する範囲は判定窓の前 5 秒・後 1.5 秒まで
  (聴取者が聞いた範囲。`boundary_pre_seconds` / `boundary_post_seconds`)。要約の列は `vapd_boundary`(事象平均、小さいほど人間 B に近い)と
  `vapd_boundary_ci_*`。Pause にも `boundary_tasks: [pause, smooth_tt]` で適用できます(t0 = 間の始まり)が、検証で選ばれたのは Smooth TT だけです。
  検証時のチェックポイントは VAP-Realtime 配布版で、既定の MIT 版での値は検証していません。
- **割り込みと無反応を罰する Pause 指標**(Pause 用、`metrics.pause_response.enabled: true`)— 間 [g_s, g_e] ごとに、
  無反応 = 間の中の部分が 0.2 秒以上の有声区間が無ければ 1、割り込み = 間の中で始まり A の再開後 0.5 秒以上続く有声区間がある、
  または間の前 2 秒で A と 0.5 秒を超えて重なっていれば 1 とし、指標 = 無反応 + 割り込み(0〜2、小さいほど良い)。
  システムの値は事象平均です。有声区間は応答音声(`combined.wav` の ch1)と A(GT の ch0)のエネルギー VAD
  (50 ms 枠、各チャネルの 99.5 パーセンタイルの 5 %)から取り、ASR も VAP も使いません。要約の列は `pause_response`
  (`pause_response_ci_*`)、`pause_no_response_rate`、`pause_interrupt_rate`。閾値 4 つは config で変えられます。
  録音の人間 B はこの評価セットの間でほとんど声を出していない(聴取に使った 100 事象のうち 12)一方、聴取者は間で短く反応したシステムを選んだため、
  人間 B との距離ではなく「反応したか」と「被せて続けなかったか」を直接数えます。相槌・息・雑音は区別しません。

### 既知の限界

二値 TOR は ASR 転写ベースの判定のため、ASR パイプラインの誤りを継承します。実データで確認した 3 パターン:

1. **整列ずれによる偽陰性** — WhisperX の文字整列が発話開始を遅らせ、イベント窓との重なりを取りこぼす
   (開始時刻ずれの中央値 0.14 秒、p90 は 0.94 秒)。
2. **正解音声の欠落** — Zoom の話者別トラックの無音ゲーティングで人間 B の短い相槌が録音から消え、
   転写にのみ残る(転写された窓内相槌の約 1/3 が音響的に無音)。VAP-distance の正解参照にも影響し得る。
3. **ASR の誤認識（無音区間での転写）による偽陽性** — ほぼ無音のモデル出力に ASR が短語を長い区間で転写し、
   相槌免除(≤6 文字かつ ≤1.5 秒)を外れてターン奪取扱いになる。

VAP-distance は生波形から計算するため 1・3 の影響を受けません(2 は要注意)。
`sample-human-eval` / `analyze-human-eval` はこれらの妥当性を人手で検証するためのものです。

アダプタ間の公平性について、未修正の点が 2 つあります(リーダーボードの商用 API 3 本の値に効きうる。修正は採点し直しと合わせて行う予定です)。

4. **商用 API アダプタは割り込みの通知を無視する** — `gemini_live` / `openai_realtime` は、サーバが送る割り込みの通知
   (Gemini の `server_content.interrupted`、OpenAI Realtime の `input_audio_buffer.speech_started` / `response.cancelled`)を処理せず、
   受信した音声を全部時間軸に残します。実際のクライアントは相手が話し始めたら再生を止めるので、API 側で途中で止まるはずの応答が
   続けて話したものとして採点されます(例: Pause の間で短く話し始めた応答が 1.5 秒を超える発話になり TO = 1 になりうる)。
5. **応答の時間軸の作り方がアダプタで異なる** — Moshi 系(`moshi_ws`)は受信したフレームを受信順に連結し、商用 API は壁時計の到着時刻で配置します。
   応答の遅れ(`lat_median`)は、アダプタをまたぐと厳密には同じ尺度になっていません。

## 構成

```
src/fdbja/
  cli.py                    # fdbja コマンド(reconstruct | infer | transcribe | score-offline
                            #              | leaderboard | analyze-pause | analyze-profile
                            #              | sample-human-eval | analyze-human-eval)
                            #              | measure | make-profile)
  config.py  runner.py  report.py  io_audio.py
  summary.py                # 95% 区間・応答の遅れ・採点条件(summary.json / summary.csv)
  turn_profile.py           # analyze-profile(システムの傾向と事象単位のレポート)
  measurement.py            # measure / make-profile(任意の 2 チャネル録音のタイミング統計、基準プロファイル)
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
examples/dummy/             # 合成コーパスの生成・音量ベースの代用 ASR・動作確認用 config(自前の対話の手順の config も)
examples/adapters/          # 自作の ModelAdapter の最小例と config
webui/                      # 人手評価用の静的 Web UI(依存ライブラリなし)
docs/                       # リーダーボード(GitHub Pages)と設計メモ
tests/                      # pytest(コアのみ、外部依存なし。合成コーパスでの通し試験を含む)
```

config 駆動で、**データはコードと別レイヤ**として扱います。自分のモデルを評価するには
`ModelAdapter`(`adapters/base.py`)を実装するか、上記レイアウトに出力を置きます。

## データ

リポジトリには音声・コーパスを同梱しません(`docs/samples/` の試聴用抜粋 — 3 事象 × 刺激・人間 B・モデル 2 つ — を除く)。
評価クリップは LLM-jp-Zoom1 から `reconstruct` でローカル生成します。
LLM-jp-Zoom1 は公開準備中です。コーパスが手元に無い場合は、合成データでの動作確認(クイックスタート 1.)と、
自前の 2 チャネル対話からの評価セット作成(同 3.)が使えます。
入手方法、in-domain / out-of-domain の報告義務、held-out の方針は [DATASET.md](DATASET.md) を参照。

## ライセンス

- **コード**: Apache-2.0([LICENSE](LICENSE))。
- **バックエンドとデータは対象外**で、それぞれの条件に従ってください。

| 構成要素 | 条件 | 出典 |
|---------|------|------|
| VAP-Realtime(コード) | MIT | [inokoj/VAP-Realtime](https://github.com/inokoj/VAP-Realtime) |
| 日本語 VAP チェックポイント `vap_state_dict_jp_kyoto_20hz_2500msec.pt`(既定) | MIT。本リポジトリには含めず、Hugging Face からリビジョン `fe24ac6` を指定して取得 | [maai-kyoto/vap_jp_kyoto](https://huggingface.co/maai-kyoto/vap_jp_kyoto) |
| 日本語 VAP チェックポイント `vap_state_dict_jp_20hz_2500msec.pt`(2026-07 時点のリーダーボードで使用。現在の掲載値は MIT チェックポイントで採点し直したもの) | **学術目的のみ**(academic purposes only)。同じコード経路で読み込める | VAP-Realtime README「Lisence」節(綴りは原文のまま) |
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
| 評価対象モデルの出力(`docs/samples/`) | LLM-jp-Moshi-v1(Apache-2.0)と J-Moshi-ext(CC BY-NC 4.0)の出力のみ収録。商用 API の出力は収録していない | 各モデルカード |

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

## ロードマップ

1. **次のバージョン(0.2.0)** — 人の判断で検証した 2 指標(境界重み付き VAP-distance、割り込みと無反応を罰する Pause 指標)は main に実装済みです(既定では無効)。下の「未リリース」の整備と合わせて 0.2.0 としてリリースします。
2. 評価データの公開(LLM-jp-Zoom1 の公開に連動)。
3. システムの相槌課題(Backchanneling)を採点対象に(聴取で確認したあと)。
4. システムが先に話している場面の 2 課題(v1 の User Interruption と、v1.5 のユーザの相槌)。設計案は [docs/design/next-tasks.md](docs/design/next-tasks.md)。

## バージョン / 変更履歴

- **未リリース(main)** — 外部の利用者が自前のデータ・モデルで評価するための整備。採点の定義と既定値は変えていない(既存の数値は同じ手順で再現する)。
  - 自前の 2 チャネル対話から評価セットを作る経路: `transcribe --wav` に `--channel 0 1 --speaker A B`(2 チャンネルを話者ラベル付きで 1 つの jchat に)、
    `reconstruct` に任意のディレクトリ構成を読む `corpus.layout: generic`(`--layout generic`)、話者ラベルが A / B でない転写の警告。合成データの `--layout generic --audio-only`。
  - 自作モデルをつなぐ口: `fdbja.adapters.register_adapter`、config の `adapter: "モジュール:クラス"` と `adapter: offline`、例 `examples/adapters/`。
    アダプタを作れない項目は `infer` が飛ばして続ける。`infer` / `score-offline` に `--model`。
  - `score-offline` の要約に `n_clips`、`TOR_ci` / `dkl_ci`(事象の bootstrap)、応答の遅れの分位点と丸めない遅れ(`onset_*`、被せの割合 `overlap_rate`)、
    採点条件(`boundaries` / `system_activity` / `asr` / `vap_checkpoint` / `ci`)を追加し、`summary.json` / `summary.csv` も出力。キー名は `docs/results.json` に合わせた。
  - `analyze-profile`: システムごとの傾向(黙る型 / 被せる型 / 遅い型)の表と散布図、事象単位の判定と理由の CSV。
  - 人の判断で検証した 2 指標を追加(既定では無効、README「提案指標」): 境界重み付き VAP-distance
    (`metrics.vap_distance.weighting: boundary`、`tau_seconds`、要約の `vapd_boundary`)と、割り込みと無反応を罰する Pause 指標
    (`metrics.pause_response`、要約の `pause_response` / `pause_no_response_rate` / `pause_interrupt_rate`)。既定の採点は変わらない。
  - `leaderboard` コマンド: `summary.json` をリーダーボードと同じ順位で表示し、`--reference docs/results.json` で掲載値と並べる(採点条件が違えば警告)。
  - CI に Python 3.13 と extras(vap / analysis / always-talk)の import・解決の確認を追加。開発依存に pytest-cov。
  - extra `always-talk`(pyopenjtalk)を追加。古い GPU ドライバでの torch の入れ替えと `uv run --no-sync` の手順を追記。
  - 商用 API の記載を、掲載値を取得したときの条件に合わせた(掲載値は変わらない)。`configs/default.yaml` の Gemini のモデル ID を、リーダーボードの出力に使った
    `gemini-3.1-flash-live-preview` に(従来の `gemini-live-2.5-flash-preview` は掲載値の取得に使っていない)。リーダーボードの表示名「Grok Voice 3.0」を
    「Grok Voice」に(取得は 2026-07 の `grok-voice-latest` で、3.0 の公開より前)。README「モデルアダプタ」とリーダーボードの注記に、取得時期とモデル ID を記載。
  - リーダーボードのサンプル欄に境界の切り替え(音声の境界 / 文字の境界)を追加し、Smooth TT のサンプル事象を差し替え(掲載値は変わらない)。
  - `measure` / `make-profile` コマンドと `from fdbja import measure, load_profile, make_profile`: 任意の 2 チャネル録音から、
    転写なしでタイミングの統計(応答の遅延、休止の中の発声、重なり、発話時間の割合)を出す入口。基本依存だけで動く。
    差し替え可能な基準プロファイル(同梱は `generic` のみ)と、数値に添える `signature`。リーダーボードの数値とは比較できない(README「自分の 2 チャネル音声を測る」)。
  - `leaderboard --reference`: 参照側の採点条件に説明文が付いていても値の一致を判定するように修正(`docs/results.json` と並べると境界が同じでも警告が出ていた)。
  - 文書の整理: 「範囲」の記述を Full-Duplex-Bench v1 / v1.5 の課題名に合わせ、VAP-distance の検証状況、サンプル音声のモデルのライセンス、`docs/` 配下の古い記述を更新。どこからも読まれていなかった config の `output.exporters` を削除。
  - **採点に影響しうる不具合の修正(0.2.0 向け)**。いずれも**既定は従来どおり**で、掲載値(`docs/`)は変わりません。
    設定で切り替えた場合は、**掲載値と比べる前に全システムを採点し直して確認が必要**です。
    - 相槌語の長音符: 正規化で長音符「ー」を先に消すため、相槌語リストの「えー」「ふーん」「あー」「へー」などが一致しなかった
      (`smooth_tt_roles` の判定に効く)。`detection.smooth_tt_roles.match_long_vowel: true` で長音符込みで照合する(数えるときだけ長音符を除く)。
    - 重なり 0.1 秒の端: `1.7 - 1.6 = 0.0999…` のように、ちょうど 0.1 秒の重なりが浮動小数で欠けていた(Pause の事象検出と TOR)。
      `detection.pause.overlap_tolerance: 1.0e-6` で許容誤差つきで比べる。
    - α の較正: 望ましい事象の D_KL の中央値が 0 のとき `alpha = inf` になっていた → 較正できないときは `alpha` / `score` を出さない
      (警告を出す。0 以下の `alpha` の指定はエラー)。自明なベースライン(`silence` / `always_talk`)を較正から除く
      `metrics.vap_distance.calibration_exclude_baselines: true` を追加(既定 false = 従来どおり全システムで較正)。
    - VAP キャッシュ: ファイルがあれば読むだけで、`infer` をやり直しても古い確率が使われていた → 各 npz に入力音声の SHA-256・
      チェックポイントの SHA-256(と任意の `vap_revision`)・フレーム設定から作る key を記録し、一致しなければ計算し直す
      (`metrics.vap_distance.cache: validate`、既定)。モデル側のキャッシュはタスクごとの階層(`<model>/<task>/`)に。
      0.1.x で書いたキャッシュ(key なし)は初回に計算し直すため、VAP の推論が一度走ります。従来の動作は `cache: trust`。
    - bootstrap の単位: `score-offline` は事象単位、`analyze-pause` はクリップ単位と 2 通りあった → `output.bootstrap.unit: event | clip`
      で選べるように(既定 `event` = 掲載の区間)。`clip` は同じクリップの事象をまとめて引き直す(事象間の相関を考慮)。
- **0.1.0(公開初版、2026-11-05)**
  - 事象の境界を人間側の音声で付け直す設定(`detection.boundaries: audio`)、Smooth TT を話者交替の事象に限る設定
    (`detection.smooth_tt_roles`)、TOR の発話区間をモデル応答の音声から取る設定(`detection.system_activity: audio`)、
    公開した評価セットをクリップ一覧から作り直す `reconstruct --clips` を追加。既定値はいずれも従来どおり。
  - 常時発話のベースラインを Open JTalk の音声ループに、無音チャンネルで ASR を飛ばす(無音区間での誤認識の防止)、TurnNat のビット順の修正。
  - リーダーボードを音声の境界・9 システム(LLM-jp-Live と自明なベースライン 2 本を追加、ベースラインは順位の対象外)で
    採点し直し、95% 区間と「人の判断との検証(2026-11)」を掲載(2026-11-04)。
  - VAP-distance を実験的指標に位置づけ直し、既定で無効に。リーダーボードの順位を TOR(望ましい振る舞いの割合)に変更(2026-10-12)。
- **0.0.1(2026-09、非公開版)** — `reconstruct` → `infer` → `transcribe` → `score-offline` の評価パイプライン、
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
| Blackwell 世代の GPU(sm_120)で torch が動かない / ASR が CPU に落ちる / CUDA エラー | cu128 以上の torch が必要。`uv pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128` を extra より先に入れる |
| `metrics.vap_distance.vap_realtime_root or $VAP_REALTIME_ROOT must be set` | VAP-Realtime を clone し、`VAP_REALTIME_ROOT=/path/to/VAP-Realtime` を設定。`asset/cpc/` は clone に含まれる。`asset/vap/` には「VAP-distance の準備」の手順で `vap_state_dict_jp_kyoto_20hz_2500msec.pt` を置く |
| Hugging Face から `vap_state_dict_jp_kyoto_20hz_2500msec.pt` が見つからない(404) | 2026-10-01 に `main` から外されました。URL の `main` をリビジョン `fe24ac60d8fcc80463edde97ed90e3ceca5e5b88` に置き換える(「VAP-distance の準備」) |
| VAP チェックポイントを替えたら D_KL の値が変わった | 仕様どおりです。VAP は凍結した計測器で、チェックポイントが違えば値は比較できません。比較する全モデルを同じチェックポイントで採点し直し、`results.json` の `vap_checkpoint` を確認してください |
| α(`exp(-α·D_KL)` のスケール)がモデル追加のたびに変わる | α はモデル集合から自動較正されます。モデルを追加したら全モデルを 1 回の `score-offline` で採点し直してください |
| GPU があるのに torch が CPU で動く / `CUDA driver version is insufficient` | ロックされた torch は CUDA 13.0 版でドライバ 580 以上が必要。「GPU ドライバが古い場合」の手順で入れ替え、`uv run --no-sync` で実行する |
| 入れ替えた torch が元に戻る | `uv run` / `uv sync` はロックファイルに合わせて環境を同期し直す。`uv run --no-sync` か `UV_NO_SYNC=1` |
| `always_talk` が pyopenjtalk を求めて止まる | `uv sync --extra always-talk`、または config で `source_wav` に音声ファイルを指定 |
| `reconstruct` が `unusable speaker labels` と警告する | 転写の `speaker` が `A` / `B` の両方を含んでいない。`transcribe --wav ... --channel 0 1 --speaker A B` で作り直す |
| `infer` が `[<model>] skipped: unknown adapter` と言う | `adapter` の名前を確認。自作のアダプタは `"モジュール:クラス"`、出力を自分で置くなら `offline` |
| `sample-human-eval` が `webui/ not found` と言う | Web UI はリポジトリ直下の `webui/` を参照します。`git clone` した作業ツリーから(`uv sync` の editable install で)実行してください |
| 商用 API アダプタが接続できない | モデル名・プロトコルは 2026-08 時点のもの。各 API のドキュメントで現行のモデル名を確認し、config の `model` を差し替えてください |

## 謝辞

[Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench)(Lin et al., 2025、arXiv:2503.04721)の評価設計と、
[VAP-Realtime](https://github.com/inokoj/VAP-Realtime) / [MaAI](https://huggingface.co/maai-kyoto)(Inoue et al.、京都大学)の日本語 VAP モデルに基づいています。
