# CLAUDE.md

## 重要なルール

- コミット前に `make lint test` を通す(CI と同じ内容: ruff + pytest、Python 3.10 / 3.12 / 3.13)。
- コミットは `type: 要約`(feat / fix / docs / chore / test)。詳細は [CONTRIBUTING.md](CONTRIBUTING.md)。
- 音声(`*.wav` / `*.mp3` など)、チェックポイント(`*.pt`)、評価ディレクトリ(`data/`, `results/`)はコミットしない。
  例外は `docs/samples/*.mp3`(リーダーボードの試聴用)だけ。
- 指標のロジック(`metrics/`, `data/event_detection.py`)を変えたら、リファレンス評価セットで既知値と一致することを確認して PR 本文に書く。

## 落とし穴

- 合成コーパスでの通し試験は `tests/test_dummy_pipeline.py`(`examples/dummy/` を読み込む)。`examples/dummy/` を変えたらこれも通す。

- `score-offline` は GT の jchat とモデル出力の ASR jchat が**両方**あるクリップだけを採点し、無いものは黙ってスキップする。
  「scored 0 clips」はデータ配置の問題であることが多い。
- ASR は `fdbja transcribe`(`asr/`)。バックエンドは `reazonspeech_whisperx`(既定。論文・リーダーボードの経路)/
  `reazonspeech_nemo` / `reazonspeech_k2` で、extra(`asr-reazon-whisperx` / `asr-nemo` / `asr-k2`)は排他。
  NeMo / k2 は整列器なしで速いが発話の切れ目が多く出るので TOR の絶対値はリーダーボードと比較不可。
  サブワード内の文字時刻は均等分割 + 0.5 秒以内の橋渡しの近似(`asr/_subword.py`)。
- `.[asr-reazon-whisperx]` は ESPnet の重みを `huggingface_hub` で直接読む(`espnet_model_zoo` の Python 3.10 向け版は Hub の
  リダイレクトで落ちる)。`ctc-segmentation` は numpy 2 に対して `--no-build-isolation` で再ビルドが要ることがある。
- α(`exp(-α·D_KL)`)はモデル集合から自動較正されるため、モデルを 1 つ足すと全モデルのスコアが変わる。
  `docs/results.json` を更新するときは全モデルを 1 回の実行で採点し直す。
- `sample-human-eval` はリポジトリ直下の `webui/` を `__file__` 相対で探す。wheel でインストールすると見つからない。
- 商用 API の `gpt-realtime` と `grok-voice-latest` は別名で、指す先のモデルは提供元の更新で替わる。リーダーボードの出力の取得時期とモデル ID は README「モデルアダプタ」の表が正(Gemini は 2026-04、他の 2 つは 2026-07)。表示名にバージョンを足すときは取得時点の証跡を確認する。
- TOR の時刻は既定(`detection.system_activity: transcript`)では ASR の文字の終端に依存する。既定の `reazonspeech_whisperx` は区間末の文字を後ろの無音まで伸ばすため、論文時の書き起こしより Pause の TOR が高く出る。ASR を替えて比べるなら `audio`(`data/system_activity.py`)。
- VAP チェックポイントはリポジトリに同梱しない。既定は MIT の `maai-kyoto/vap_jp_kyoto`(20 Hz / 2.5 s)。2026-10-01 に HF の `main` から外れたのでリビジョン `fe24ac6` で取得する(README「VAP-distance の準備」)。旧来の VAP-Realtime 配布版は学術目的限定。チェックポイントを替えたら `metrics/turnnat.py` の手順でビット順を再検証し、比較対象を全部同じ重みで採点し直す。
