# Contributing

## ブランチ

- `main` が唯一の長寿ブランチ。直接 push しない。
- 作業は `main` から切ったトピックブランチで行う(例: `feat/asr-backend`, `fix/tor-window`, `docs/readme`)。

## コミット

- 形式は `type: 要約`。`type` は `feat` / `fix` / `docs` / `chore` / `test` のいずれか。
  - 例: `feat: analyze-pause にギャップ長別の bootstrap CI を追加`
- 1 コミット 1 目的。指標のロジックを変えるコミットは、既知値(リファレンス評価セット)との一致を本文に書く。

## Pull Request

- PR テンプレート(変更内容 / 確認したこと)を埋める。
- レビュアー 1 人の approve と CI(ruff + pytest、Python 3.10 / 3.12 / 3.13、extras の import 確認)の通過を条件に **squash merge**。
  squash 後のコミットメッセージも `type: 要約` にそろえる。
- 送る前に `make lint test`。

## リリース

1. `pyproject.toml` の `version` と `src/fdbja/__init__.py` の `__version__` を同じ値にし(`tests/test_packaging.py` が一致を確認)、`uv lock` を実行する。
2. `CITATION.cff` の `version` と `date-released`、README の「更新履歴」の見出しを更新する。
3. PR を `main` に merge したあと、`main` に `vX.Y.Z` の tag を付けて push する。
4. tag の push で `.github/workflows/publish.yml` が動き、lint とテスト、配布物の組み立て、検査
   (tag とバージョンの一致、メタデータに直接 URL の依存が無いこと、wheel だけを入れた環境で `fdbja measure` が動くこと)
   のあと PyPI に公開する。公開は Trusted Publishing(GitHub の environment `pypi`)で、API トークンはリポジトリに置かない。
   Actions の画面から手動で実行すると、検査までを行い公開はしない。
5. PyPI は同じバージョンの再アップロードを受け付けない。公開後に直す場合はバージョンを上げる。

## 触ってよいもの・注意するもの

- `docs/results.json` はリーダーボードの元データ。`score-offline` が書く `summary.json` から作り、手で数値を編集しない。
  `summary.json` のキー(`summary` の各行の `n` / `n_clips` / `TOR` / `TOR_ci` / `event_mean_dkl` / `dkl_ci` / `lat_median`、
  トップレベルの `boundaries` / `asr` / `vap_checkpoint` / `ci`)は `docs/results.json` と同じ。
  - `summary` と採点条件: 掲載の設定(`detection.boundaries: audio`)で全モデルを 1 回の `score-offline` で採点した `summary.json` をそのまま使う。
  - `turn_transition_only`: 同じ設定に `detection.smooth_tt_roles.enabled: true` を足した実行の `summary` の Smooth TT の行。
  - `transcript_boundaries`: `detection.boundaries: transcript` の実行の `summary`。
  - `rescored`(採点日)と、`boundaries` / `asr` の説明文を書き換えるかは更新する人が決める(`asr` は `transcribe` が記録したバックエンド名になる)。
  α はモデル集合に依存するので、モデル追加時は全モデルを 1 回の実行で採点し直す。
  95% 区間は事象の bootstrap(10000 回、百分位)で、乱数列が違うと端点が小数第 3 位で揺れることがある。
- `docs/samples/` の音声は LLM-jp-Zoom1 からの抜粋。追加・差し替えはコーパスの条件を確認してから。
- 音声・チェックポイント・評価ディレクトリはコミットしない(`.gitignore` 参照)。
