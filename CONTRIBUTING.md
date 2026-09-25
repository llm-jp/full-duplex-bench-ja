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
- レビュアー 1 人の approve と CI(ruff + pytest、Python 3.10 / 3.12)の通過を条件に **squash merge**。
  squash 後のコミットメッセージも `type: 要約` にそろえる。
- 送る前に `make lint test`。

## 触ってよいもの・注意するもの

- `docs/results.json` はリーダーボードの元データ。`score-offline` の出力で丸ごと差し替える(手で編集しない)。
  α はモデル集合に依存するので、モデル追加時は全モデルを 1 回の実行で採点し直す。
- `docs/samples/` の音声は LLM-jp-Zoom1 からの抜粋。追加・差し替えはコーパスの条件を確認してから。
- 音声・チェックポイント・評価ディレクトリはコミットしない(`.gitignore` 参照)。
