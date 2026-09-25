# 人手評価(Human evaluation)

自動指標(二値 TOR / VAP-distance)を人手で検証するためのクラウドソーシング
用マニフェストを生成する。設計の背景・分析方法は研究側の設計書を参照。

```
fdbja sample-human-eval --config configs/default.yaml --out ./human_eval \
    [--n-a 60] [--n-b 50] [--seed 42] [--no-audio]
```

出力(`--out` 配下、そのままホストできる自己完結バンドル):
- `manifest.json` … Web UI が読む設問・項目定義
- `index.html` / `app.js` / `style.css` … クラウドソーシング画面(`webui/` から複製)
- `audio_a/*.wav` … Study A の刺激音声(話者 A を決定点で切り詰めたモノラル)
- `audio_b/*.wav` … Study B の刺激音声(モデル出力 combined.wav を複製)
- `study_a.csv` / `study_b.csv` … 分析用の項目表

`--no-audio` を付けると音声複製を省略(CSV と manifest のみ)。

## クラウドソーシング画面(Web UI)

`webui/` に依存ライブラリなしの静的アプリ。`sample-human-eval` が出力先へ
自動複製するので、`--out` ディレクトリをそのまま静的配信すればよい:

```
cd <out>
python -m http.server 8000   # 任意の静的ホスティングで可
# ブラウザで http://localhost:8000/ を開く
```

画面の流れ:
1. 作業者 ID と評価セット(A / B / 両方)を入力して開始
2. 各項目で音声を**最後まで再生**(再生完了で回答欄が有効化)
3. Study A は 3 択、Study B は 5 段階 + カテゴリを回答
4. 完了後、提出用トークン `FDBJA1.<base64>.<完了コード>` をコピーして
   プラットフォームの回答欄に貼り付けて提出(JSON ダウンロードは予備)

項目順は作業者 ID をシードに固定シャッフル(リロードしても同順)。
**完了コードは回答内容のハッシュ**(FNV-1a)なので、提出後にトークンを
1 文字でも編集するとコードが合わなくなり検収で検出できる。回収した
回答欄のテキストをまとめて `webui/decode_submissions.py <file> <dir>` に
通すと、検証済みの `responses_*.json` が生成され(改竄・極端な高速回答は
フラグ表示)、そのまま `analyze-human-eval` に渡せる。

## Study A:正解設計の妥当性(モデル非依存)

各事象で、話者 A の文脈(決定点まで)だけを聞かせ、適切な振る舞いを選ばせる。
人間 B の実際の行動から導いた自動正解との一致率・Fleiss κ を算出する。

- 設問:「今聞いた発話の直後、対話の相手として最も自然な振る舞いはどれですか?」
- 選択肢:
    1. すぐに話し始める(ターンを取る)
    2. 黙って相手の発話を待つ/聞き続ける
    3. 短い相槌だけ打つ(「うん」「はい」など)
- 期待(自動正解): smooth_tt→1, pause→2, bc→3(CSV の `expected_choice` は 0 始まり)

## Study B:指標の妥当性(モデル出力の評価)

各 (事象 × モデル) で、話者 A + モデル応答を聞かせ、タイミングの自然さを評価。
モデルが二値 TO で一致する事象と**不一致の事象**が両方含まれるよう層化する
(不一致事象は「二値が同点とする群の内部で VAP-distance が人手と相関するか」を
見る最重要サンプル。`models_disagree=1`)。

- 主設問(5 段階):「このシステムの応答の『間の取り方・発話のタイミング』は
  自然で適切でしたか?」(内容ではなくタイミングで判断、1: 非常に不自然 〜 5: 非常に自然)
- 補助(カテゴリ):「このシステムは実際に何をしましたか?」
    - 適切なタイミングで話し始めた / 話すべきでない所で割り込んだ・喋りすぎた /
      話すべき所で黙ってしまった / 適切に相槌を打った / 相槌が過剰・不自然だった / その他

CSV には分析用に `TO`(モデルの二値判定)・`desired_TO`・`models_disagree` を
記録する。集計時に Spearman(VAP-distance, 人手平均)、二値内部での相関、
偏相関(二値・遅延を共変量)を計算する。

## 集計・分析(回収後)

作業者から集めた `responses_*.json` を 1 ディレクトリに置き、バンドル
(`study_a.csv` / `study_b.csv`)と、任意で `score-offline` の `results.json`
(VAP-distance を結合)を指定して集計する:

```
fdbja analyze-human-eval \
    --responses ./collected \           # responses_*.json を集めたディレクトリ
    --bundle ./human_eval \             # sample-human-eval の出力(study_*.csv)
    --results ./out/results.json \      # 任意: VAP-distance を結合
    --out ./human_eval/analysis
```

出力:
- `human_eval_summary.md` … 主要指標の表(下記)
- `human_eval_analysis.json` … 全数値(機械可読)
- `study_b_items.csv` … (事象×モデル)ごとの人手平均 Likert・二値 TO・VAP-distance

算出する指標:
- **Study A**:多数決一致率(自動正解との一致)、Fleiss κ、pairwise 一致率、
  「判断困難」率(`results.json` 不要)。
- **Study B**:ICC(1)(評価者間信頼性)、Spearman(VAP-distance, 人手 Likert)
  +ブートストラップ 95%CI、AUC(二値 TO 正誤→Likert)、**二値が同点とする群の
  内部での Spearman**(核心)、二値・遅延を統制した偏相関。

> 評価者数が項目ごとに異なるクラウドソーシングを想定し、ICC は不釣り合いにも
> 対応する一元配置 ICC(1)、一致率は pairwise を併記する。Fleiss κ は最頻評価者数の
> 項目に限定して算出する。

## 品質管理

- 各タスクにつき注意チェック(明白事例)を混在させる。
- 日本語母語話者を対象。Fleiss κ / ICC を報告(クラス不均衡時は κ が低く出る点を明記)。
