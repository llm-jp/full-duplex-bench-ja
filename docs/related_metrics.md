# 関連指標・関連ベンチマークとの関係

最終更新: 2026-08-18

VAP-distance と同時期に、同じ問題意識の研究が複数出ています。本書は差分と、
比較実験として何を回すべきかを記録します。

## 1. TurnNat (arXiv:2607.01345, 2026-07-01)

Zhang, Thebaud, Tinchev, Ravichandran, Moro-Velazquez.
*TurnNat: Automatic Evaluation of Turn-Taking Naturalness in Dyadic Spoken
Dialogue.*

### 手法

- 因果的なターンテイキング予測器で **256 クラスの将来音声活動**を推定。
  H = 2 s、ビンは [0,200] / [200,600] / [600,1200] / [1200,2000] ms —
  本リポジトリが使う日本語 VAP チェックポイントと**同一の定式化**。
- VAD で 200 ms 以上の活動区間を取り、各発話の onset / offset の**直前 2 秒**を
  TBU (turn-taking boundary unit) とする。
- TBU 内のフレーム NLL を平均し、対話レベルで
  `m(x) = -[λ·MeanNLL + (1-λ)·TailNLL]`(TailNLL = 上位 K 個の平均)。

### 検証

- Seamless Interaction から 2,251 dyad、20–25 s。
- 摂動 5 種: 遅延応答 (1.2–2.0 s) / 早すぎる割込み (1.2–2.5 s) /
  shift すべき所で hold / hold すべき所で shift / 過剰な相槌。
- 人手評価: 英語母語話者 18 名、150 ペア、3 判定/ペア。
  自然選好率 0.680 ± 0.043、多数決一致 0.780、Krippendorff ordinal α 0.341。
- 最良系 (DualTurn D4, α=8) で matched-pair accuracy 88.0%、C-index 0.676。
- **公開 VAP チェックポイントをスコアラーとして baseline (V0/V1) に含めており、
  fine-tune した自前モデルに負けている。**

### VAP-distance との差分

| 観点 | TurnNat | VAP-distance (本リポジトリ) |
|---|---|---|
| 基盤表現 | 256 クラス VAP (H=2s, 4 ビン) | **同一** |
| スコアの型 | 観測音声の NLL = 学習分布からの絶対的逸脱 | 同一事象での人間応答 vs モデル応答の前向き KL |
| 参照 | 不要 | **同一事象の人間応答が必要**(ペア設計) |
| 単位 | 境界前 2 秒の TBU → 対話レベル (mean + tail) | イベント単位 → ポーズ長で層別化可能 |
| 評価対象 | 自然対話への**人工摂動** | **実 SDM の実応答** |
| 言語 | 英語 (Seamless Interaction) | 日本語 (LLM-jp-Zoom1) |
| 人手検証 | 済 | 未(Study A/B を用意済み、`human_eval.md`) |

TurnNat 自身の limitation に「本手法は自然対話の統制された摂動を評価するもので、
実運用の音声対話システムが起こす失敗(ASR 誤り、意味的誤解、韻律不整合、
システム側レイテンシ)は網羅しない」とある。**実 SDM 出力上での妥当性検証**が
本リポジトリの取りうる差分。

### 比較実験でやること

1. `metrics/turnnat.py` を同じ日本語クリップに適用し、VAP-distance と併記する。
   - 先に **クラス符号の bit 順をチェックポイントと突き合わせる**こと
     (`pattern_to_class` の docstring 参照)。ずれても NLL は一見もっともらしい
     値になるため、既知の刺激で argmax クラスを確認する。
2. TurnNat の摂動 5 種を日本語クリップに実装し、両指標の matched-pair accuracy を
   比較する(TurnNat 側の土俵での比較)。
3. 人手評価 (Study A/B) と両指標の相関を取る(本リポジトリ側の土俵での比較)。
   TurnNat は人手評価を「ベンチマークの妥当性確認」に使い、スコアの較正には
   使っていないので、ここは未踏。
4. 素の VAP チェックポイントが弱いスコアラーだという彼らの結果を踏まえ、
   fine-tune 版との差も見ておく。

## 2. M3-DuplexBench (arXiv:2607.29125, NTT, 2026-07-31)

Ryo Fukuda et al. *M3-DuplexBench: A Multi-Turn, Multilingual, Multidomain
Benchmark for Full-Duplex Spoken Dialogue Models.*

英語 + **日本語**の full-duplex ベンチマーク。事象タイプ (turn shift / long pause
/ backchannel / barge-in) と TOR の採用が本リポジトリと重なる。

| | M3-DuplexBench | Full-Duplex-Bench-JA |
|---|---|---|
| 日本語コーパス | MagicData 約 10 時間 | LLM-jp-Zoom1 934 時間 |
| 日本語事象数 (chat) | SHIFT 2,153 / PAUSE 1,857 / BC 1,093 / BARGE_IN 785 | pause 240 / smooth_tt 197 / bc 314 |
| 評価モデル | J-Moshi, LLM-jp-Moshi の 2 つ | 6 つ(+ GPT-Realtime / Gemini Live / Grok-Voice) |
| TOR の定義 | 窓の 0.4 秒前以降に発話開始で takeover(音響オンセット基準) | ASR 転写ベース |
| 連続指標 | なし | VAP-distance |
| 内容評価 | あり (LLM-as-a-Judge: relevance / consistency / QA acc) | なし |
| 文脈条件 | None / User / Full の 3 条件(teacher-forced) | 単発事象 |

重複は「事象分類 + TOR + J-Moshi / LLM-jp-Moshi」まで。**日本語で商用モデルを
評価しているのは現状こちらだけ**で、逆に content 指標と多ターン文脈は向こうの強み。

**TOR の定義が両者で異なる**(音響オンセット基準 vs ASR 転写基準)ため数値の
直接比較はできないが、同じモデルで TOR が大きく食い違うこと自体が
「二値指標は実装依存で脆い」という主張の一次証拠になる。

## 3. その他(2026 年の周辺動向)

- Full-Duplex-Bench v1.5 (ICASSP 2026) — overlap handling。
- Full-Duplex-Bench v2 (arXiv:2510.07838, ACL 2026) — 自動 examiner による多ターン
  評価。turn-taking fluency + 指示追従 + タスク遂行。
- Full-Duplex-Bench v3 (arXiv:2604.04847) — 実人間音声の非流暢性 + 連鎖 API 呼び出し。
- ICASSP 2026 HumDial Challenge 総括 (arXiv:2604.21406) — HumDial-FDBench。
- Instruct-FD (arXiv:2607.20460) — ターンテイキングの**指示追従**。最高でも
  adherence 64.4%。
- full-duplex 対話システムのサーベイ (arXiv:2606.19453)。
