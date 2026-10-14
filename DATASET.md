# Datasets

Following the llm-jp-eval convention, **this repository contains code only.**
No audio or corpus data is bundled. Evaluation clips are reconstructed locally
from a corpus you obtain separately, and the toolkit's license (Apache-2.0)
applies to the code, **not** to any data.

## Evaluation corpus: LLM-jp-Zoom1

The reference evaluation set is built from **LLM-jp-Zoom1** (~934 h of
Zoom-recorded casual Japanese dialogue, character-level time-aligned
transcripts). It is **not redistributed here**.

- Obtain LLM-jp-Zoom1 under its own license/terms from the corpus owners.
- Point `corpus.zoom1_root` in your config at the local copy.
- `fdbja reconstruct` then detects events and builds the 20 s evaluation clips
  and manifests under `corpus.eval_dir` (git-ignored).

### Release gating
LLM-jp-Zoom1 becomes **publicly available for commercial use after the corpus
paper is published at APSIPA** (by a team member). Public release of
Full-Duplex-Bench-JA **data** is therefore gated on that publication; cite the
APSIPA dataset paper once available. Until then this repository
ships **code only** (the toolkit can be open-sourced independently of the data).

### Open items before public release
- [ ] **Data release timing**: LLM-jp-Zoom1 has been accepted at APSIPA 2026. Public release
      of the Full-Duplex-Bench-JA evaluation clips (and of the excerpts under `docs/samples/`)
      follows the corpus release; until then only the code is public and clips are rebuilt
      locally with `fdbja reconstruct`. Cite the APSIPA dataset paper once available.
- [x] **Provenance instead of a held-out split**: `reconstruct` writes `manifest.jsonl` with the
      source `session_id`, the event `key` and `event_time` of every clip. Users report
      `domain: in|out` per model (required by the config) and can filter clips whose sessions
      overlap a model's training data using the manifest. A fixed held-out split is not
      maintained: which sessions count as "seen" depends on the model, so the manifest plus
      the `domain` declaration is the portable form of that information.
- [ ] **Human-validation subset**: the crowdsourced validation (see `docs/human_eval.md`) is
      released as an item list with aggregated ratings (no worker identifiers) together with
      the clips, once the clips themselves can be released.

## Models

`domain: in|out` in the config records whether a model was trained on
LLM-jp-Zoom1. This must be reported: in-domain models (e.g. LLM-jp-Moshi-v1)
share data with the eval corpus and are not directly comparable to
out-of-domain models (J-Moshi, Gemini).

## VAP / ASR checkpoints

- VAP-distance uses a **frozen** Japanese VAP checkpoint. The default is
  `vap_state_dict_jp_kyoto_20hz_2500msec.pt` from
  [maai-kyoto/vap_jp_kyoto](https://huggingface.co/maai-kyoto/vap_jp_kyoto) (MIT), run
  through the [inokoj/VAP-Realtime](https://github.com/inokoj/VAP-Realtime) code (MIT) with
  its CPC encoder (`60k_epoch4-d0f474de.pt`, facebookresearch/CPC_audio, MIT). Download the
  checkpoint into `<VAP_REALTIME_ROOT>/asset/vap/`.
- The measurement instrument must stay fixed within a comparison: VAP-distance values from
  different checkpoints are not comparable. `docs/results.json` records which checkpoint
  produced each table (see `README.md`, "リーダーボード").
- ASR uses ReazonSpeech-ESPnet + WhisperX (wav2vec 2.0 forced alignment) by default; see the
  README for the other backends.
