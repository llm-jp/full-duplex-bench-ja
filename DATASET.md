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
APSIPA dataset paper once available. Until then this repo stays **private** and
ships **code only** (the toolkit can be open-sourced independently of the data).

### Open items before public release
- [ ] Time the data release with the APSIPA LLM-jp-Zoom1 publication.
- [ ] Provide a **held-out split** excluding sessions used to train any
      evaluated model (e.g. LLM-jp-Moshi-v1 is in-domain — document leakage).
- [ ] Human-validation subset (metric ↔ human judgment) for release.
- [ ] Rewrite the README status block (currently "private / 開発中" with
      internal progress notes) into release notes / versioning.

## Models

`domain: in|out` in the config records whether a model was trained on
LLM-jp-Zoom1. This must be reported: in-domain models (e.g. LLM-jp-Moshi-v1)
share data with the eval corpus and are not directly comparable to
out-of-domain models (J-Moshi, Gemini).

## VAP / ASR checkpoints

- VAP-distance uses a frozen Japanese VAP checkpoint from
  [inokoj/VAP-Realtime](https://github.com/inokoj/VAP-Realtime)
  (`vap_state_dict_jp_20hz_2500msec.pt` + CPC encoder); obtain under its terms.
- ASR uses ReazonSpeech-ESPnet + WhisperX alignment.
