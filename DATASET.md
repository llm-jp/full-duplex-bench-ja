# Datasets

Following the llm-jp-eval convention, **this repository contains code only**,
with one exception: the short listening excerpts in `docs/samples/` (three events:
stimulus, human B and two model outputs) used by the leaderboard page. They are
cut from LLM-jp-Zoom1 and follow its terms; no other audio or corpus data is bundled. Evaluation clips are reconstructed locally
from a corpus you obtain separately, and the toolkit's license (Apache-2.0)
applies to the code, **not** to any data.

## Evaluation corpus: LLM-jp-Zoom1

The reference evaluation set is built from **LLM-jp-Zoom1** (~934 h of
Zoom-recorded casual Japanese dialogue, character-level time-aligned
transcripts). Apart from the excerpts above, it is **not redistributed here**.

- Obtain LLM-jp-Zoom1 under its own license/terms (its public release is in preparation; see below).
- Point `corpus.zoom1_root` in your config at the local copy.
- `fdbja reconstruct` then detects events and builds the 20 s evaluation clips
  and manifests under `corpus.eval_dir` (git-ignored).

### Release gating
LLM-jp-Zoom1 is being prepared for public release under terms that allow commercial use.
The release of the Full-Duplex-Bench-JA evaluation **data** is gated on that release; cite the
LLM-jp-Zoom1 dataset paper once available. Until then this repository
ships **code only**, plus the excerpts in `docs/samples/`.

### Provenance instead of a held-out split
`reconstruct` writes `manifest.jsonl` with the source `session_id`, the event `key` and
`event_time` of every clip. Users report `domain: in|out` per model (required by the config) and
can filter clips whose sessions overlap a model's training data using the manifest, or rebuild the
set from listed recordings only with `reconstruct --sessions FILE`. A fixed held-out split is not
maintained: which sessions count as "seen" depends on the model, so the manifest plus the `domain`
declaration is the portable form of that information.

### Trying the toolkit without the corpus
`examples/dummy/make_dummy_corpus.py` writes a small synthetic corpus in the same layout
(modulated noise and placeholder characters, no real speech). It exercises every stage of the
pipeline but its scores are meaningless; see the quick start in `README.md`.

## Models

`domain: in|out` in the config records whether a model was trained on
LLM-jp-Zoom1. This must be reported: in-domain models (e.g. LLM-jp-Moshi-v1)
share data with the eval corpus and are not directly comparable to
out-of-domain models (J-Moshi, Gemini).

## VAP / ASR checkpoints

- VAP-distance uses a **frozen** Japanese VAP checkpoint. The default is
  `vap_state_dict_jp_kyoto_20hz_2500msec.pt` from
  [maai-kyoto/vap_jp_kyoto](https://huggingface.co/maai-kyoto/vap_jp_kyoto) (MIT; pinned to revision
  `fe24ac60d8fcc80463edde97ed90e3ceca5e5b88`, SHA-256 `7ad0bf0b59e17a30ee55cee2aa6db89b84d64b670881ccd3f74b52ccede05d29`, because the file was removed from `main` on 2026-10-01), run
  through the [inokoj/VAP-Realtime](https://github.com/inokoj/VAP-Realtime) code (MIT) with
  its CPC encoder (`60k_epoch4-d0f474de.pt`, facebookresearch/CPC_audio, MIT). Download the
  checkpoint into `<VAP_REALTIME_ROOT>/asset/vap/`.
- The measurement instrument must stay fixed within a comparison: VAP-distance values from
  different checkpoints are not comparable. `docs/results.json` records which checkpoint
  produced each table (the `vap_checkpoint` key).
- ASR uses ReazonSpeech-ESPnet + WhisperX (wav2vec 2.0 forced alignment) by default; see the
  README for the other backends.
