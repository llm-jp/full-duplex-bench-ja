# Full-Duplex-Bench-JA

Turn-taking evaluation toolkit for **Japanese full-duplex spoken dialogue models**.
It brings the evaluation design of [Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench)
(Lin et al., 2025) to Japanese and **needs no manual annotation**: the behaviour of the human
interlocutor (speaker B) in a real dialogue corpus (LLM-jp-Zoom1) is used as the reference.

The full README is in Japanese: [README.md](README.md). Leaderboard: <https://llm-jp.github.io/full-duplex-bench-ja/>

## Requirements

- Python >= 3.10 and [uv](https://docs.astral.sh/uv/) for binary TOR.
- For VAP-distance: `.[vap]` (torch) and a local clone of
  [VAP-Realtime](https://github.com/inokoj/VAP-Realtime) with the Japanese VAP checkpoint
  (`vap_state_dict_jp_20hz_2500msec.pt`) and the CPC encoder. Set `VAP_REALTIME_ROOT`.
- One ASR extra for `fdbja transcribe`, which turns the recorded responses into character-level,
  time-aligned "jchat" JSON: `.[asr-reazon-whisperx]` (default; ReazonSpeech-ESPnet segments +
  WhisperX character alignment, the pipeline behind the published numbers), `.[asr-nemo]`
  (ReazonSpeech NeMo v2, token timestamps split evenly over a subword's characters, ~5x faster) or
  `.[asr-k2]` (ReazonSpeech k2 v2 via sherpa-onnx, CPU-friendly). The extras are mutually
  exclusive; weights are fetched from the Hugging Face Hub on first use. The aligner-free backends
  segment utterances differently, so their TOR is not directly comparable with the leaderboard.
- LLM-jp-Zoom1 to build the evaluation clips (see [DATASET.md](DATASET.md)).

## Quick start

```bash
git clone https://github.com/llm-jp/full-duplex-bench-ja.git && cd full-duplex-bench-ja
uv sync --extra asr-reazon-whisperx
uv run fdbja transcribe --config configs/default.yaml
uv run fdbja score-offline --config configs/default.yaml
```

Pipeline: `reconstruct` (corpus -> events -> 20 s clips) -> `infer` (stream speaker A to a model
adapter, record the response) -> `transcribe` (ASR of the response channel -> jchat) ->
`score-offline` (binary TOR + VAP-distance -> `results.json`). TOR depends on the ASR backend;
report the backend name with your numbers and use the same one when comparing runs.
Run `uv run fdbja <command> --help` for arguments.

## What is evaluated

| Task | Event | Desired behaviour |
|------|-------|-------------------|
| Pause Handling | intra-turn pause (0.4-1.0 s), human B silent | stay silent (TO=0) |
| Smooth Turn-Taking | A (>=4 s) -> gap (<0.4 s) -> B (>=4 s) | take the turn (TO=1) |
| Backchanneling | human B only backchannels during A (>=3 s) | backchannel only (TO=0) |

Metrics: **TOR** (binary takeover rate, ASR-based, same definition as Full-Duplex-Bench) and
**VAP-distance** (KL divergence between next-speaker distributions of a frozen VAP model given the
human response vs. the model response). Degenerate baselines (`silence`, `always_talk`) are
reported on every task. A TurnNat-style likelihood score is included for comparison.

## Model adapters

`moshi_ws` (Moshi-family WebSocket servers), `gemini_live` (Gemini Live API), `openai_realtime`
(OpenAI Realtime protocol: GPT-Realtime, Grok Voice). Commercial-API adapters were verified against
the protocols and model names available **as of 2026-08**; we do not guarantee to track API changes.

## License

Code: Apache-2.0. Backends and data are **not** covered:

- The Japanese VAP checkpoint from VAP-Realtime is distributed for **academic purposes only**
  (its code is MIT; the CPC encoder is MIT). Results that include VAP-distance therefore cannot be
  used commercially.
- ReazonSpeech code and models (nemo-v2 / k2-v2 / espnet-v2): Apache-2.0. NVIDIA NeMo: Apache-2.0.
  WhisperX: BSD-2-Clause; its Japanese alignment model `jonatasgrosman/wav2vec2-large-xlsr-53-japanese`
  is Apache-2.0.
- Full-Duplex-Bench is CC BY-NC 4.0; this toolkit reimplements the event and TOR definitions
  independently and does not reuse its code.
- LLM-jp-Zoom1 and the excerpts in `docs/samples/` follow the corpus terms (see DATASET.md).
  J-Moshi-family outputs in `docs/samples/` are outputs of CC BY-NC 4.0 models.

## Citation

See [CITATION.cff](CITATION.cff) (RTCA 2026 and ASJ Autumn 2026; bibliographic details will be
updated after acceptance).
