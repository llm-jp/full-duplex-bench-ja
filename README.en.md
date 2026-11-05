# Full-Duplex-Bench-JA

Turn-taking evaluation toolkit for **Japanese full-duplex spoken dialogue models**.
It brings the evaluation design of [Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench)
(Lin et al., 2025) to Japanese and **needs no manual annotation**: the behaviour of the human
interlocutor (speaker B) in a real dialogue corpus (LLM-jp-Zoom1) is used as the reference.

The full README is in Japanese: [README.md](README.md). Leaderboard: <https://llm-jp.github.io/full-duplex-bench-ja/>
(ranked by TOR agreement, the binary metric of the original Full-Duplex-Bench. **VAP-distance is an
experimental metric**: its validation against human judgements is still in progress, so it is shown
for reference only and is disabled in the default config (`metrics.vap_distance.enabled: true` to
opt in). The current table (2026-11-04, nine systems) was scored with human-side boundaries
re-placed on the audio (`detection.boundaries: audio`; the default stays `transcript`), the paper's
transcripts (ReazonSpeech-ESPnet + WhisperX) and the MIT default checkpoint below,
`maai-kyoto/vap_jp_kyoto`; values from different boundaries, transcripts or checkpoints are not
comparable — see `boundaries`, `asr` and `vap_checkpoint` in `docs/results.json`. The leaderboard lists
Pause Handling and Smooth Turn-Taking; the backchannel (bc) task stays in the code as an experimental
task and is not scored by default. A summary of the human-judgement study (2026-11) is on the page).
Optional settings: `detection.boundaries: audio`, `detection.smooth_tt_roles.enabled: true` (Smooth TT
restricted to genuine turn transitions, 197 → 72 events) and `detection.system_activity: audio`.

## Requirements

- Python >= 3.10 and [uv](https://docs.astral.sh/uv/) for binary TOR.
- For VAP-distance: `.[vap]` (torch), a local clone of
  [VAP-Realtime](https://github.com/inokoj/VAP-Realtime) (ships the CPC encoder) and the Japanese VAP
  checkpoint `vap_state_dict_jp_kyoto_20hz_2500msec.pt` from
  [maai-kyoto/vap_jp_kyoto](https://huggingface.co/maai-kyoto/vap_jp_kyoto) (MIT) placed under
  `<VAP_REALTIME_ROOT>/asset/vap/`. Set `VAP_REALTIME_ROOT`. The file was removed from the `main`
  branch of the Hugging Face repository on 2026-10-01, so fetch it from a pinned revision:
  `https://huggingface.co/maai-kyoto/vap_jp_kyoto/resolve/fe24ac60d8fcc80463edde97ed90e3ceca5e5b88/vap_state_dict_jp_kyoto_20hz_2500msec.pt`
  (SHA-256 `7ad0bf0b59e17a30ee55cee2aa6db89b84d64b670881ccd3f74b52ccede05d29`, the file the leaderboard was scored with).
- One ASR extra for `fdbja transcribe`, which turns the recorded responses into character-level,
  time-aligned "jchat" JSON: `.[asr-reazon-whisperx]` (default; ReazonSpeech-ESPnet segments +
  WhisperX character alignment, the pipeline behind the published numbers), `.[asr-nemo]`
  (ReazonSpeech NeMo v2, token timestamps split evenly over a subword's characters, ~5x faster) or
  `.[asr-k2]` (ReazonSpeech k2 v2 via sherpa-onnx, CPU-friendly). The extras are mutually
  exclusive; weights are fetched from the Hugging Face Hub on first use. The aligner-free backends
  segment utterances differently, so their TOR is not directly comparable with the leaderboard.
- LLM-jp-Zoom1 to build the evaluation clips (see [DATASET.md](DATASET.md)), or your own two-channel
  dialogue recordings (see "Evaluation set from your own two-channel dialogues" below).
- `.[always-talk]` (pyopenjtalk) for the `always_talk` baseline's default speech loop (`source: auto`),
  or set `source_wav` to a speech file instead.
- GPU: the locked torch is the PyPI Linux build for CUDA 13.0 and needs NVIDIA driver 580 or newer.
  See "Older GPU drivers" below.

### Older GPU drivers (replacing torch, `--no-sync`)

If `nvidia-smi` reports `CUDA Version` 12.x (e.g. a 550-series driver = CUDA 12.4), the locked torch cannot
use the GPU (it falls back to CPU or fails with `CUDA driver version is insufficient`). Install the extras,
then replace torch with a build for your driver:

```bash
uv sync --extra asr-reazon-whisperx
uv pip install --reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu126   # CUDA 12.x drivers
uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
uv run --no-sync fdbja transcribe --config configs/default.yaml
```

**Afterwards run everything with `uv run --no-sync` (or set `UV_NO_SYNC=1`).** `uv run` and `uv sync` re-sync
the environment to the lockfile by default and would put the CUDA 13.0 torch back. Pick the index suffix
(`cu126`, ...) that your driver supports from the [PyTorch install page](https://pytorch.org/get-started/locally/).

## Quick start

Smoke test on a synthetic corpus (no LLM-jp-Zoom1, model server, API key or speech model needed;
the "speech" is modulated noise, so **the numbers are meaningless** and only check the wiring):

```bash
git clone https://github.com/llm-jp/full-duplex-bench-ja.git && cd full-duplex-bench-ja
uv sync
uv run python examples/dummy/make_dummy_corpus.py --out data/dummy_zoom1
uv run fdbja reconstruct --config examples/dummy/config.yaml
uv run fdbja infer --config examples/dummy/config.yaml           # the two trivial baselines
uv run python examples/dummy/energy_asr.py --config examples/dummy/config.yaml   # energy-based stand-in for ASR
uv run fdbja score-offline --config examples/dummy/config.yaml
```

Expected: `silence_baseline` has TOR 0 on both tasks and `always_talk_baseline` TOR 1 on Pause.
`examples/dummy/energy_asr.py` also shows how to register your own ASR with `fdbja.asr.register_backend`.

Real evaluation (point `corpus.zoom1_root` in `configs/default.yaml` at LLM-jp-Zoom1 and list your models):

```bash
uv sync --extra asr-reazon-whisperx
uv run fdbja reconstruct --config configs/default.yaml
uv run fdbja infer --config configs/default.yaml
uv run fdbja transcribe --config configs/default.yaml
uv run fdbja score-offline --config configs/default.yaml
```

Pipeline: `reconstruct` (corpus -> events -> 20 s clips) -> `infer` (stream speaker A to a model
adapter, record the response) -> `transcribe` (ASR of the response channel -> jchat) ->
`score-offline` (binary TOR, plus VAP-distance when enabled -> `results.json`, `summary.json`, `summary.csv`).
TOR depends on the ASR backend; report the backend name with your numbers and use the same one when comparing runs.
Run `uv run fdbja <command> --help` for arguments.

### Evaluation set from your own two-channel dialogues

The LLM-jp-Zoom1 evaluation data is not public yet (see [DATASET.md](DATASET.md)). Until then, or to evaluate
on another domain, build an evaluation set from your own recordings of two-person dialogues. You need:

- **Stereo audio** (wav / flac, any sample rate): ch0 = speaker A (played to the model), ch1 = speaker B (the
  human whose behaviour is the reference). Record the speakers separately and put them on two channels; a
  mixed mono recording does not work.
- **Character-level transcripts (jchat)**: one JSON per recording, a list with one item per character:
  `{"speaker": "A", "word": "こ", "start": 0.52, "end": 0.61}`. `speaker` is `A` (ch0) or `B` (ch1) and
  **both must occur** (the event detectors look at the sequence of A and B utterances); `word` is one
  character; `start` / `end` are seconds from the start of the recording.

`transcribe --wav` writes such transcripts. Give one `--speaker` per `--channel` to transcribe both channels
into one file (without `--speaker` every character is labelled `B`, the default for model responses):

```bash
uv run fdbja transcribe --wav my_dialogues/audio/*.wav --channel 0 1 --speaker A B --out my_dialogues/text
```

Then set `corpus.layout: generic`: every `*.json` under `text_dir` is one recording, paired with the audio file
of the same stem anywhere under `audio_dir` (`reconstruct --layout generic` does the same):

```yaml
corpus:
  layout: generic              # zoom1 (default): <text_dir>/<session>/<key>.json + <audio_dir>/<session>/<key>.wav
  text_dir: ./my_dialogues/text
  audio_dir: ./my_dialogues/audio
  eval_dir: ./data/my_eval
```

`infer` -> `transcribe` -> `score-offline` are unchanged. On synthetic recordings (meaningless numbers):

```bash
uv run python examples/dummy/make_dummy_corpus.py --out data/my_dialogues --layout generic --audio-only
uv run python examples/dummy/energy_asr.py --backend energy --wav data/my_dialogues/audio/*.wav \
    --channel 0 1 --speaker A B --out data/my_dialogues/text
uv run fdbja reconstruct --config examples/dummy/config_own_dialogues.yaml
uv run fdbja infer --config examples/dummy/config_own_dialogues.yaml
uv run python examples/dummy/energy_asr.py --config examples/dummy/config_own_dialogues.yaml
uv run fdbja score-offline --config examples/dummy/config_own_dialogues.yaml
```

Events are detected from the character timestamps only, so consider `detection.boundaries: audio` to re-place
the human boundaries on the audio. Clips are 20 s long; shorter recordings are not used. Numbers on your own set
are not comparable with the leaderboard; report the data source, event counts and settings.

### Outputs (`score-offline`, `analyze-profile`)

`score-offline` writes `results.json` (summary, scoring conditions and per-event decisions, including the event
times `t_start` / `t_end` and the system utterances near the judgement window, `sys_segments`), `summary.json`
(the same without the events; the shape of `docs/results.json`) and `summary.csv` (one row per task x system,
intervals split into `*_low` / `*_high`). The summary uses the keys of `docs/results.json`: `n` (events),
`n_clips`, `TOR`, `TOR_ci` (95% percentile bootstrap over events, 10000 replicates; `output.bootstrap: {n, seed}`),
`lat_median` / `lat_q1` / `lat_q3` (Smooth TT response latency of the taken turns, negative values clipped to 0
as on the leaderboard), `onset_median` / `onset_q1` / `onset_q3` (the same latency without clipping, -0.5 to
+2.0 s) and `overlap_rate` (share of taken turns that started before A finished), `event_mean_dkl` / `dkl_ci`
(VAP-distance, when enabled). The scoring conditions are recorded at the top level: `boundaries`,
`system_activity`, `smooth_tt_roles`, `asr` (the backend `transcribe` recorded per model), `vap_checkpoint`
(file name and the first 12 hex digits of its SHA-256) and `ci`.

`uv run fdbja analyze-profile --results results/results.json --out results/profile` summarises each system on
one table (`profile.md` / `profile.csv`): on Pause, the share of pauses it talked through (= TOR), filled with
backchannel-sized speech only, or left silent; on Smooth TT, the share of turn ends it took (= TOR), of which it
started before A finished, and left unanswered, with the quartiles of the signed onset; and a rough tendency
(talks over / stays silent / slow; thresholds are arguments). It also plots Pause talked-through share against
Smooth TT no-response share (`profile.png` / `profile.svg`, needs matplotlib from `.[analysis]`) and writes
`events.csv`: per event, its time, the judgement window, the system's utterances, the outcome and the reason.

## What is evaluated

| Task | Event | Desired behaviour |
|------|-------|-------------------|
| Pause Handling | intra-turn pause (0.4-1.0 s), human B silent | stay silent (TO=0) |
| Smooth Turn-Taking | A (>=4 s) -> gap (<0.4 s) -> B (>=4 s) | take the turn (TO=1) |
| Backchanneling | human B only backchannels during A (>=3 s) | backchannel only (TO=0); also backchannel frequency (`bc_freq`, per second of A; human: `human_freq`) and the timing JSD against the human onsets in 0.2 s bins (`jsd`, 0-1, 1.0 when the model never backchannels; `bc_any_rate` = share of events with at least one model backchannel and `jsd_cond` = JSD over those events), as in the original Backchannel task (experimental; not in the default `tasks`) |

Primary metric: **TOR** (binary takeover rate, ASR-based, same definition as Full-Duplex-Bench).
Experimental metric (off by default): **VAP-distance** (KL divergence between next-speaker
distributions of a frozen VAP model given the human response vs. the model response); its
validation against human judgements and existing metrics is still in progress, so label it as
experimental when reporting. Trivial baselines (`silence`, `always_talk`) are reported on every
task. A TurnNat-style likelihood score is included for comparison. The `always_talk` speech loop is synthesised on first use with Open JTalk (voice "Mei", CC BY 3.0; attribution in [src/fdbja/assets/README.md](src/fdbja/assets/README.md)) and needs `pyopenjtalk`, or pass `source_wav`.

### Speech intervals for the binary TOR (`detection.system_activity`)

`transcript` (default) takes the model's speech intervals from the character timestamps of the ASR, so the TOR
depends on how the ASR backend places character boundaries (an aligner may stretch the last character of a segment
over the following silence, which then counts as talking through a pause). `audio` takes the intervals from the
model channel of `combined.wav` with the same energy-based detector as `detection.boundaries: audio`, joins silences
of up to 0.3 s, and uses the ASR only for the character counts (shared among the speech intervals each ASR
utterance overlaps). On the evaluation set, two ASR backends disagree on 1-18 of 240 Pause events with `audio`
versus 17-73 with `transcript`. The published numbers are reproduced only with `transcript` and the transcripts
used for the paper; report `audio` as a separate condition. VAP-distance is unaffected.

## Model adapters

`moshi_ws` (Moshi-family WebSocket servers), `gemini_live` (Gemini Live API), `openai_realtime`
(OpenAI Realtime protocol: GPT-Realtime, Grok Voice), `silence` / `always_talk` (trivial baselines).
Commercial-API adapters were verified against the protocols and model names available **as of 2026-08**;
we do not guarantee to track API changes.

Your own model: implement `ModelAdapter.infer(input_audio, sample_rate) -> DialogueResponse`
(`src/fdbja/adapters/base.py`; `input_audio` is speaker A, mono 16 kHz; `infer` writes the returned
`response_audio` into `combined.wav`) and either

- name it in the config as `adapter: "package.module:ClassName"` (other keys except `name` / `domain` are
  constructor arguments; the module must be importable), or
- register it with `fdbja.adapters.register_adapter("name", ClassName)` before calling `fdbja.cli.main`.

`adapter: offline` runs nothing: put each clip's `combined.wav` (ch0 = speaker A, ch1 = your system) in place
yourself. An entry whose adapter cannot be built is skipped by `infer` with the reason (exit code 1) while the
other systems run; `score-offline` builds no adapters and scores every system whose outputs are present, so one
config serves both. `--model` restricts `infer`, `transcribe` and `score-offline`. Minimal example
(replies once the partner has been silent for 0.6 s):

```bash
PYTHONPATH=examples/adapters uv run fdbja infer --config examples/adapters/config.yaml
uv run python examples/dummy/energy_asr.py --config examples/adapters/config.yaml
uv run fdbja score-offline --config examples/adapters/config.yaml
```

## License

Code: Apache-2.0. Backends and data are **not** covered:

- Default VAP checkpoint `vap_state_dict_jp_kyoto_20hz_2500msec.pt` (maai-kyoto/vap_jp_kyoto, revision `fe24ac6`): MIT.
  VAP-Realtime code: MIT. CPC encoder: MIT. With the default configuration there is no
  non-commercial restriction on the scores. The older VAP-Realtime Japanese checkpoint
  (`vap_state_dict_jp_20hz_2500msec.pt`) is academic-only and loads through the same code path;
  its terms then apply to the results. The leaderboard no longer uses it.
- ReazonSpeech code and models (nemo-v2 / k2-v2 / espnet-v2): Apache-2.0. NVIDIA NeMo: Apache-2.0.
  WhisperX: BSD-2-Clause; its Japanese alignment model `jonatasgrosman/wav2vec2-large-xlsr-53-japanese`
  is Apache-2.0.
- Full-Duplex-Bench is CC BY-NC 4.0; this toolkit reimplements the event and TOR definitions
  independently and does not reuse its code.
- LLM-jp-Zoom1 and the excerpts in `docs/samples/` (three events; stimulus, human B and two models)
  follow the corpus terms (see DATASET.md). The model outputs kept there are from LLM-jp-Moshi-v1 and
  J-Moshi-ext (CC BY-NC 4.0 models); commercial-API outputs are not included.

## Citation

Please cite the ASJ Autumn 2026 talk (2-7-12): Yuto Abe, Mao Saeki, Shinnosuke Takamichi, Shinya Fujie,
Tetsuji Ogawa and Ryuichiro Higashinaka, "Full-Duplex-Bench-JA: 日本語フルデュプレックス音声対話モデル評価のための
ベンチマーク構築" (A benchmark for evaluating Japanese full-duplex spoken dialogue models), *Proceedings of the
Autumn Meeting of the Acoustical Society of Japan*, Sept. 2026. The same entry is in [CITATION.cff](CITATION.cff)
and as BibTeX in [README.md](README.md#引用).
