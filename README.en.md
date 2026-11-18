# Full-Duplex-Bench-JA

Turn-taking evaluation toolkit for **Japanese full-duplex spoken dialogue models**.
It brings the evaluation design of [Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench)
(Lin et al., 2025, [arXiv:2503.04721](https://arxiv.org/abs/2503.04721)) to Japanese and **needs no manual annotation**: one speaker's (A's)
audio from a real dialogue corpus (LLM-jp-Zoom1) is played to the model, and the events (turn ends and mid-turn
pauses) are taken automatically from the turn changes and pauses in the recordings. Smooth Turn-Taking measures
whether the model takes the turn where the human interlocutor (speaker B) did. Pause Handling is scored with a
metric that agreed with listeners' judgements in a listening test (whether the model reacted to the pause, and
whether it avoided talking over the partner).

**Scope**: of the four Full-Duplex-Bench v1 tasks (Pause Handling, Backchanneling, Smooth Turn-Taking,
User Interruption), the two that can be measured with the partner's (speaker A's) audio held fixed — Pause
Handling and Smooth Turn-Taking — are implemented on real Japanese dialogues. For Backchanneling (the system's
backchannels) only the event definition is published; it is not scored. User Interruption and the overlap
scenarios of v1.5 (user backchannels, third-party and background speech) are not covered yet (see Roadmap).

The full README is in Japanese: [README.md](README.md). Leaderboard: <https://llm-jp.github.io/full-duplex-bench-ja/>

- **Ranking**: per task. Smooth Turn-Taking is ranked by TOR, the binary metric of Full-Duplex-Bench (the
  share of turn ends at which the system took the turn). Pause Handling is ranked, from 0.2.0, by the pause
  reaction penalty: the share of pauses with no reaction plus the share with a barge-in (0-2, lower is
  better). The Pause TOR improves the more a system stays silent, and in the listening test it pointed the
  opposite way to the listeners' preference (see the human-judgement validation section of the leaderboard,
  2026-11), so it is kept as a reference value and not used for the ranking. No combined rank is given.
- **VAP-distance is an experimental metric**: it has been compared with human judgements in one listening
  test (2026-11) and did not agree with the listeners on Pause, so it is shown for reference only and is
  disabled in the default config (`metrics.vap_distance.enabled: true` to opt in).
- **Scoring conditions of the current table** (rescored on 2026-11-15, nine systems): human-side boundaries re-placed on
  the audio (`detection.boundaries: audio`; the default stays `transcript`), the transcripts used for the ASJ
  talk paper (see [Citation](#citation); "the paper" below) made with ReazonSpeech-ESPnet + WhisperX, and the
  MIT default checkpoint `maai-kyoto/vap_jp_kyoto`. Values from different boundaries, transcripts or
  checkpoints are not comparable — see `boundaries`, `asr` and `vap_checkpoint` in `docs/results.json`.
- **Tasks**: the leaderboard lists Pause Handling and Smooth Turn-Taking; the backchannel (bc) task stays in
  the code as an experimental task and is not scored by default.

Optional settings: `detection.boundaries: audio`, `detection.smooth_tt_roles.enabled: true` (Smooth TT
restricted to genuine turn transitions, 197 → 72 events) and `detection.system_activity: audio`.

**Quick start** (with LLM-jp-Zoom1 at hand; a one-minute smoke test on synthetic data is in [Quick start](#quick-start) below)

```bash
uv sync --group asr-reazon-whisperx                        # 1. install (pick one ASR backend)
uv run fdbja reconstruct --config configs/default.yaml     # 2. build the eval clips (set corpus.zoom1_root)
# responses: `fdbja infer` + `fdbja transcribe`, or place combined.wav ([Model adapters](#model-adapters))
uv run fdbja score-offline --config configs/default.yaml   # 3. score -> results/summary.json
uv run fdbja leaderboard --reference docs/results.json     # 4. rank like the leaderboard, next to the published rows
```

**Contents**: [Requirements](#requirements) / [Quick start](#quick-start) / [What is evaluated](#what-is-evaluated) /
[Model adapters](#model-adapters) / [Known limitations](#known-limitations) / [Roadmap](#roadmap) / [License](#license) / [Citation](#citation)

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
- One ASR dependency group for `fdbja transcribe`, which turns the recorded responses into character-level,
  time-aligned "jchat" JSON: `asr-reazon-whisperx` (default; ReazonSpeech-ESPnet segments +
  WhisperX character alignment, the pipeline behind the published numbers), `asr-nemo`
  (ReazonSpeech NeMo v2, token timestamps split evenly over a subword's characters, ~5x faster) or
  `asr-k2` (ReazonSpeech k2 v2 via sherpa-onnx, CPU-friendly). Install one from a clone with
  `uv sync --group <name>` (with pip 25.1 or newer: `pip install -e . --group <name>`). The groups are
  mutually exclusive; weights are fetched from the Hugging Face Hub on first use. The aligner-free backends
  segment utterances differently, so their TOR is not directly comparable with the leaderboard. They are
  groups rather than extras because ReazonSpeech is distributed from its git repository only and PyPI does
  not accept packages with direct-URL dependencies (in 0.1.0 they were extras: read `--extra asr-...` as
  `--group asr-...`).
- LLM-jp-Zoom1 to build the evaluation clips (see [DATASET.md](DATASET.md)), or your own two-channel
  dialogue recordings (see "Evaluation set from your own two-channel dialogues" below).
- `.[always-talk]` (pyopenjtalk) for the `always_talk` baseline's default speech loop (`source: auto`),
  or set `source_wav` to a speech file instead.
- GPU: the locked torch is the PyPI Linux build for CUDA 13.0 and needs NVIDIA driver 580 or newer.
  See "Older GPU drivers" below.

### Older GPU drivers (replacing torch, `--no-sync`)

If `nvidia-smi` reports `CUDA Version` 12.x (e.g. a 550-series driver = CUDA 12.4), the locked torch cannot
use the GPU (it falls back to CPU or fails with `CUDA driver version is insufficient`). Install the dependencies,
then replace torch with a build for your driver:

```bash
uv sync --group asr-reazon-whisperx
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
uv sync --group asr-reazon-whisperx
uv run fdbja reconstruct --config configs/default.yaml
uv run fdbja infer --config configs/default.yaml
uv run fdbja check-responses --config configs/default.yaml --out results/response_check   # check the recordings
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

### Measuring your own two-channel recordings (`fdbja measure`)

Timing statistics of any dialogue recording with the user and the system on separate channels, without the
evaluation set. No transcript and no annotation; the base dependencies (numpy, soundfile, pyyaml) are enough
(`uv sync`, no extra). From 0.2.0 the package is distributed on PyPI as well: `pip install fdbja`
and then `fdbja measure rec.wav`, without cloning the repository (drop `uv run` from the commands below).

```bash
uv run fdbja measure rec.wav --condition model=my-model      # ch0 = user, ch1 = system (--user-channel 1 to swap)
uv run fdbja measure recordings/ --json out.json             # a directory: one row per recording and the total
uv run fdbja make-profile human_dialogues/ --language ja --relationship strangers \
    --situation chat --medium "video call" -o profile.json   # reference profile from human-human recordings
uv run fdbja measure rec.wav --profile profile.json          # adds the deviation from the human norm (same as --target both)
uv run fdbja measure rec.wav --profile profile.json --target human-likeness   # the deviation from the human norm only
```

```python
from fdbja import measure, load_profile, make_profile
m = measure("rec.wav", conditions={"model": "my-model"})    # two mono files: measure("user.wav", "system.wav")
print(m.stats["turn_end"]["latency_median"], m.signature)    # arrays: measure(x, sample_rate=16000)
```

It reports, from events detected in the audio with the frame-energy detector: the signed response latency at
the user's turn ends (median, quartiles, count; negative = started before the user had finished) and the share
of turn ends without a response, with the number and share of turn ends that were not counted because the
system was already speaking; the share of pauses inside the user's turns (0.4 to 1.0 s) in which the system
was voiced, split into short (at most 1.5 s) and long; how often the system starts while the user is speaking
and the share of time both are voiced; and each speaker's share of speaking time.

- **Not comparable with the leaderboard.** The events are detected in the recording itself (not the fixed
  events of the evaluation set) and no transcript is used; the keys are named differently from `results.json`.
- Short / long speech in a pause is judged **by length only**; whether it is a backchannel or a take-over is
  not examined.
- Report the numbers with the `signature` of the output (version, detection settings, profile, scored target
  and channel assignment in one line, e.g.
  `fdbja:0.2.0|measure|vad:energy(p56)|events:default|profile:none|target:listener-judgement|ch:user=0,sys=1`)
  and the recording conditions, which `--condition key=value` copies to the output (model, voice, prompt,
  network, ...; any key). `--condition-file prompt=prompt.txt` records only a hash of the file.
- Thresholds are changed with `--set turn_min_seconds=4` (`config={"turn_min_seconds": 4}` in Python) and then
  appear in the signature. The defaults are the benchmark's rule constants. Setting `turn_min_seconds` to 4
  does not bring the events closer to the annotated events of the benchmark (the default is 1.5).
- A user turn end that arrives while the system is speaking is not counted (no response latency can be
  defined). For a system that keeps talking this share (`turn_end.skipped_system_speaking_rate`) is large, and
  the no-response rate and the latency then describe the remaining turn ends only; a warning is printed from
  0.3 (a guide chosen by hand, not a validated value).
- **What to score against (`--target`).** Besides the descriptive statistics (layer a), two targets can be
  scored. They are returned under separate keys; no argument changes the meaning of a key. The chosen target
  is part of the `signature`; `layers` in the JSON says which value belongs to which layer, and
  `layers_reported` which layers were written.
  - `listener-judgement` (layer c; the default without `--profile`): `pause_reaction_penalty`, the sum of the
    no-reaction and barge-in rates (0 to 2, lower is better) of the pause rule of `metrics/pause_response.py`
    applied to the pauses detected here. No profile needed. The rule agreed with what third-party listeners
    judged natural in one pre-registered listening test, run separately from the test on which its
    composition was chosen (see "Proposed metrics"). That was confirmed on the fixed pause events of the
    benchmark, not on the pauses that `measure` detects itself, and no metric for turn ends has been confirmed
    in this way. What listeners judge natural is not necessarily what the person in the conversation prefers.
  - `human-likeness` (layer b; needs `--profile`): the share of turn ends whose response falls inside the
    profile's tolerance range, and the distance from it. **A turn end without a response counts as outside
    the range.** This says how close the timing is to the human behaviour in the profile, which is not
    necessarily what listeners judge to be natural.
  - `both` (the default with `--profile`): both.

  `human-likeness` and `both` without `--profile` are an error (there is no norm to compare with). The
  spelling `listener-judgment` is accepted too. In Python:
  `measure("rec.wav", profile="profile.json", target="human-likeness")`.

  A profile made with `make-profile` carries the scores of its own recordings under the same settings
  (`reference_score`), and `measure` prints them next to each score. Human recordings do not reach 100 % in
  the range either, so interpret a score against these values (the same recordings the range was derived
  from: an optimistic guide). The only bundled profile, `generic`, is made
  of the rule constants and is **not a norm derived from human dialogue data** (its tolerance range is the
  response window, so the in-range share equals the response rate). Timing norms depend on the language, the
  relationship, the situation and the medium: build a profile from matching human-human recordings with
  `make-profile`, which records those four fields, the source, the numbers of recordings and events, and the
  detection settings.
- Energy-based: noise, laughter and breaths are not told apart from speech, and noise, cross-talk between the
  channels, clipping and a very low recording level shift the numbers ("Supported recording conditions" below).

#### Supported recording conditions

What follows was **checked on artificially degraded recordings only**; it was **not checked on real room noise,
reverberation or coded (compressed) speech**. Recordings of the evaluation set (16 kHz, clips of a few tens of
seconds) were degraded in one of three ways — stationary noise (white, pink) added, the other channel mixed in at
a fixed gain with a delay of 0.05 s, the level changed by a fixed gain — and the range was sought in which the
numbers of `measure` (event counts, median latency, the rates) stay where they were.

- Channel separation: the other speaker's voice should leak in at −30 dB or less when it leaks one way (the
  system's voice into the user channel) and at −40 dB or less when it leaks both ways. Above that the leaked
  voice is counted as speech of the channel: overlap and pauses start to grow, and at −20 dB the overlap, the
  system starts during user speech and `pause_reaction_penalty` grow a lot while turn ends are lost. If a
  microphone picks up a loudspeaker, use the recording after echo cancellation, one speaker per channel.
- A nearly silent channel: when one channel has no or almost no voice (a system that does not speak), even a weak leak
  of −40 dB is counted as speech of that channel, because the voice threshold follows whatever sound the
  channel has. A channel with only a little voice is also moved by a weaker leak (−30 dB) than a channel with
  more voice.
- Noise: stationary noise should be at least 25 dB below the speech (SNR). Around 20 dB the noise fills the
  silences: slightly more overlap, slightly fewer pauses and turn ends. At 15 dB and below weak speech is
  missed, and the number of turn ends, the no-response rate and the latency come out too large. Laughter,
  breaths and coughs are not told apart from speech at any level.
- Level: use recordings that are not clipped (clipping gives slightly more overlap and slightly fewer pauses
  and turn ends). When the speech is below −40 dBFS, the fixed lower limit of the voice threshold
  (`min_level_db`, −60 dBFS by default) cuts weak speech away and pauses and turn ends increase; lower the
  limit, e.g. `--set min_level_db=-84`.
- Sampling rate: 16 / 24 / 48 kHz give almost the same result.
- Length: checked up to 60 minutes. Memory use is about 6 times the size of the wav file. There is one
  voice threshold per recording and channel, so split a recording whose level changes a lot on the way.
- A loud sound in a short recording: in a recording of less than a minute, half a second of a sound much
  louder than the speech (a bang, howling) raises the threshold and speech is missed. In recordings of ten
  minutes and more the effect is slight.

`measure` estimates the quality of each recording per channel (`quality` and `recordings[].quality` in the
JSON, the "recording quality" table in the text) and puts a warning in `warnings` when a recording seems to be
outside the range above, with the direction in which the numbers are shifted. The measured values themselves
do not depend on the warnings.

| Warning | Raised when | Direction of the shift |
|---|---|---|
| noise (`noise`) | the speech level is less than 22 dB above the median level of the frames without voice (not raised at an SNR of 25 dB, raised at 20 dB) | fewer pauses and turn ends, more overlap; with stronger noise a higher no-response rate and longer latency |
| suspected leak (`leak`) | the leak level estimated from the cross-correlation of the channels while the other speaker is talking (normalised maximum 0.05 or more) is −27.5 dB or more; in a channel with little voice also below that, when the leaked voice comes within 5 dB of the channel's voice threshold | more overlap, more system starts during user speech and more voice in pauses, fewer turn ends |
| clipping (`clipping`) | 0.1 % or more of the samples are at the channel's peak value | slightly more overlap, slightly fewer pauses and turn ends |
| low level (`low_level`) | the voice threshold was set by the fixed lower limit (`min_level_db`) | more pauses and turn ends, less overlap; lowering the limit (`--set min_level_db=-84`) restores them |

- A channel with almost no voice (voice in less than 2 % of the frames) gets a note (`notes`), not a
  warning: the numbers are not necessarily shifted, but even a weak leak would change them.
- The warnings are guides based on estimates from the recording, with thresholds taken from the artificial
  degradations above. The leak estimate comes from the cross-correlation of the channels (within ±0.2 s) while
  the other speaker is talking, and can also be large when both speakers make similar sounds at the same time
  (laughing together). A recording without a warning is not guaranteed to be inside the range either
  (reverberation, non-stationary noise and coding artefacts are not estimated).
- The share of samples at the peak value looks at the waveform only, so test signals such as pure tones count
  as "clipped".
- The thresholds are in one place, `fdbja/recording_quality.py`, and are recorded in `quality.thresholds`.
  For a directory, `quality` holds the median, minimum and maximum over the recordings and the number of
  recordings with each warning (`quality.flags`).

Design, constraints found when validating on real recordings, and limits:
[docs/design/measure-entry.md](docs/design/measure-entry.md) (in Japanese).

#### Measuring per dialogue context (`fdbja measure-contexts`)

What counts as well-timed depends on the dialogue: the task, the scene, the relationship of the speakers, the
persona of the system. Human dialogues differ too; a chat and a task-oriented dialogue do not have the same
response latency or the same amount of overlap. `measure-contexts` measures one system in several contexts and
reports two things:

- per context, the values of the system, of the human recordings of the same context, and their difference
  (`system - human`);
- per pair of contexts, how the values moved from context A to context B (the shift) for the human recordings
  and for the system, with a verdict: **did the system move in the direction in which the human recordings
  moved?**

A system that times its speech the same way in every context can be close to the human recordings of one
context and still not change where the human recordings change; the second table makes that visible. Like
`measure`, it works on the audio alone: no transcript, no annotation, base dependencies only.

```bash
# 1. one reference profile per context, from human-human recordings of that context
uv run fdbja make-profile human/chat/ --language ja --relationship strangers \
    --situation chat --medium "video call" -o chat.json
uv run fdbja make-profile human/task/ --language ja --relationship strangers \
    --situation task --medium "video call" -o task.json
# 2. the recordings of the same system in each context, under a name of your choice (matched with the profile by name)
uv run fdbja measure-contexts chat=system/chat/ task=system/task/ \
    --profile chat=chat.json --profile task=task.json --factor situation --json contexts.json
```

```python
from fdbja import measure_contexts
c = measure_contexts({"chat": "system/chat/", "task": "system/task/"},
                     {"chat": "chat.json", "task": "task.json"}, factor="situation")
print(c)                                                # the same tables as the command
c.contexts["chat"]["features"]["latency_median"]        # {"system": ..., "human": ..., "gap": ...}
c.shifts[0]["features"]["latency_median"]["verdict"]    # "followed", ...
```

Example of the output (**synthetic signals, not measured values**; one context table and a few rows only):

```text
context: chat   system: 6 recording(s)   human: ja-strangers-chat-video_call@48a934d5 (6 recording(s))
                                         system                  human                   system - human
  response latency, median               0.40 s [0.35, 0.45]     0.30 s [0.25, 0.35]     +0.10 s [+0.03, +0.17] *
  turn ends without a response           0.0 % [0.0, 0.0]        0.0 % [0.0, 0.0]        +0.0 pt [+0.0, +0.0]

shift: chat -> task
                                         human                      system                     verdict
  response latency, median               +0.60 s [+0.52, +0.68] *   +0.40 s [+0.32, +0.47] *   followed (ratio +0.67)
  turn ends without a response           +0.0 pt [+0.0, +0.0]       +0.0 pt [+0.0, +0.0]       no clear human shift
```

- **The unit of analysis is the recording.** Every value is the mean over the recordings of a per-recording
  value. `[low, high]` is a 95 % bootstrap interval over recordings (2,000 draws by default; `--bootstrap`,
  `--seed`), so the events of one dialogue are not treated as independent observations. When fewer than 5
  recordings on one of the sides have a value, no interval is given and the shift is not assessed (5 is a rule
  of thumb, not a validated limit). `*` marks a difference whose interval excludes 0.
- Six values are reported; the first five are compared between contexts.

  | Name | Meaning |
  |---|---|
  | `latency_median` | median time from a turn end to the start of the system's speech (seconds) |
  | `no_response_rate` | share of turn ends without a response |
  | `starts_in_user_speech_per_minute` | system starts during user speech, per minute of user speech |
  | `overlap_share` | share of the recording in which both are voiced |
  | `pause_voiced_rate` | share of the pauses inside user turns with system voice |
  | `pause_reaction_penalty` | no-reaction rate + barge-in rate at pauses (per context only; no verdict between contexts) |

- One verdict per value and pair of contexts. "Shifts" means that the interval of the shift excludes 0.

  | Verdict | Meaning |
  |---|---|
  | `followed` | the human recordings shift and the system shifts in the same direction |
  | `opposite` | the human recordings shift and the system shifts in the other direction |
  | `no_clear_system_shift` | the human recordings shift; the interval of the system's shift contains 0 |
  | `system_only` | the interval of the human shift contains 0; the system shifts |
  | `no_clear_human_shift` | both intervals contain 0 |
  | `not_assessed` | a value is missing, fewer than 5 recordings, or the profile has no per-recording values |

  When the human recordings shift, `ratio` (system shift / human shift) is given as well: 1 = the same size,
  0 = no shift, negative = the other direction. With three or more contexts every pair is compared;
  `--baseline NAME` compares that context with each of the others only. The last lines count how many of the
  clear human shifts were `followed` (`summary` in the JSON).
- **What this does and does not tell you**:
  - The comparison is with the behaviour of the human recordings. **None of the values was compared with
    listeners' judgement.** `followed` says that the system moved where the human recordings moved, not that
    listeners judge the system natural in that context.
  - `pause_reaction_penalty` was compared with listeners' judgement on dialogues of one kind only (the
    evaluation set of this benchmark). Whether listeners' judgement depends on the context was not tested,
    so the value is reported per context without a verdict between contexts.
  - A shift between two contexts contains every difference between the two sets of recordings (users,
    topics, recording set-up). The system's shift describes the system alone only when the user's side is
    held fixed and nothing but the instruction to the system changes.
  - The intervals are per comparison and are not corrected for the number of comparisons; with many values
    and pairs some intervals exclude 0 by chance. An interval that contains 0 is not evidence of no shift
    (few recordings give wide intervals).
  - As with `measure`, the values are not comparable with the leaderboard, and the limits on the recording
    conditions above apply unchanged.
- The detection settings are the same for every context. Without `--set`, the settings the profiles were made
  with are used; profiles made with different settings are an error.
- The intervals of the human side come from the per-recording values stored in the profile
  (`reference_recordings`), a field added to `make-profile` together with this command. Profiles written
  before that, and `generic`, do not have it: the human values are then left out and a warning advises to
  rebuild the profile with `make-profile`. The field holds six numbers per recording, no file names and no
  audio.
- A context may have a profile only (shift of the human recordings alone) or recordings only (shift of the
  system alone).

Design and open questions: [docs/design/measure-contexts.md](docs/design/measure-contexts.md) (in Japanese).

### Outputs (`score-offline`, `analyze-profile`)

`score-offline` writes `results.json` (summary, scoring conditions and per-event decisions, including the event
times `t_start` / `t_end` and the system utterances near the judgement window, `sys_segments`), `summary.json`
(the same without the events; the shape of `docs/results.json`) and `summary.csv` (one row per task x system,
intervals split into `*_low` / `*_high`). The summary uses the keys of `docs/results.json`: `n` (events),
`n_clips`, `TOR`, `TOR_ci` (95% percentile bootstrap over events, 10000 replicates; `output.bootstrap: {n, seed}`; the intervals on the leaderboard use the same setting),
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
(talks over / does not react in pauses / stays silent / slow; thresholds are arguments). It also plots the Pause
barge-in rate against the Smooth TT no-response share (`profile.png` / `profile.svg`, needs matplotlib from
`.[analysis]`) and writes `events.csv`: per event, its time, the judgement window, the system's utterances, the
outcome and the reason.

On Pause the profile also reports the audio-based rule that ranks the leaderboard: `pause_no_reaction` and
`pause_barge_in` (the leaderboard rates), and `pause_brief`, the share of pauses with a reaction and no barge-in.
The tendency and the figure follow this rule, the one that was checked against listeners. The transcript-based
talked-through share (= TOR) also counts short reactions slightly over the backchannel limit, so it is larger
than the barge-in rate for systems that react briefly and stop; up to 0.2.0 such a system could be shown as
talking over. Results scored without the rule keep the transcript-based tendency; results scored by 0.2.0 follow
the rule and leave the split below empty.

The brief reactions are split by the transcript characters that start within 0.2 s of the reaction
(`metrics.pause_response.reaction_text_margin_seconds`): backchannel words or laughter only
(`brief_backchannel`), other words (`brief_other`), or no characters at all (`brief_no_text`: a breath, noise, or
speech the ASR missed). This split is a reference. It is not used for the ranking and was not checked against
listeners, and the ranking metric itself still does not tell backchannels, breaths and noise apart. It depends on
the ASR and on its character times: in our check, moving the margin between 0 and 0.3 s changed the split by more
than 10 points for some systems. With few brief reactions the shares are unstable (`profile.md` prints the
count). It covers reactions inside pauses only, not backchannels while the other speaker is talking.
`score-offline` stores the two character counts per Pause event (`reaction_chars`, `reaction_residual_chars`),
not the transcript text.

## What is evaluated

| Task | Event | Desired behaviour |
|------|-------|-------------------|
| Pause Handling | intra-turn pause (0.4-1.0 s), human B silent | stay silent (TO=0) |
| Smooth Turn-Taking | A (>=4 s) -> gap (<0.4 s) -> B (>=4 s) | take the turn (TO=1) |
| Backchanneling | human B only backchannels during A (>=3 s) | backchannel only (TO=0); also backchannel frequency (`bc_freq`, per second of A; human: `human_freq`) and the timing JSD against the human onsets in 0.2 s bins (`jsd`, 0-1, 1.0 when the model never backchannels; `bc_any_rate` = share of events with at least one model backchannel and `jsd_cond` = JSD over those events), as in the original Backchannel task (experimental; not in the default `tasks`) |

Ranking metrics, one per task: **TOR** (binary takeover rate, ASR-based, same definition as Full-Duplex-Bench)
for Smooth TT, and the **pause reaction penalty** (`pause_response`: the share of pauses with no reaction plus
the share with a barge-in, 0-2, lower is better, judged from the response audio) for Pause. The Pause TOR is
reported as a reference value only, because it pointed the opposite way to the listeners in the listening test.
Experimental metric (off by default): **VAP-distance** (KL divergence between next-speaker
distributions of a frozen VAP model given the human response vs. the model response); it has been
compared with human judgements in one listening test (2026-11) and did not agree with the listeners on
Pause, so label it as experimental when reporting. Trivial baselines (`silence`, `always_talk`) are reported on every
task. A TurnNat-style likelihood score is included for comparison. The `always_talk` speech loop is synthesised on first use with Open JTalk (voice "Mei", CC BY 3.0; attribution in [src/fdbja/assets/README.md](src/fdbja/assets/README.md)) and needs `pyopenjtalk`, or pass `source_wav`.

**What the metrics are for**: the turn-taking metrics measure whether a system falls outside the range
that people perceive as natural; individual style inside that range (how often it backchannels,
how long its pauses are) is not scored. For Smooth TT, "close to the recorded human B" held as the
centre of the range; for Pause it did not (in the human-judgement study, systems that produced a short
vocalisation in the pause were preferred), so the metric used for the Pause ranking measures whether the
system reacted briefly in the pause (a backchannel or not) and did not keep talking over the partner.
Backchannelling during the partner's speech is a separate task whose definition is published but not scored.

### The Pause ranking metric and the proposed metric

Two metrics that agreed better with listeners than TOR and the original VAP-distance in the human-judgement
study ([leaderboard, "human-judgement validation"](https://llm-jp.github.io/full-duplex-bench-ja/#validation-sec),
listening test, 2026-11) are implemented. Neither changes the values of TOR or the original VAP-distance; they
add columns to the `score-offline` summary (`summary.json` / `summary.csv`) and to the per-event results. The
two differ in how they are used and in how far they have been checked:

- The Pause metric (pause reaction penalty): from 0.2.0 it is **the ranking of the Pause task on the
  leaderboard** and is scored by default (`metrics.pause_response.enabled: false` switches it off; clips
  without response audio are skipped). After the listening test on which its composition was chosen (nine
  systems), its agreement with listeners was confirmed in **a second, pre-registered listening test** on the
  same events, using only pairs of systems that had not been shown in the first test (seven real systems).
  Some of the participants had also taken part in the first test (the analysis without them pointed the same
  way). The numbers of the second test are not published yet.
- Boundary-weighted VAP-distance: still a proposed metric, off by default. It was validated in the first
  listening test only, on whose data its composition was chosen.

For both, the validation is limited to the evaluation set of one corpus (LLM-jp-Zoom1) and has not been
repeated on another corpus. When reading the Pause ranking, note that:

- the reference point is 1.0, the value of a system that outputs nothing. A value above 1.0 means that the
  share of pauses with a barge-in is larger than the share of pauses with a reaction;
- any voice of at least 0.2 s counts as a reaction; backchannels, breaths and noise are not told apart;
- listeners preferred systems that reacted briefly in the pause and did not talk over the partner. This is
  not the behaviour of the recorded human B, who rarely speaks in the pauses of this evaluation set;
- the three commercial-API rows come from responses recorded before the adapters were fixed (items 4 to 6
  of the known limitations below; the code is fixed, the responses have not been recorded again). In those
  recordings interruption notices are ignored, so a response that the API would have cut short can be
  counted as a barge-in, and Gemini Live has only the first model turn of each clip. Responses of the
  commercial APIs are placed by arrival time, so a reaction in the pause is later by the network delay.

- **Boundary-weighted VAP-distance** (Smooth TT; `metrics.vap_distance.weighting: boundary`, needs
  `vap_distance.enabled: true`). Instead of the equal-weight mean of D_KL(P_t‖Q_t) over the judgement window, d_t is
  averaged with weights w_t = exp(−|t − t0| / τ) around the boundary t0 (end of A's utterance; `tau_seconds`,
  default 2.0 s), from 5 s before to 1.5 s after the judgement window (the stretch the listeners heard;
  `boundary_pre_seconds` / `boundary_post_seconds`). Summary columns: `vapd_boundary` (event mean, lower = closer to
  the human B) and `vapd_boundary_ci_*`. `boundary_tasks: [pause, smooth_tt]` also applies it to Pause (t0 = start
  of the gap), but only Smooth TT was selected in the study. The study used the VAP-Realtime checkpoint; values with
  the default MIT checkpoint have not been validated.
- **Pause response** (Pause; `metrics.pause_response`, on by default). For each gap [g_s, g_e]: no response = 1 when
  no voiced run has at least 0.2 s inside the gap; interrupt = 1 when a voiced run starts inside the gap and lasts at
  least 0.5 s after A resumes, or when the system overlaps A for more than 0.5 s in the 2 s before the gap; the metric
  is no response + interrupt (0-2, lower is better), averaged over events. Voice activity comes from the response
  audio (ch1 of `combined.wav`) and A (ch0 of the GT clip) with an energy detector (50 ms frames, 5 % of each
  channel's 99.5th-percentile energy); neither ASR nor VAP is used. Summary columns: `pause_response`
  (`pause_response_ci_*`), `pause_no_response_rate`, `pause_interrupt_rate`; the four thresholds are configurable.
  The recorded human B rarely speaks in these gaps (12 of the 100 events used in the listening test) while listeners preferred systems that reacted
  briefly, so the metric counts the reaction and the talking-over directly instead of the distance to the human B.
  Backchannels, breaths and noise are not told apart.

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
The commercial-API rows of the leaderboard were obtained with `gemini-3.1-flash-live-preview` (voice Aoede, 2026-04),
`gpt-realtime` (voice marin, 2026-07) and `grok-voice-latest` (voice eve, 2026-07), all with the servers' default
turn detection (the Gemini Live outputs come from the script the `gemini_live` adapter was ported from, with the same
settings). `gpt-realtime` and `grok-voice-latest` are aliases whose target changes with provider updates
(newer Grok Voice generations were released after our access), so the same IDs may no longer give the same models;
we do not guarantee to track API changes.

**Response timeline (`timeline`).** The streaming adapters (`moshi_ws`, `gemini_live`, `openai_realtime`) place
the received audio chunks on the time axis with shared rules (`src/fdbja/adapters/_timeline.py`), selected per
model entry in the config:

- `timeline: playout` (default): what a real client plays. A chunk starts when it arrives, or when the audio
  queued before it has finished playing, whichever is later (servers may send audio faster than real time).
  On an interruption notice (`server_content.interrupted` of Gemini Live, `input_audio_buffer.speech_started`
  of the OpenAI Realtime protocol) playback stops at the time the notice arrived and the audio not yet played
  is discarded; audio of the cut-off response that arrives later is not played. `openai_realtime` reports the
  played length to the server with `conversation.item.truncate` (`send_truncate: false` turns this off), and
  `gemini_live` keeps receiving until the input has been sent.
- `timeline: legacy`: the construction up to 0.2.0, for reproducing values published with 0.2.0. `moshi_ws`
  joins the received frames in order from time 0; `gemini_live` / `openai_realtime` place chunks by arrival
  time and ignore interruption notices; `gemini_live` stops receiving at the first `turn_complete`.

`infer` writes, next to `combined.wav`, the event record `events.json` (arrival time and sample count of every
chunk, time of every interruption notice and of other server events, and the time and reason listening
stopped, `client.receive_end`; no audio, no text) and `received.wav`
(the received audio joined in order). With these two files the timeline can be rebuilt under another rule
without calling the model again (`fdbja.adapters._timeline.timeline_from_record`). Custom adapters work without
returning `events` / `received_audio`.

`infer` skips clips that already have `combined.wav`, so a run that stopped continues with the same command. A clip
whose adapter call fails (network or API error) gets no output and one line in the log, and the run goes on; running
`infer` again records only those clips. A model is abandoned after 5 failed clips in a row (wrong API key, server
down), and `infer` exits with status 1 when any clip failed. `combined.wav` is written last, so an interrupted run
leaves no half-written clip that would count as done. Error messages in the log have anything that looks like a key
or token masked. `events.json` also holds the version of fdbja that wrote it (`fdbja`) and the time the clip was
started (`recorded_at`, UTC). Commercial APIs can change behaviour under an unchanged model name, so report the
day of recording with the results; `check-responses` lists both per system.

The leaderboard rows were recorded with the construction up to 0.2.0 and have not been recorded again; see
items 4 to 6 of the [known limitations](#known-limitations).

`gemini_live` and `openai_realtime` send the same instruction text by default (in Japanese: reply naturally in
Japanese, use backchannels such as "うん", "はい", "なるほど" where appropriate, reply in Japanese only). The
text has not changed since the first release (0.1.0). Because it asks for backchannels, the amount of
backchannels and brief reactions depends on it; nothing is sent to Moshi-type servers. Set `instructions` in the
model entry of the config to replace it, and treat results with another instruction as a separate condition. Both
adapters leave the model name, the voice and an identifier of the instruction (first 12 hex digits of its
SHA-256, not the text) at the start of `events.json` as the note `client.session`; `check-responses` lists them
per system and reports an error when they differ inside one system.

### Checking what `infer` recorded (`fdbja check-responses`)

Before scoring, `uv run fdbja check-responses --config configs/default.yaml --out results/response_check` checks
that the recordings of the configured systems can be compared. It does not call a model or an API and does not run
ASR. It replays each `events.json` under one set of rules and writes per-clip values (`clips.csv`), per-system
totals (`systems.csv`) and the findings (`response_check.md`).

| Finding | Level | Meaning |
|---------|-------|---------|
| `missing_output` | error | no `combined.wav` for the clip |
| `bad_output` | error | `combined.wav` cannot be read or has no model channel |
| `bad_record` | error | `events.json` cannot be read or has an unknown format |
| `received_mismatch` | error | `received.wav` does not have the length the record states |
| `rebuild_differs` | error | the timeline rebuilt from the record differs from `output.wav` (audio or record changed after recording) |
| `ended_early` | error | the client stopped listening before the input ended; later replies are not in the recording |
| `all_no_audio` | error | no clip of the task holds any model audio (check the endpoint and the credentials) |
| `mixed_rules` | error | clips of one system were recorded under different timeline rules |
| `mixed_settings` | error | clips of one system were recorded with different model names, voices or instructions (from the note `client.session` of the commercial-API adapters) |
| `response_failed` | warning | the server reported a failed or incomplete response |
| `stall` | warning | inside one response the audio arrived too late for playback for longer than `--stall-limit` (default 0.2 s); the timeline holds that much silence |
| `tail_cut` | warning | model audio beyond the written length was cut |
| `no_record` | warning | no `events.json`, nothing to check (recordings up to 0.2.0, hand-placed `combined.wav`, custom adapters without a record) |
| `mixed_versions` | warning | clips of one system were recorded with different versions of fdbja |
| `rules_differ` | warning | systems were recorded under different timeline rules, so response positions are not on one scale |

The totals also list the timeline rule, the number of interruption notices, the seconds of audio discarded at a
notice, the median time of the first audio and the number of clips without model audio. Notices and discarded audio
are behaviour of the system, not errors; a difference of an order of magnitude between systems is a reason to look
at the adapter settings. `--strict` exits with status 1 when there is any error (warnings alone give 0).
`--records-only` reads only `events.json`: an output counts as empty when no chunk was received, and `bad_output`,
`received_mismatch` and `rebuild_differs` are not checked. `--model` / `--task` restrict the check; the trivial baselines are skipped unless named with `--model`.
The end of listening comes from the note `client.receive_end` (with a reason) that the adapters leave in the record;
recordings without it are not judged for `ended_early`. The check finds problems of the recording. It does not
measure how good a response is, and a report without findings does not show that the scores are right.

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

## Known limitations

The binary TOR is decided on ASR transcripts and inherits their errors (details in the Japanese README,
「既知の限界」).

1. **Alignment delay**: the character alignment can place the start of an utterance late, so a response that
   overlaps the judgement window is missed (onset shift: median 0.14 s, 90th percentile 0.94 s).
2. **Missing reference audio**: short backchannels of human B can be removed by the silence gating of the
   per-speaker recording tracks and remain only in the transcript. This can also affect the reference of
   VAP-distance.
3. **ASR output on near-silent audio**: a short word transcribed over a long stretch falls outside the
   backchannel exemption (at most 6 characters and 1.5 s) and counts as a takeover.

Items 4 to 6 concern fairness across adapters. They are fixed in the code (`timeline: playout`, see
[Model adapters](#model-adapters)), but **the leaderboard values still come from the recordings made before
the fix**: the commercial-API responses have not been recorded again, and the recordings of the Moshi-family
systems do not contain frame arrival times.

4. **The commercial-API recordings ignore interruption notices.** Up to 0.2.0, `gemini_live` /
   `openai_realtime` did not handle the servers' notices (`server_content.interrupted`;
   `input_audio_buffer.speech_started`, after which the server ends the response with `response.done`,
   `status: cancelled`) and kept all received audio on the timeline. A real client stops playback on barge-in,
   so a response that the API would have cut short is scored as if it had continued (a short start in a
   Pause can become an utterance over 1.5 s, TO = 1).
5. **The response timeline was built differently per adapter.** Up to 0.2.0, the Moshi adapter joined the
   frames in arrival order from time 0, while the API adapters placed audio by arrival time, so the published
   response latencies (`lat_median`) are not strictly on the same scale across adapters.
6. **The Gemini Live recordings contain only the first model turn of each clip.** Up to 0.2.0, `gemini_live`
   stopped receiving at the first `turn_complete` (the SDK's `session.receive()` ends after each turn), so
   later responses in the same clip were not recorded. Counting stretches of audio separated by at least
   0.5 s of silence in the responses behind the published values, Gemini Live has at most one in all 400
   clips (one in 380, none in 20); with the same count, 300 clips of GPT-Realtime and 210 clips of Grok Voice
   have two or more. Events after the first turn are scored as having no response; the direction of the
   resulting bias is not known until the responses are recorded again. The clock of this adapter also
   started before the connection was opened, unlike the other adapters.

VAP-distance is computed from the waveform and is not affected by 1 and 3.

## Roadmap

1. Release of the evaluation data (tied to the public release of LLM-jp-Zoom1).
2. Score the system backchannel task (after confirming it with a listening test).
3. The two tasks where the system is already speaking (User Interruption of v1 and the user backchannel
   scenario of v1.5).

## License

Code: Apache-2.0. Backends and data are **not** covered:

- Default VAP checkpoint `vap_state_dict_jp_kyoto_20hz_2500msec.pt` (maai-kyoto/vap_jp_kyoto, revision `fe24ac6`): MIT.
  VAP-Realtime code: MIT. CPC encoder: MIT. With the default configuration there is no
  non-commercial restriction on the scores. The older VAP-Realtime Japanese checkpoint
  (`vap_state_dict_jp_20hz_2500msec.pt`) is academic-only and loads through the same code path;
  its terms then apply to the results. The leaderboard no longer uses it.
- ReazonSpeech code and models (nemo-v2 / k2-v2 / espnet-v2): Apache-2.0. NVIDIA NeMo: Apache-2.0.
  ESPnet and sherpa-onnx (dependencies of the `asr-reazon-whisperx` and `asr-k2` groups): Apache-2.0.
  WhisperX: BSD-2-Clause; its Japanese alignment model `jonatasgrosman/wav2vec2-large-xlsr-53-japanese`
  is Apache-2.0.
- Full-Duplex-Bench is CC BY-NC 4.0; this toolkit reimplements the event and TOR definitions
  independently and does not reuse its code.
- LLM-jp-Zoom1 and the excerpts in `docs/samples/` (three events; stimulus, human B and two models)
  follow the corpus terms (see DATASET.md). The model outputs kept there are from LLM-jp-Moshi-v1
  (Apache-2.0) and J-Moshi-ext (CC BY-NC 4.0); commercial-API outputs are not included.

## Citation

Please cite the ASJ Autumn 2026 talk (2-7-12): Yuto Abe, Mao Saeki, Shinnosuke Takamichi, Shinya Fujie,
Tetsuji Ogawa and Ryuichiro Higashinaka, "Full-Duplex-Bench-JA: 日本語フルデュプレックス音声対話モデル評価のための
ベンチマーク構築" (A benchmark for evaluating Japanese full-duplex spoken dialogue models), *Proceedings of the
Autumn Meeting of the Acoustical Society of Japan*, Sept. 2026. The same entry is in [CITATION.cff](CITATION.cff)
and as BibTeX in [README.md](README.md#引用).
