"""Degenerate reference systems ("null models").

Both metrics in this toolkit have a degenerate solution that no real system
should be allowed to win with:

* ``silence`` never speaks. Binary TOR is then 0 everywhere, which is the
  *desired* behaviour for pause handling and backchanneling, and VAP-distance
  against a human who also stayed silent is near zero. A leaderboard without
  this row cannot tell "correctly held the floor" from "produced nothing".
* ``always_talk`` speaks for the whole clip, the opposite failure mode
  (TOR = 1, which is desired for smooth turn taking). It must be *speech*:
  TOR reads an ASR transcript, and noise is not transcribed.

Reporting both bounds every task is what makes a model's number meaningful:
a system must beat the degenerate reference on *every* task, not just the one
whose desired behaviour it happens to match.
"""

from __future__ import annotations

import numpy as np

from fdbja.adapters.base import DialogueResponse, ModelAdapter


class SilenceAdapter(ModelAdapter):
    """Returns digital silence of the same length as the input clip."""

    name = "silence"

    def infer(self, input_audio: np.ndarray, sample_rate: int) -> DialogueResponse:
        return DialogueResponse(
            input_audio=input_audio,
            response_audio=np.zeros_like(np.asarray(input_audio, dtype=np.float32)),
            sample_rate=sample_rate,
        )


class AlwaysTalkAdapter(ModelAdapter):
    """Returns continuous *speech* for the whole clip: the always-interrupting bound.

    Binary TOR reads the ASR transcript of the response channel, so the signal
    has to be something a speech recogniser transcribes. Band-limited noise
    (the previous implementation) is not: reazonspeech_whisperx returns almost
    nothing for it, and the baseline then scored Pause TOR 0.025 instead of
    ~1.0 on the held-out set (found 2026-10-18). The published v4_eval numbers
    used a looped speech recording, which ``source_wav`` reproduces.

    ``source`` selects the signal:

    * ``"auto"`` (default): ``source_wav`` if given, else ``"input_loop"``.
    * ``"wav"``: loop the mono speech file ``source_wav`` over the clip.
    * ``"input_loop"``: concatenate the voiced frames of the input clip
      (speaker A) and loop them over the clip, so there is real speech without
      shipping any audio. Its content echoes A, which does not matter for
      either metric (both read when B is active, not what B says).
    * ``"noise"``: the old band-limited noise. Only meaningful with an
      energy-based stand-in ASR (``examples/dummy``); do not report it.

    Only voiced frames are looped, so the response has no pauses; it is then
    RMS-normalised to ``level`` (0.035 ~ the v4_eval loop, 0.0345).
    """

    name = "always_talk"
    SOURCES = ("auto", "wav", "input_loop", "noise")

    def __init__(self, level: float = 0.035, seed: int = 0, source: str = "auto",
                 source_wav: str = None, frame_s: float = 0.05,
                 voiced_rel: float = 0.1):
        if source not in self.SOURCES:
            raise ValueError(f"source must be one of {self.SOURCES}: {source!r}")
        if source == "wav" and not source_wav:
            raise ValueError("source='wav' needs source_wav")
        self.level = float(level)
        self.seed = int(seed)
        self.source = ("wav" if source_wav else "input_loop") if source == "auto" else source
        self.source_wav = source_wav
        self.frame_s = float(frame_s)
        self.voiced_rel = float(voiced_rel)

    def _voiced(self, x: np.ndarray, sample_rate: int) -> np.ndarray:
        """Samples of the frames whose RMS exceeds ``voiced_rel`` x the 99th percentile."""
        hop = max(1, int(round(self.frame_s * sample_rate)))
        n = len(x) // hop
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        frames = x[:n * hop].reshape(n, hop)
        rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))
        ref = np.percentile(rms, 99)
        if ref <= 0:
            return np.zeros(0, dtype=np.float32)
        return frames[rms > self.voiced_rel * ref].reshape(-1).astype(np.float32)

    def _loop_source(self, sample_rate: int, input_audio: np.ndarray) -> np.ndarray:
        if self.source == "wav":
            import soundfile as sf
            data, sr = sf.read(self.source_wav, dtype="float32")
            if data.ndim == 2:
                data = data.mean(axis=1)
            if sr != sample_rate:
                import torch
                import torchaudio.functional as AF
                data = AF.resample(torch.from_numpy(data), sr, sample_rate).numpy()
            return self._voiced(data, sample_rate)
        x = np.asarray(input_audio, dtype=np.float32)
        if x.ndim == 2:
            x = x.mean(axis=1)
        return self._voiced(x, sample_rate)

    def _noise(self, n: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        noise = rng.normal(0.0, self.level, n).astype(np.float32)
        alpha = 0.9   # 1-pole low-pass, roughly speech-band
        for i in range(1, n):
            noise[i] = alpha * noise[i - 1] + (1 - alpha) * noise[i]
        return noise

    def infer(self, input_audio: np.ndarray, sample_rate: int) -> DialogueResponse:
        n = len(np.asarray(input_audio))
        if self.source == "noise":
            out = self._noise(n)
        else:
            seg = self._loop_source(sample_rate, input_audio)
            if len(seg) == 0:
                raise ValueError(
                    "always_talk: no voiced frames in the loop source "
                    f"({self.source}); give source_wav with speech")
            out = np.resize(seg, n).astype(np.float32)   # tile, then cut to n
            rms = float(np.sqrt((out.astype(np.float64) ** 2).mean()))
            if rms > 0:
                out *= self.level / rms
            np.clip(out, -1.0, 1.0, out=out)
        return DialogueResponse(
            input_audio=input_audio,
            response_audio=out,
            sample_rate=sample_rate,
        )
