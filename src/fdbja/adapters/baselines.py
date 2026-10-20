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

    * ``"auto"`` (default): ``source_wav`` if given, else ``"bundled"``.
    * ``"bundled"``: the 60 s Open JTalk loop (voice "Mei", CC BY 3.0; see
      ``fdbja/assets/README.md``), synthesised on first use and cached
      (:mod:`fdbja.adapters.always_talk_loop`). Needs ``pyopenjtalk``; without
      it the adapter stops and asks for ``source_wav``.
    * ``"wav"``: loop the mono speech file ``source_wav`` over the clip.
    * ``"input_loop"``: concatenate the voiced frames of the input clip
      (speaker A) and loop them over the clip, so there is real speech without
      shipping any audio. Its content echoes A, which does not matter for
      either metric (both read when B is active, not what B says).
    * ``"noise"``: the old band-limited noise. Only meaningful with an
      energy-based stand-in ASR (``examples/dummy``); do not report it.

    ``input_loop`` is kept for comparison only: with reazonspeech_whisperx its
    seams make the ASR split the loop, so smooth_tt counts a fresh B onset
    after A's turn end (TOR 0.20-0.40 on 10 held-out clips, any ``join``,
    vs 0.00 for a fixed speech loop; 2026-10-19).

    Only voiced frames are looped, so the response has no pauses; it is then
    RMS-normalised to ``level`` (0.035 ~ the v4_eval loop, 0.0345).
    """

    name = "always_talk"
    SOURCES = ("auto", "bundled", "wav", "input_loop", "noise")
    #: how voiced runs are joined into the loop (2026-10-19 trial: "concat" made the ASR split the
    #: loop at its seams, so smooth_tt counted a fresh B onset after A's turn end; TOR 0.40)
    #:   concat    plain concatenation, plain tiling
    #:   crossfade linear crossfade of ``fade_ms`` at every seam (between runs and between loops)
    #:   longest   only the longest voiced run, ends faded over ``fade_ms``, plain tiling
    JOINS = ("concat", "crossfade", "longest")

    def __init__(self, level: float = 0.035, seed: int = 0, source: str = "auto",
                 source_wav: str = None, frame_s: float = 0.05,
                 voiced_rel: float = 0.1, join: str = "concat", fade_ms: float = 30.0):
        if source not in self.SOURCES:
            raise ValueError(f"source must be one of {self.SOURCES}: {source!r}")
        if source == "wav" and not source_wav:
            raise ValueError("source='wav' needs source_wav")
        if join not in self.JOINS:
            raise ValueError(f"join must be one of {self.JOINS}: {join!r}")
        self.level = float(level)
        self.seed = int(seed)
        if source == "auto":
            source = "wav" if source_wav else "bundled"
        if source == "bundled":
            from fdbja.adapters.always_talk_loop import ensure_loop
            source, source_wav = "wav", ensure_loop()
        self.source = source
        self.source_wav = source_wav
        self.frame_s = float(frame_s)
        self.voiced_rel = float(voiced_rel)
        self.join = join
        self.fade_ms = float(fade_ms)

    def _voiced_runs(self, x: np.ndarray, sample_rate: int) -> list:
        """Contiguous runs of frames whose RMS exceeds ``voiced_rel`` x the 99th percentile."""
        hop = max(1, int(round(self.frame_s * sample_rate)))
        n = len(x) // hop
        if n == 0:
            return []
        frames = x[:n * hop].reshape(n, hop)
        rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))
        ref = np.percentile(rms, 99)
        if ref <= 0:
            return []
        on = rms > self.voiced_rel * ref
        runs, i = [], 0
        while i < n:
            if not on[i]:
                i += 1
                continue
            j = i
            while j < n and on[j]:
                j += 1
            runs.append(frames[i:j].reshape(-1).astype(np.float32))
            i = j
        return runs

    def _voiced(self, x: np.ndarray, sample_rate: int) -> np.ndarray:
        runs = self._voiced_runs(x, sample_rate)
        return np.concatenate(runs) if runs else np.zeros(0, dtype=np.float32)

    @staticmethod
    def _xfade_join(segs: list, nf: int, n_out: int) -> np.ndarray:
        """Join ``segs`` cyclically with an ``nf``-sample linear crossfade until ``n_out`` samples."""
        out = segs[0].copy()
        k = 1
        while len(out) < n_out:
            s = segs[k % len(segs)]
            k += 1
            f = min(nf, len(out), len(s))
            if f > 0:
                w = np.linspace(0.0, 1.0, f, dtype=np.float32)
                out[-f:] = out[-f:] * (1 - w) + s[:f] * w
            out = np.concatenate([out, s[f:]])
        return out[:n_out]

    def _loop_runs(self, sample_rate: int, input_audio: np.ndarray) -> list:
        if self.source == "wav":
            import soundfile as sf
            data, sr = sf.read(self.source_wav, dtype="float32")
            if data.ndim == 2:
                data = data.mean(axis=1)
            if sr != sample_rate:
                import torch
                import torchaudio.functional as AF
                data = AF.resample(torch.from_numpy(data), sr, sample_rate).numpy()
            return self._voiced_runs(data, sample_rate)
        x = np.asarray(input_audio, dtype=np.float32)
        if x.ndim == 2:
            x = x.mean(axis=1)
        return self._voiced_runs(x, sample_rate)

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
            runs = self._loop_runs(sample_rate, input_audio)
            if not runs:
                raise ValueError(
                    "always_talk: no voiced frames in the loop source "
                    f"({self.source}); give source_wav with speech")
            nf = int(round(self.fade_ms / 1000.0 * sample_rate))
            if self.join == "crossfade":
                out = self._xfade_join(runs, nf, n).astype(np.float32)
            elif self.join == "longest":
                seg = max(runs, key=len).copy()
                f = min(nf, len(seg) // 2)
                if f > 0:
                    w = np.linspace(0.0, 1.0, f, dtype=np.float32)
                    seg[:f] *= w
                    seg[-f:] *= w[::-1]
                out = np.resize(seg, n).astype(np.float32)
            else:
                out = np.resize(np.concatenate(runs), n).astype(np.float32)   # tile, cut to n
            rms = float(np.sqrt((out.astype(np.float64) ** 2).mean()))
            if rms > 0:
                out *= self.level / rms
            np.clip(out, -1.0, 1.0, out=out)
        return DialogueResponse(
            input_audio=input_audio,
            response_audio=out,
            sample_rate=sample_rate,
        )
