"""A minimal custom ModelAdapter: reply after the partner has been silent.

The "model" listens to speaker A frame by frame (20 ms) and, once A has been
silent for ``wait_s`` seconds after speaking, plays a fixed reply. It reads
only past frames, as a live system would. It is a template for wiring your
own model, not a system worth reporting.

The reply is ``reply_wav`` (mono, any sample rate) when given; otherwise the
last ``reply_s`` seconds of A's voiced audio, so the example needs no audio
file and the ASR still finds speech in it.

Use it from a config in one of two ways:

1. By import path (the module must be importable)::

       models:
         - name: reply_after_silence
           adapter: "reply_after_silence:ReplyAfterSilence"
           wait_s: 0.6
           domain: out

       PYTHONPATH=examples/adapters uv run fdbja infer --config ...

2. By name, after ``fdbja.adapters.register_adapter`` (this file registers
   ``reply_after_silence`` and runs ``fdbja`` when executed)::

       uv run python examples/adapters/reply_after_silence.py infer --config ...

``infer`` writes ``<task>_<model>_output/<clip>/combined.wav`` (ch0 = A,
ch1 = the reply); ``transcribe`` and ``score-offline`` then work as for any
other model.
"""

from __future__ import annotations

import sys

import numpy as np

from fdbja.adapters import DialogueResponse, ModelAdapter, register_adapter


class ReplyAfterSilence(ModelAdapter):
    name = "reply_after_silence"

    def __init__(self, wait_s: float = 0.6, reply_s: float = 2.0,
                 reply_wav: str = None, threshold: float = 0.005,
                 frame_s: float = 0.02, level: float = 0.05):
        self.wait_s = float(wait_s)
        self.reply_s = float(reply_s)
        self.reply_wav = reply_wav
        self.threshold = float(threshold)
        self.frame_s = float(frame_s)
        self.level = float(level)

    def _fixed_reply(self, sample_rate: int):
        if not self.reply_wav:
            return None
        import soundfile as sf
        y, sr = sf.read(self.reply_wav, dtype="float32")
        if y.ndim == 2:
            y = y.mean(axis=1)
        if sr != sample_rate:   # linear resampling is enough for a template
            t = np.arange(int(len(y) * sample_rate / sr)) / sample_rate
            y = np.interp(t, np.arange(len(y)) / sr, y).astype(np.float32)
        return y

    def infer(self, input_audio: np.ndarray, sample_rate: int) -> DialogueResponse:
        x = np.asarray(input_audio, dtype=np.float32)
        out = np.zeros_like(x)
        hop = max(1, int(self.frame_s * sample_rate))
        fixed = self._fixed_reply(sample_rate)
        voiced_hist = []            # voiced frames of A heard so far
        silent_run = 0.0
        heard_since_reply = False
        busy_until = 0              # sample index where the current reply ends
        for i in range(len(x) // hop):
            frame = x[i * hop:(i + 1) * hop]
            voiced = float(np.sqrt((frame.astype(np.float64) ** 2).mean())) > self.threshold
            if voiced:
                voiced_hist.append(frame)
                silent_run = 0.0
                heard_since_reply = True
                continue
            silent_run += self.frame_s
            now = (i + 1) * hop
            if heard_since_reply and silent_run >= self.wait_s and now >= busy_until:
                if fixed is not None:
                    reply = fixed
                else:
                    n = int(self.reply_s / self.frame_s)
                    reply = np.concatenate(voiced_hist[-n:])
                reply = reply[:len(x) - now]
                peak = float(np.abs(reply).max()) if len(reply) else 0.0
                if peak > 0:
                    out[now:now + len(reply)] = reply * (self.level / peak)
                busy_until = now + len(reply)
                heard_since_reply = False
        return DialogueResponse(input_audio=input_audio, response_audio=out,
                                sample_rate=sample_rate)


register_adapter("reply_after_silence", ReplyAfterSilence)


if __name__ == "__main__":
    from fdbja.cli import main
    raise SystemExit(main(sys.argv[1:]))
