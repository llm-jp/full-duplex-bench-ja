"""Energy-based stand-in for ASR, for the synthetic corpus only.

Emits one placeholder character per 0.1 s of frames whose RMS exceeds a
threshold. It is **not** a speech recogniser: it only lets ``transcribe`` run
on the synthetic corpus without downloading a speech model. It also shows how
to plug in your own backend with ``fdbja.asr.register_backend``.

Usage (same arguments as ``fdbja transcribe``)::

    uv run python examples/dummy/energy_asr.py --config examples/dummy/config.yaml
"""

from __future__ import annotations

import sys

import numpy as np

from fdbja.asr import ASRModel, register_backend


class EnergyASR(ASRModel):
    name = "energy"

    def __init__(self, threshold: float = 0.005, frame_s: float = 0.02,
                 char_s: float = 0.1, device=None):
        self.threshold = float(threshold)
        self.frame_s = float(frame_s)
        self.char_s = float(char_s)

    def transcribe(self, audio, sample_rate, speaker="B"):
        x = np.asarray(audio, dtype=np.float32)
        hop = max(1, int(self.frame_s * sample_rate))
        n = len(x) // hop
        if n == 0:
            return []
        rms = np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(axis=1))
        active = rms > self.threshold
        items, t, end_t = [], None, None
        for i, a in enumerate(np.append(active, False)):
            if a and t is None:
                t = i * self.frame_s
            elif not a and t is not None:
                end_t = i * self.frame_s
                s = t
                while s + self.char_s <= end_t + 1e-9:
                    items.append({"speaker": speaker, "word": "ー",
                                  "start": round(s, 3),
                                  "end": round(s + self.char_s * 0.8, 3)})
                    s += self.char_s
                t = None
        return items


register_backend("energy", EnergyASR)


if __name__ == "__main__":
    from fdbja.cli import main
    raise SystemExit(main(["transcribe", *sys.argv[1:]]))
