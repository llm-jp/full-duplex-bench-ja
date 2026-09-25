"""Degenerate reference systems ("null models").

Both metrics in this toolkit have a degenerate solution that no real system
should be allowed to win with:

* ``silence`` never speaks. Binary TOR is then 0 everywhere, which is the
  *desired* behaviour for pause handling and backchanneling, and VAP-distance
  against a human who also stayed silent is near zero. A leaderboard without
  this row cannot tell "correctly held the floor" from "produced nothing".
* ``always_talk`` speaks for the whole clip, the opposite failure mode
  (TOR = 1, which is desired for smooth turn taking).

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
    """Returns continuous low-level noise: the always-interrupting bound.

    Speech-like content is irrelevant to both metrics (TOR reads the ASR
    transcript, VAP-distance reads voice activity), so band-limited noise at a
    conversational level is enough to make every frame "active".
    """

    name = "always_talk"

    def __init__(self, level: float = 0.05, seed: int = 0):
        self.level = float(level)
        self.seed = int(seed)

    def infer(self, input_audio: np.ndarray, sample_rate: int) -> DialogueResponse:
        rng = np.random.default_rng(self.seed)
        n = len(np.asarray(input_audio))
        noise = rng.normal(0.0, self.level, n).astype(np.float32)
        # 1-pole low-pass, roughly speech-band, so VAD/VAP see a voiced-like level
        alpha = 0.9
        for i in range(1, n):
            noise[i] = alpha * noise[i - 1] + (1 - alpha) * noise[i]
        return DialogueResponse(
            input_audio=input_audio,
            response_audio=noise,
            sample_rate=sample_rate,
        )
