"""VAP backend interface for the VAP-distance metric.

A VAP backend turns a stereo (A, B) waveform into a per-frame sequence of
256-class voice-activity-projection probability distributions. The reference
backend wraps inokoj/VAP-Realtime with a frozen Japanese checkpoint
(20 Hz, 2.5 s context); any model producing the same distribution can be
substituted.
"""

from __future__ import annotations

import abc
from typing import Tuple

import numpy as np


class VAPModel(abc.ABC):
    """Frozen turn-taking predictor used as an evaluation reference."""

    frame_rate: int = 20
    sample_rate: int = 16000

    @abc.abstractmethod
    def probs(self, left: np.ndarray, right: np.ndarray
              ) -> Tuple[np.ndarray, np.ndarray]:
        """Run VAP over a stereo pair (both mono, ``sample_rate``).

        Returns ``(times, probs)`` where ``times`` is ``(T,)`` frame centers in
        seconds and ``probs`` is ``(T, 256)`` softmax distributions.
        """
        raise NotImplementedError
