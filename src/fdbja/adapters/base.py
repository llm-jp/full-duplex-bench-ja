"""Model adapter interface.

Unlike text LLM evaluation, full-duplex evaluation must *stream* speaker A's
audio to a live model in real time and record the model's response audio.
Each backend (a Moshi-style WebSocket server, the Gemini Live API, ...)
implements this interface so the rest of the toolkit stays model-agnostic.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np


@dataclass
class DialogueResponse:
    """Model output for one input clip.

    Attributes:
        input_audio:  speaker A audio actually sent (mono, ``sample_rate``)
        response_audio: model's speaker-B response audio (mono, ``sample_rate``)
        sample_rate:  sample rate of both arrays
    The toolkit builds the stereo (A, model-B) signal for VAP-distance from
    these, and ASR-transcribes ``response_audio`` for the binary TOR metric.
    """
    input_audio: np.ndarray
    response_audio: np.ndarray
    sample_rate: int


class ModelAdapter(abc.ABC):
    """Stream a clip of speaker-A audio to a model and capture its response."""

    name: str = "base"

    @abc.abstractmethod
    def infer(self, input_audio: np.ndarray, sample_rate: int) -> DialogueResponse:
        """Send ``input_audio`` (mono speaker A) to the model in real time and
        return the recorded response. Implementations are responsible for the
        streaming protocol, real-time pacing, and resampling."""
        raise NotImplementedError

    def close(self) -> None:  # optional cleanup hook
        pass
