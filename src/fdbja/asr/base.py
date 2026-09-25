"""ASR backend interface.

The binary TOR metric needs the model's response audio transcribed to
character-level time-aligned tokens (jchat format) so utterances can be
grouped and classified. The reference pipeline is ReazonSpeech-ESPnet +
WhisperX alignment; any aligner producing ``{"speaker","word","start","end"}``
items works.
"""

from __future__ import annotations

import abc
from typing import List, Dict, Any

import numpy as np


class ASRModel(abc.ABC):
    @abc.abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int,
                   speaker: str = "B") -> List[Dict[str, Any]]:
        """Return character-level jchat items for ``audio``."""
        raise NotImplementedError
