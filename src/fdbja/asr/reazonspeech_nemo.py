"""ReazonSpeech NeMo v2 backend (aligner-free).

``reazon-research/reazonspeech-nemo-v2`` (FastConformer-RNNT, Japanese,
Apache-2.0) returns subword tokens with emission timestamps, so no separate
aligner is needed. Character times are derived by splitting each subword's
span evenly (:func:`fdbja.asr._subword.subwords_to_chars`).

Install: ``uv sync --extra asr-nemo`` (pulls ``nemo_toolkit[asr]``; the model
weights are downloaded from the Hugging Face Hub on first use).
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional

import numpy as np

from fdbja.asr._subword import (
    DEFAULT_BRIDGE_SECONDS,
    DEFAULT_CHAR_SECONDS,
    subwords_to_chars,
)
from fdbja.asr.base import ASRModel, validate_jchat


class ReazonSpeechNeMoASR(ASRModel):
    name = "reazonspeech_nemo"

    def __init__(self, device: Optional[str] = None,
                 char_seconds: float = DEFAULT_CHAR_SECONDS,
                 bridge_seconds: float = DEFAULT_BRIDGE_SECONDS, verbose: bool = False):
        self._rs = importlib.import_module("reazonspeech.nemo.asr")
        self.model = self._rs.load_model(device=device)
        self.char_seconds = char_seconds
        self.bridge_seconds = bridge_seconds
        self.verbose = verbose

    def transcribe(self, audio: np.ndarray, sample_rate: int,
                   speaker: str = "B") -> List[Dict[str, Any]]:
        rs = self._rs
        wav = np.asarray(audio, dtype=np.float32)
        ret = rs.transcribe(self.model, rs.audio_from_numpy(wav, sample_rate),
                            rs.TranscribeConfig(verbose=self.verbose))
        items = subwords_to_chars(((s.token, s.seconds) for s in ret.subwords),
                                  speaker=speaker, char_seconds=self.char_seconds,
                                  bridge_seconds=self.bridge_seconds,
                                  audio_duration=len(wav) / sample_rate)
        return validate_jchat(items)
