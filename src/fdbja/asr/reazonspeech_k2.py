"""ReazonSpeech k2 v2 backend (aligner-free, CPU-friendly).

``reazon-research/reazonspeech-k2-v2`` (Zipformer transducer via sherpa-onnx,
Japanese, Apache-2.0). The lightest install of the three backends (ONNX
runtime, no torch) at some cost in accuracy relative to the NeMo model.
Character times come from the same even split as the NeMo backend.

The upstream ``transcribe`` pads the waveform with ``PAD_SECONDS`` of silence
and reports token times on the padded axis; the offset is removed here.

Install: ``uv sync --extra asr-k2``.
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


class ReazonSpeechK2ASR(ASRModel):
    name = "reazonspeech_k2"

    def __init__(self, device: Optional[str] = None, precision: str = "fp32",
                 char_seconds: float = DEFAULT_CHAR_SECONDS,
                 bridge_seconds: float = DEFAULT_BRIDGE_SECONDS):
        self._rs = importlib.import_module("reazonspeech.k2.asr")
        # sherpa-onnx provider: "cpu" (default), "cuda" needs onnxruntime-gpu
        self.model = self._rs.load_model(device=device or "cpu", precision=precision)
        self.char_seconds = char_seconds
        self.bridge_seconds = bridge_seconds
        self.pad_seconds = float(getattr(
            importlib.import_module("reazonspeech.k2.asr.transcribe"), "PAD_SECONDS", 0.0))

    def transcribe(self, audio: np.ndarray, sample_rate: int,
                   speaker: str = "B") -> List[Dict[str, Any]]:
        rs = self._rs
        wav = np.asarray(audio, dtype=np.float32)
        ret = rs.transcribe(self.model, rs.audio_from_numpy(wav, sample_rate),
                            rs.TranscribeConfig(verbose=False))
        items = subwords_to_chars(((s.token, s.seconds - self.pad_seconds)
                                   for s in ret.subwords),
                                  speaker=speaker, char_seconds=self.char_seconds,
                                  bridge_seconds=self.bridge_seconds,
                                  audio_duration=len(wav) / sample_rate)
        return validate_jchat(items)
