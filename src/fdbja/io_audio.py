"""Audio I/O helpers."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import soundfile as sf

VAP_SR = 16000


def load_stereo_16k(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a stereo wav, resample to 16 kHz, return (left, right) float32.

    Left is speaker A, right is speaker B (the convention used by the GT clips
    and the model ``combined.wav`` outputs).
    """
    data, sr = sf.read(path, dtype="float32")  # (T, C)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f"expected stereo wav: {path} has shape {data.shape}")
    if sr != VAP_SR:
        import torch
        import torchaudio.functional as AF
        wav = AF.resample(torch.from_numpy(data.T), sr, VAP_SR).numpy()
        return wav[0], wav[1]
    return data[:, 0], data[:, 1]
