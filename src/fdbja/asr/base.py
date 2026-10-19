"""ASR backend interface and jchat helpers.

The binary TOR metric needs the model's response audio transcribed to
character-level time-aligned tokens (jchat format) so utterances can be
grouped and classified. A jchat transcript is a list of

    {"speaker": "B", "word": "<one character>", "start": <sec>, "end": <sec>}

sorted by ``start``. Two families of backends produce it:

* ASR + aligner (the reference pipeline used for the published numbers:
  ReazonSpeech-ESPnet segments re-aligned to characters with WhisperX);
* ASR with token-level timestamps (ReazonSpeech NeMo / k2), where each
  subword's time span is split evenly over its characters
  (see :mod:`fdbja.asr._subword`).
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List

import numpy as np

JCHAT_KEYS = ("speaker", "word", "start", "end")
ASR_SAMPLE_RATE = 16000


class ASRModel(abc.ABC):
    """Transcribe one mono response channel into jchat items."""

    #: registry name, set by subclasses
    name: str = ""

    @abc.abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int,
                   speaker: str = "B") -> List[Dict[str, Any]]:
        """Return character-level jchat items for ``audio`` (1-D float array)."""
        raise NotImplementedError


def validate_jchat(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Check jchat items and return them sorted by ``start``.

    Raises ``ValueError`` on a missing key, a non-string ``word``, a negative
    time or ``end < start``. Zero-length items are allowed (an aligner may
    collapse a character onto a single frame).
    """
    for i, it in enumerate(items):
        missing = [k for k in JCHAT_KEYS if k not in it]
        if missing:
            raise ValueError(f"jchat item {i} lacks {missing}: {it!r}")
        if not isinstance(it["word"], str) or not it["word"]:
            raise ValueError(f"jchat item {i} has an empty/non-string word: {it!r}")
        s, e = float(it["start"]), float(it["end"])
        if s < 0 or e < s:
            raise ValueError(f"jchat item {i} has a bad time span [{s}, {e}]: {it!r}")
    return sorted(items, key=lambda x: (float(x["start"]), float(x["end"])))


def to_mono_16k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Return ``audio`` as mono float32 at 16 kHz (what the aligners expect)."""
    wav = np.asarray(audio, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=1) if wav.shape[1] < wav.shape[0] else wav.mean(axis=0)
    if sample_rate == ASR_SAMPLE_RATE:
        return np.ascontiguousarray(wav)
    try:
        import torch
        import torchaudio.functional as AF
        return AF.resample(torch.from_numpy(wav), sample_rate, ASR_SAMPLE_RATE).numpy()
    except ImportError:
        import librosa
        return librosa.resample(wav, orig_sr=sample_rate, target_sr=ASR_SAMPLE_RATE)


#: Default silence gate applied before ASR (see :func:`silence_gate`).
DEFAULT_GATE = {"min_rms": 1e-4, "drop_unvoiced_items": False,
                "frame_s": 0.05, "rel_threshold": 0.05}


def frame_rms(audio: np.ndarray, sample_rate: int, frame_s: float = 0.05) -> np.ndarray:
    """Per-frame RMS of a mono signal (non-overlapping frames of ``frame_s``)."""
    x = np.asarray(audio, dtype=np.float64)
    hop = max(1, int(round(frame_s * sample_rate)))
    n = len(x) // hop
    if n == 0:
        return np.zeros(0)
    return np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(axis=1))


def silence_gate(audio: np.ndarray, sample_rate: int, items_fn,
                 min_rms: float = 1e-4, drop_unvoiced_items: bool = False,
                 frame_s: float = 0.05, rel_threshold: float = 0.05
                 ) -> List[Dict[str, Any]]:
    """Run ``items_fn()`` (the ASR call) only where there is signal.

    Speech recognisers hallucinate on silence: reazonspeech_whisperx returns
    "はい。" for an all-zero channel and WhisperX stretches it over the whole
    clip, which made the always-silent baseline take the floor on every pause
    (held-out set, found 2026-10-18).

    * If no frame of the channel reaches ``min_rms`` (1e-4 = -80 dBFS; digital
      silence and dither), the ASR is not called and ``[]`` is returned.
    * With ``drop_unvoiced_items``, characters whose ``[start, end]`` span
      contains no frame above ``rel_threshold`` x the channel's 99.5th
      percentile frame RMS are dropped (hallucinations inside long silences of
      a real response). Off by default: it changes the transcripts of real
      systems, so published numbers were computed without it.
    """
    rms = frame_rms(audio, sample_rate, frame_s)
    if len(rms) == 0 or float(rms.max()) < min_rms:
        return []
    items = items_fn()
    if not drop_unvoiced_items or not items:
        return items
    active = rms > max(rel_threshold * float(np.percentile(rms, 99.5)), min_rms)
    kept = []
    for it in items:
        lo = int(float(it["start"]) / frame_s)
        hi = max(lo + 1, int(np.ceil(float(it["end"]) / frame_s)))
        if active[max(0, lo):min(len(active), hi)].any():
            kept.append(it)
    return kept
