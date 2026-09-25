"""ASR backends that turn a model's response audio into jchat transcripts.

Select one in ``configs/default.yaml``::

    asr:
      backend: reazonspeech_whisperx   # pipeline of the published numbers
      options: {}                      # passed to the backend constructor

``reazonspeech_nemo`` / ``reazonspeech_k2`` need no aligner and run several
times faster, but segment utterances differently, so TOR values are not
directly comparable with the leaderboard.

and run ``fdbja transcribe``. Every backend needs its own extra
(``uv sync --extra asr-nemo`` etc.); the core package stays dependency-light.
"""

from __future__ import annotations

import importlib
from typing import Callable, Dict, List, Tuple, Union

from fdbja.asr.base import ASRModel, validate_jchat

__all__ = ["ASRModel", "BACKENDS", "DEFAULT_BACKEND", "load_asr",
           "register_backend", "validate_jchat"]

#: name -> (module, class) or a factory callable
BACKENDS: Dict[str, Union[Tuple[str, str], Callable[..., ASRModel]]] = {
    "reazonspeech_nemo": ("fdbja.asr.reazonspeech_nemo", "ReazonSpeechNeMoASR"),
    "reazonspeech_k2": ("fdbja.asr.reazonspeech_k2", "ReazonSpeechK2ASR"),
    "reazonspeech_whisperx": ("fdbja.asr.reazonspeech_whisperx",
                              "ReazonSpeechWhisperXASR"),
}
DEFAULT_BACKEND = "reazonspeech_whisperx"

_EXTRA_FOR = {
    "reazonspeech_nemo": "asr-nemo",
    "reazonspeech_k2": "asr-k2",
    "reazonspeech_whisperx": "asr-reazon-whisperx",
}


def list_backends() -> List[str]:
    return sorted(BACKENDS)


def register_backend(name: str, factory: Callable[..., ASRModel]) -> None:
    """Register a custom backend (e.g. from user code or tests)."""
    BACKENDS[name] = factory


def load_asr(name: str, **kwargs) -> ASRModel:
    """Instantiate the backend ``name`` with constructor ``kwargs``."""
    if name not in BACKENDS:
        raise ValueError(f"unknown ASR backend {name!r}; "
                         f"available: {', '.join(list_backends())}")
    entry = BACKENDS[name]
    if callable(entry):
        return entry(**kwargs)
    module_name, cls_name = entry
    try:
        module = importlib.import_module(module_name)
        cls = getattr(module, cls_name)
        return cls(**kwargs)
    except ImportError as e:
        extra = _EXTRA_FOR.get(name)
        hint = f" (install with `uv sync --extra {extra}`)" if extra else ""
        raise ImportError(f"ASR backend {name!r} is missing a dependency{hint}: {e}") from e
