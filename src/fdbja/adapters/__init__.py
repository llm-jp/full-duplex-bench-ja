"""Model adapter registry.

A config model entry selects its adapter with ``adapter:``:

* a registered name (``moshi_ws``, ``gemini_live``, ``openai_realtime``,
  ``silence``, ``always_talk`` or one added with :func:`register_adapter`);
* ``"package.module:ClassName"``, a :class:`ModelAdapter` subclass imported
  from your own code (the module must be importable, e.g. installed or on
  ``PYTHONPATH``);
* ``offline`` (or no ``adapter`` key): no live model. ``infer`` skips the
  entry and only reports how many clips still lack their ``combined.wav``;
  put the recordings there yourself and run ``transcribe`` / ``score-offline``.
"""

from __future__ import annotations

import importlib
from typing import Callable, Dict, List, Optional, Tuple, Union

from fdbja.adapters.base import DialogueResponse, ModelAdapter

#: name -> (module, class) or a factory callable returning a ModelAdapter
ADAPTERS: Dict[str, Union[Tuple[str, str], Callable[..., ModelAdapter]]] = {
    "moshi_ws": ("fdbja.adapters.moshi_ws", "MoshiWebSocketAdapter"),
    "gemini_live": ("fdbja.adapters.gemini_live", "GeminiLiveAdapter"),
    "openai_realtime": ("fdbja.adapters.openai_realtime", "OpenAIRealtimeAdapter"),
    "silence": ("fdbja.adapters.baselines", "SilenceAdapter"),
    "always_talk": ("fdbja.adapters.baselines", "AlwaysTalkAdapter"),
}

#: adapter values that mean "outputs are placed by hand; nothing to run"
OFFLINE = ("offline", None, "")

#: config keys that are not constructor arguments
_RESERVED = ("adapter", "name", "domain")

#: the trivial baselines (``adapters/baselines.py``): reported on every task,
#: never ranked, optionally kept out of the VAP-distance alpha calibration
BASELINE_ADAPTERS = frozenset({"silence", "always_talk"})


def register_adapter(name: str, factory: Callable[..., ModelAdapter]) -> None:
    """Register a custom adapter (a ModelAdapter subclass or any factory)."""
    if name in ("offline", ""):
        raise ValueError(f"{name!r} is reserved")
    ADAPTERS[name] = factory


def list_adapters() -> List[str]:
    return sorted(ADAPTERS)


def is_offline(spec: dict) -> bool:
    """True when the model entry has no adapter to run (``adapter: offline``)."""
    return spec.get("adapter") in OFFLINE


def _resolve(kind: str):
    if kind in ADAPTERS:
        entry = ADAPTERS[kind]
        if callable(entry):
            return entry
        module_name, cls_name = entry
        return getattr(importlib.import_module(module_name), cls_name)
    if isinstance(kind, str) and ":" in kind:
        module_name, _, attr = kind.partition(":")
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            raise ValueError(f"cannot import adapter module {module_name!r} "
                             f"(from adapter: {kind!r}): {e}") from e
        obj = module
        for part in attr.split("."):
            if not hasattr(obj, part):
                raise ValueError(f"{module_name!r} has no attribute {attr!r} "
                                 f"(from adapter: {kind!r})")
            obj = getattr(obj, part)
        return obj
    raise ValueError(
        f"unknown adapter: {kind!r}; available: {', '.join(list_adapters())}, "
        "'module.path:ClassName' for your own ModelAdapter, or 'offline' to "
        "score combined.wav files placed by hand")


def build_adapter(spec: dict) -> Optional[ModelAdapter]:
    """Instantiate a ModelAdapter from a config model spec.

    ``spec['adapter']`` selects the backend (see the module docstring);
    remaining keys (minus name/domain) are passed as kwargs. Returns ``None``
    for ``adapter: offline``.
    """
    kind = spec.get("adapter")
    if kind in OFFLINE:
        return None
    kwargs = {k: v for k, v in spec.items() if k not in _RESERVED}
    return _resolve(kind)(**kwargs)


__all__ = ["BASELINE_ADAPTERS", "ADAPTERS", "DialogueResponse", "ModelAdapter", "build_adapter",
           "is_offline", "list_adapters", "register_adapter"]
