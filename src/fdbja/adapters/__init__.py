"""Model adapter registry."""

from __future__ import annotations

from fdbja.adapters.base import ModelAdapter, DialogueResponse


def build_adapter(spec: dict) -> ModelAdapter:
    """Instantiate a ModelAdapter from a config model spec.

    ``spec['adapter']`` selects the backend; remaining keys (minus name/domain)
    are passed as kwargs.
    """
    kind = spec.get("adapter")
    kwargs = {k: v for k, v in spec.items()
              if k not in ("adapter", "name", "domain")}
    if kind == "moshi_ws":
        from fdbja.adapters.moshi_ws import MoshiWebSocketAdapter
        return MoshiWebSocketAdapter(**kwargs)
    if kind == "gemini_live":
        from fdbja.adapters.gemini_live import GeminiLiveAdapter
        return GeminiLiveAdapter(**kwargs)
    if kind == "openai_realtime":
        from fdbja.adapters.openai_realtime import OpenAIRealtimeAdapter
        return OpenAIRealtimeAdapter(**kwargs)
    if kind == "silence":
        from fdbja.adapters.baselines import SilenceAdapter
        return SilenceAdapter(**kwargs)
    if kind == "always_talk":
        from fdbja.adapters.baselines import AlwaysTalkAdapter
        return AlwaysTalkAdapter(**kwargs)
    raise ValueError(f"unknown adapter: {kind!r}")


__all__ = ["ModelAdapter", "DialogueResponse", "build_adapter"]
