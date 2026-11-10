"""Configuration loading."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Dict, List

import yaml


@dataclass
class Config:
    raw: Dict[str, Any]

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path) as f:
            return cls(yaml.safe_load(f))

    def get(self, *keys, default=None):
        node: Any = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    # --- convenience accessors ---
    @property
    def tasks(self) -> List[str]:
        return self.get("tasks", default=["pause", "smooth_tt", "bc"])

    @property
    def eval_dir(self) -> str:
        return self.get("corpus", "eval_dir", default="./data/eval")

    @property
    def models(self) -> List[Dict[str, Any]]:
        return self.get("models", default=[])

    @property
    def clip_seconds(self) -> float:
        return float(self.get("detection", "clip_seconds", default=20.0))

    def detector_params(self, task: str) -> Dict[str, Any]:
        """Keyword arguments for the ``task`` detector from ``detection.<task>``.

        Unknown keys raise instead of being ignored, so a typo in the config
        cannot silently fall back to the default thresholds."""
        from fdbja.data.event_detection import DETECTORS

        params = dict(self.get("detection", task, default={}) or {})
        allowed = set(inspect.signature(DETECTORS[task]).parameters) - {"jchat", "roles"}
        unknown = set(params) - allowed
        if unknown:
            raise ValueError(f"unknown detection.{task} keys: {sorted(unknown)} "
                             f"(allowed: {sorted(allowed)})")
        return params

    @property
    def output_dir(self) -> str:
        return self.get("output", "dir", default="./results")

    def layout(self) -> Dict[str, str]:
        """Filesystem layout templates under eval_dir (overridable in config).

        Placeholders: {task}, {model}, {clip}.
        """
        lay = self.get("layout", default={}) or {}
        return {
            "gt_wav": lay.get("gt_wav", "{task}/{clip}.wav"),
            "gt_json": lay.get("gt_json", "{task}/{clip}.json"),
            "model_wav": lay.get("model_wav", "{task}_{model}_output/{clip}/combined.wav"),
            "model_asr": lay.get("model_asr", "{task}_{model}_asr/text/{clip}.json"),
        }
