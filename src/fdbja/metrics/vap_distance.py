"""VAP-distance: continuous turn-taking timing metric.

For each clip, run a frozen Japanese Voice Activity Projection (VAP) model on
two stereo streams that share the same speaker-A audio:

    P_t = VAP(A + human_B)   (ground-truth stereo)
    Q_t = VAP(A + model_B)   (speaker A + model response)

and compute the frame-wise KL divergence D_KL(P_t || Q_t) over VAP's 256-class
voice-activity-projection distribution. Per event window the mean d_t is the
event score; ``exp(-alpha * D_KL)`` optionally maps it to [0, 1].

The VAP backend is pluggable (see ``fdbja.vap.base.VAPModel``); the reference
implementation wraps inokoj/VAP-Realtime. This module is backend-agnostic: it
only needs per-frame 256-class probability arrays.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np

EPS = 1e-8


def kl_series(p_human: np.ndarray, p_model: np.ndarray) -> np.ndarray:
    """Frame-wise D_KL(P_human || P_model) over the class axis.

    Both inputs are ``(T, n_classes)`` probability arrays; the shorter length
    is used. Clipped for numerical stability (forward KL, mode-covering).
    """
    n = min(len(p_human), len(p_model))
    p = np.clip(p_human[:n], EPS, 1.0)
    q = np.clip(p_model[:n], EPS, 1.0)
    return np.sum(p * np.log(p / q), axis=-1)


def window_stats(times: np.ndarray, dkl: np.ndarray,
                 start: float, end: float) -> dict | None:
    """Aggregate d_t inside an event window [start, end]."""
    mask = (times >= start) & (times <= end)
    if not mask.any():
        return None
    vals = dkl[mask]
    return {
        "mean_dkl": float(vals.mean()),
        "max_dkl": float(vals.max()),
        "n_frames": int(mask.sum()),
    }


def score_from_dkl(dkl: float, alpha: float) -> float:
    """Map a divergence to a [0, 1] score (higher = closer to human)."""
    return float(np.exp(-alpha * dkl))


def calibrate_alpha(desired_dkls: np.ndarray) -> float:
    """alpha so the median desired-behavior event maps to score 0.5."""
    return float(np.log(2) / np.median(desired_dkls))


def event_window(task: str, event: dict, tt_horizon: float = 2.0
                 ) -> Tuple[float, float]:
    """Return the (start, end) time window for an event of a given task."""
    if task == "pause":
        return event["start"], event["end"]
    if task == "smooth_tt":
        return event["a_end"], event["a_end"] + tt_horizon
    if task == "bc":
        return event["a_utt_start"], event["a_utt_end"]
    raise ValueError(f"unknown task: {task}")
