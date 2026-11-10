"""VAP-distance: continuous turn-taking timing metric.

For each clip, run a frozen Japanese Voice Activity Projection (VAP) model on
two stereo streams that share the same speaker-A audio:

    P_t = VAP(A + human_B)   (ground-truth stereo)
    Q_t = VAP(A + model_B)   (speaker A + model response)

and compute the frame-wise KL divergence D_KL(P_t || Q_t) over VAP's 256-class
voice-activity-projection distribution. Per event window the mean d_t is the
event score; ``exp(-alpha * D_KL)`` optionally maps it to [0, 1].

Boundary-weighted variant (``metrics.vap_distance.weighting: boundary``):
instead of the equal-weight mean over the event window, d_t is averaged with
weights ``w_t = exp(-|t - t0| / tau)`` around the boundary t0 (Smooth TT: the
end of A's utterance; Pause: the start of the gap), over the stretch a
listener heard in the human-judgement study (the judgement window extended by
5 s before and 1.5 s after). ``tau = 2.0`` s is the setting chosen there. The
equal-weight score is left as it is; the weighted one is reported next to it.

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


#: listened stretch around the judgement window in the human-judgement study (s)
BOUNDARY_PRE = 5.0
BOUNDARY_POST = 1.5
DEFAULT_TAU = 2.0


def boundary_window(task: str, event: dict, pre: float = BOUNDARY_PRE,
                    post: float = BOUNDARY_POST) -> Tuple[float, float, float]:
    """``(start, end, t0)`` of the boundary-weighted average of an event.

    The judgement window is the one of ``metrics/tor.py`` (Pause: the gap;
    Smooth TT: 0.5 s before to 2.0 s after the end of A's utterance), widened
    by ``pre`` / ``post`` seconds; t0 is the start of the gap (Pause) or the
    end of A's utterance (Smooth TT)."""
    if task == "pause":
        return event["start"] - pre, event["end"] + post, event["start"]
    if task == "smooth_tt":
        a_end = event["a_end"]
        return a_end - 0.5 - pre, a_end + 2.0 + post, a_end
    raise ValueError(f"boundary weighting is defined for pause / smooth_tt, not {task}")


def boundary_weights(times: np.ndarray, t0: float, tau: float) -> np.ndarray:
    """``exp(-|t - t0| / tau)``; ``tau = inf`` gives equal weights."""
    times = np.asarray(times, dtype=np.float64)
    if np.isinf(tau):
        return np.ones_like(times)
    if tau <= 0:
        raise ValueError(f"tau_seconds must be positive, got {tau}")
    return np.exp(-np.abs(times - t0) / tau)


def weighted_window_stats(times: np.ndarray, dkl: np.ndarray, start: float,
                          end: float, t0: float, tau: float = DEFAULT_TAU
                          ) -> dict | None:
    """Boundary-weighted mean of d_t over [start, end] (weights normalised
    over the frames in the window)."""
    times = np.asarray(times, dtype=np.float64)
    mask = (times >= start) & (times <= end)
    if not mask.any():
        return None
    w = boundary_weights(times[mask], t0, tau)
    return {"mean_dkl": float((w * dkl[mask]).sum() / w.sum()),
            "n_frames": int(mask.sum())}
