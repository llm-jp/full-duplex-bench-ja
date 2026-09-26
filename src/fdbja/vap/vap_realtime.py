"""Reference VAP backend wrapping inokoj/VAP-Realtime.

Requires the VAP-Realtime repository on the path, a Japanese VAP checkpoint
and the CPC encoder. The default checkpoint is
``vap_state_dict_jp_kyoto_20hz_2500msec.pt`` from
https://huggingface.co/maai-kyoto/vap_jp_kyoto (MIT); the older academic-only
``vap_state_dict_jp_20hz_2500msec.pt`` distributed with VAP-Realtime loads
through the same code path. Set the repo location via ``VAP_REALTIME_ROOT`` or
pass ``repo_root``; relative model paths are resolved under ``asset/vap`` and
``asset/cpc``.

This runs VAP incrementally over a stereo pair and captures the full 256-class
softmax per frame (the public offline script only exposes the aggregated
p_now/p_future, so we reimplement the loop here).
"""

from __future__ import annotations

import os
import sys
from typing import Tuple

import numpy as np

from fdbja.vap.base import VAPModel


class VAPRealtimeBackend(VAPModel):
    def __init__(self, vap_model: str, cpc_model: str,
                 repo_root: str | None = None, use_gpu: bool = True,
                 frame_rate: int = 20, context_len_sec: float = 2.5):
        repo_root = repo_root or os.environ.get("VAP_REALTIME_ROOT")
        if not repo_root:
            raise RuntimeError("set VAP_REALTIME_ROOT or pass repo_root")
        sys.path.insert(0, repo_root)
        sys.path.insert(0, os.path.join(repo_root, "rvap", "vap_main"))
        import torch
        from vap_main import VAPRealTime

        self.frame_rate = frame_rate
        self.sample_rate = 16000
        device = torch.device("cuda" if use_gpu and torch.cuda.is_available()
                              else "cpu")
        self._torch = torch
        self._vap = VAPRealTime(vap_model, cpc_model, device, frame_rate,
                                context_len_sec)

    def probs(self, left: np.ndarray, right: np.ndarray
              ) -> Tuple[np.ndarray, np.ndarray]:
        torch = self._torch
        vap = self._vap
        n = min(len(left), len(right))
        left, right = left[:n], right[:n]
        frame_size = vap.audio_frame_size
        shift = frame_size - vap.frame_contxt_padding

        times, probs_seq = [], []
        vap.e1_context, vap.e2_context = [], []
        with torch.no_grad():
            for i in range(0, n, shift):
                if i + frame_size > n:
                    break
                x1 = torch.from_numpy(left[i:i + frame_size]).to(vap.device)
                x2 = torch.from_numpy(right[i:i + frame_size]).to(vap.device)
                x1 = x1.unsqueeze(0).unsqueeze(0).float()
                x2 = x2.unsqueeze(0).unsqueeze(0).float()
                e1, e2 = vap.vap.encode_audio(x1, x2)
                vap.e1_context.append(e1)
                vap.e2_context.append(e2)
                if len(vap.e1_context) > vap.audio_context_len:
                    vap.e1_context = vap.e1_context[-vap.audio_context_len:]
                    vap.e2_context = vap.e2_context[-vap.audio_context_len:]
                c1 = torch.cat(vap.e1_context, dim=1)
                c2 = torch.cat(vap.e2_context, dim=1)
                o1 = vap.vap.ar_channel(c1, attention=False)
                o2 = vap.vap.ar_channel(c2, attention=False)
                out = vap.vap.ar(o1["x"], o2["x"], attention=False)
                p = vap.vap.vap_head(out["x"]).softmax(dim=-1)[0, -1]
                times.append((i + frame_size) / self.sample_rate)
                probs_seq.append(p.cpu().numpy())
        return (np.asarray(times, dtype=np.float32),
                np.asarray(probs_seq, dtype=np.float32))
