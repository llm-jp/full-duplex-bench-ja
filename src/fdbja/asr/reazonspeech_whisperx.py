"""Reference backend: ReazonSpeech-ESPnet segments + WhisperX character alignment.

This is the pipeline behind the published TOR numbers. ESPnet
(``reazon-research/reazonspeech-espnet-v2``, Apache-2.0) yields segment-level
text with rough start/end; WhisperX (BSD-2-Clause) re-aligns each segment's
characters with a wav2vec2 CTC model (for Japanese, by default
``jonatasgrosman/wav2vec2-large-xlsr-53-japanese``, Apache-2.0). For
languages written without spaces WhisperX treats every character as a
"word", which is exactly the jchat granularity.

WhisperX assumes 16 kHz input, so the audio is resampled before alignment.

The ESPnet checkpoint is fetched with ``huggingface_hub`` and loaded directly
(same repo and ``lm_weight=0`` as ``reazonspeech.espnet.asr.load_model``);
``espnet_model_zoo``'s own downloader is not used because its releases for
Python < 3.12 no longer cope with the Hub's redirects.

Install: ``uv sync --extra asr-reazon-whisperx``.
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from fdbja.asr.base import ASRModel, to_mono_16k, validate_jchat

ESPNET_REPO = "reazon-research/reazonspeech-espnet-v2"


def load_espnet_model(device: str, repo_id: str = ESPNET_REPO):
    """Download ``repo_id`` (meta.yaml layout of espnet_model_zoo) and build
    ``espnet2.bin.asr_inference.Speech2Text`` with relative paths resolved."""
    from espnet2.bin.asr_inference import Speech2Text
    from huggingface_hub import snapshot_download

    root = snapshot_download(repo_id)
    with open(os.path.join(root, "meta.yaml"), encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    cfg_rel = meta["yaml_files"]["asr_train_config"]
    model_file = os.path.join(root, meta["files"]["asr_model_file"])
    with open(os.path.join(root, cfg_rel), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # paths inside the training config are relative to the recipe directory
    for key in ("normalize_conf", "frontend_conf"):
        conf = cfg.get(key) or {}
        for k, v in list(conf.items()):
            if isinstance(v, str) and v.startswith("exp/"):
                conf[k] = os.path.join(root, v)
    resolved = os.path.join(root, cfg_rel[:-5] + ".resolved.yaml")
    if not os.path.exists(resolved):
        with open(resolved, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True)
    return Speech2Text(asr_train_config=resolved, asr_model_file=model_file,
                       lm_weight=0, device=device)


class ReazonSpeechWhisperXASR(ASRModel):
    name = "reazonspeech_whisperx"

    def __init__(self, device: Optional[str] = None, language: str = "ja",
                 align_model_name: Optional[str] = None, verbose: bool = False):
        import torch
        self._rs = importlib.import_module("reazonspeech.espnet.asr")
        self._whisperx = importlib.import_module("whisperx")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.asr_model = load_espnet_model(self.device)
        self.align_model, self.align_meta = self._whisperx.load_align_model(
            language, self.device, model_name=align_model_name)
        self.verbose = verbose

    def segments(self, audio: np.ndarray, sample_rate: int) -> List[Dict[str, Any]]:
        """ESPnet pass only: ``[{"start","end","text"}]`` in seconds.

        Audio is resampled here (the packaged ``norm_audio`` still uses the
        positional ``librosa.resample`` signature removed in librosa 0.10).
        """
        rs = self._rs
        wav = to_mono_16k(audio, sample_rate)
        ret = rs.transcribe(self.asr_model, rs.audio_from_numpy(wav, 16000),
                            rs.TranscribeConfig(verbose=self.verbose))
        return [{"start": float(s.start_seconds), "end": float(s.end_seconds),
                 "text": s.text} for s in ret.segments if s.text.strip()]

    def align(self, segments: List[Dict[str, Any]], audio16k: np.ndarray,
              speaker: str = "B") -> List[Dict[str, Any]]:
        """WhisperX pass only. ``audio16k`` must be mono 16 kHz."""
        if not segments:
            return []
        aligned = self._whisperx.align(segments, self.align_model, self.align_meta,
                                       audio16k, self.device,
                                       return_char_alignments=False)
        items = []
        for seg in aligned.get("segments", []):
            for w in seg.get("words", []):
                if "start" not in w or "end" not in w:
                    continue  # WhisperX could not place this character
                text = str(w.get("word", "")).strip()
                if not text:
                    continue
                items.append({"speaker": speaker, "word": text,
                              "start": round(float(w["start"]), 3),
                              "end": round(float(w["end"]), 3)})
        return validate_jchat(items)

    def transcribe(self, audio: np.ndarray, sample_rate: int,
                   speaker: str = "B") -> List[Dict[str, Any]]:
        segs = self.segments(audio, sample_rate)
        return self.align(segs, to_mono_16k(audio, sample_rate), speaker=speaker)
