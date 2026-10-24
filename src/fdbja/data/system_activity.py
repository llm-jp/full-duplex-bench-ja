"""System-side speech activity from the model's output audio.

``detection.system_activity: audio``. The binary TOR asks *when* the model is
speaking (does its speech overlap a pause of speaker A, does it start within
the window after A's turn ends) and *what* it says (how many characters, to
tell a backchannel from a take-over). With ``transcript`` (the published
definition) both come from the ASR jchat of the model channel, so the timing
depends on how the ASR backend places character boundaries. Backends differ a
lot there: an aligner that is not asked for character alignments stretches the
last character of a segment over the trailing silence (e.g. one character
spanning 2.08-5.30 s), and the model then appears to talk through the
following pause of A.

``audio`` takes the timing from the output audio instead and keeps the ASR for
the content only:

  * voice activity of the model channel with the same frame-energy detector as
    ``detection.boundaries: audio`` (``event_boundaries.voice_activity``: 20 ms
    frames, ITU-T P.56 style threshold, short blips dropped, short holes
    filled);
  * voiced runs separated by at most ``merge_gap`` seconds of silence are joined
    into one speech segment (the same 0.3 s that ``group_chars_to_utterances``
    uses between characters);
  * the characters come from the ASR utterances (``group_chars_to_utterances``
    of the jchat): the characters of each ASR utterance are shared among the
    speech segments it overlaps, in proportion to the overlap. Character
    *start* times are not trusted either: backends that align segment by
    segment spread the characters of a segment evenly over it, so a single
    character can land on the wrong side of a silence. An ASR utterance that
    overlaps no segment is matched within ``max_offset`` seconds; if there is
    still none its characters are not counted (an ASR hallucination on
    silence, or speech below the detector);
  * an utterance is a speech segment that receives at least ``min_chars``
    characters: ``start`` / ``end`` from the audio, ``num_chars`` (fractional,
    rounded to 0.01) from the ASR. Segments without characters (breaths,
    noise, laughter the ASR did not write) are not utterances, as with
    ``transcript``.

The TOR rules themselves (``metrics/tor.py``) are unchanged. VAP-distance does
not read the transcript and is not affected.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from fdbja.data.event_boundaries import DEFAULT_VAD, VADConfig, _runs, voice_activity
from fdbja.data.event_detection import group_chars_to_utterances

JChat = List[Dict[str, Any]]
Utterance = Dict[str, Any]


@dataclass(frozen=True)
class SystemActivityConfig:
    vad: VADConfig = DEFAULT_VAD
    merge_gap: float = 0.3      # join voiced runs separated by <= this silence (s)
    max_offset: float = 0.5     # an ASR utterance may lie this far outside any segment (s)
    min_chars: float = 0.5      # a segment needs this many characters to be an utterance
    channel: int = 1            # model channel of combined.wav

    @classmethod
    def from_config(cls, cfg: Optional[Dict[str, Any]]) -> "SystemActivityConfig":
        """From ``detection.system_vad``: ``merge_gap`` / ``max_offset`` /
        ``channel`` plus any ``VADConfig`` field (``mode``, ``margin_db``, ...)."""
        cfg = dict(cfg or {})
        own = {f.name for f in fields(cls)} - {"vad"}
        vad_names = {f.name for f in fields(VADConfig)}
        unknown = set(cfg) - own - vad_names
        if unknown:
            raise ValueError(f"unknown detection.system_vad keys: {sorted(unknown)}")
        vad = replace(DEFAULT_VAD, **{k: v for k, v in cfg.items() if k in vad_names})
        return cls(vad=vad, **{k: v for k, v in cfg.items() if k in own})


DEFAULT_SYSTEM_ACTIVITY = SystemActivityConfig()


def speech_segments(va: np.ndarray, frame: float,
                    merge_gap: float = 0.3) -> List[Tuple[float, float]]:
    """(start, end) seconds of the voiced runs of ``va``, joining runs whose
    silence in between is at most ``merge_gap``."""
    segs: List[Tuple[float, float]] = []
    for s, e, v in _runs(va):
        if not v:
            continue
        ts, te = s * frame, e * frame
        if segs and ts - segs[-1][1] <= merge_gap + 1e-9:
            segs[-1] = (segs[-1][0], te)
        else:
            segs.append((ts, te))
    return [(round(s, 3), round(e, 3)) for s, e in segs]


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def utterances_from_activity(jchat: JChat, va: np.ndarray, speaker: str = "B",
                             cfg: SystemActivityConfig = DEFAULT_SYSTEM_ACTIVITY
                             ) -> Tuple[List[Utterance], Dict[str, float]]:
    """Utterances of ``speaker`` whose timing comes from ``va`` and whose
    characters come from ``jchat``. Returns ``(utterances, stats)`` with
    ``stats = {"chars", "unassigned_chars", "segments", "empty_segments"}``."""
    segs = speech_segments(va, cfg.vad.frame, cfg.merge_gap)
    asr_utts = group_chars_to_utterances(jchat, speaker)
    alloc = [0.0] * len(segs)
    texts: List[List[str]] = [[] for _ in segs]
    chars = 0
    unassigned = 0.0
    for u in asr_utts:
        n = u["num_chars"]
        chars += n
        for pad in (0.0, cfg.max_offset):
            ov = [_overlap(s, e, u["start"] - pad, u["end"] + pad) for s, e in segs]
            tot = sum(ov)
            if tot > 0:
                break
        if tot <= 0:
            unassigned += n
            continue
        for i, o in enumerate(ov):
            if o > 0:
                alloc[i] += n * o / tot
                texts[i].append(u["text"])
    utts: List[Utterance] = []
    for (s, e), a, tx in zip(segs, alloc, texts):
        if a < cfg.min_chars:
            unassigned += a
            continue
        utts.append({
            "start": s,
            "end": e,
            "num_chars": round(a, 2),
            # the ASR utterances this segment took characters from (not cut)
            "text": " / ".join(tx),
            "source": "audio",
        })
    stats = {"chars": chars, "unassigned_chars": round(unassigned, 2),
             "segments": len(segs), "empty_segments": len(segs) - len(utts)}
    return utts, stats


def utterances_from_wav(jchat: JChat, wav_path: str, speaker: str = "B",
                        cfg: SystemActivityConfig = DEFAULT_SYSTEM_ACTIVITY
                        ) -> Tuple[List[Utterance], Dict[str, float]]:
    """``utterances_from_activity`` on the model channel of ``wav_path``
    (``cfg.channel`` of a stereo file, a mono file as is)."""
    import soundfile as sf
    x, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    ch = min(cfg.channel, x.shape[1] - 1)
    va = voice_activity(x[:, ch], sr, cfg.vad)
    return utterances_from_activity(jchat, va, speaker, cfg)
