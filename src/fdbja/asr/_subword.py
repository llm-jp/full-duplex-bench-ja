"""Character-level times from subword tokens that carry a single timestamp.

ReazonSpeech NeMo / k2 return each subword with one time (its emission point,
roughly the onset), not a span. TOR needs a ``[start, end]`` per character, so:

* a subword starts at its timestamp ``t_i``;
* its nominal end is ``t_i + char_seconds * n_chars``; if the next subword
  starts within ``bridge_seconds`` after that, the span is extended to the next
  onset (emission times are sparse, and a transducer often fires a token late
  within a word; without bridging one word splinters into several
  "utterances"); otherwise the token before a silence keeps its nominal
  duration. The final token gets the nominal duration, clipped to the audio
  length;
* the span is **split evenly** over the subword's characters. This is an
  approximation: within a multi-character subword the per-character boundaries
  are not acoustically measured.

Utterance grouping downstream (``group_chars_to_utterances``) splits at gaps
longer than 0.3 s. With the defaults, silences longer than about 0.6 s after a
one-character token still split utterances; on six 20 s clips this gave 24
utterances where the ESPnet + WhisperX reference path gave 21 (without
bridging: 40). ``bridge_seconds`` matches the 0.5 s phonemic-break threshold
ReazonSpeech itself uses to cut segments.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

#: mean Japanese mora duration is ~0.1-0.15 s; one character ~ one mora
DEFAULT_CHAR_SECONDS = 0.10
#: extend a token to the next onset when the silence after its nominal end is
#: at most this long (ReazonSpeech's own segmenter breaks at 0.5 s)
DEFAULT_BRIDGE_SECONDS = 0.5
#: SentencePiece whitespace marker
_SP_SPACE = "▁"


def subwords_to_chars(subwords: Iterable[Tuple[str, float]], speaker: str = "B",
                      char_seconds: float = DEFAULT_CHAR_SECONDS,
                      bridge_seconds: float = DEFAULT_BRIDGE_SECONDS,
                      audio_duration: Optional[float] = None) -> List[Dict[str, Any]]:
    """Convert ``(token, seconds)`` pairs into character-level jchat items.

    Tokens are cleaned (``▁`` and surrounding whitespace removed) and empty
    ones dropped. Timestamps are assumed non-decreasing; a token whose time is
    earlier than the previous one is clamped to the previous time.
    """
    toks: List[Tuple[str, float]] = []
    prev = 0.0
    for token, sec in subwords:
        text = str(token).replace(_SP_SPACE, "").strip()
        if not text:
            continue
        t = max(float(sec), prev, 0.0)
        toks.append((text, t))
        prev = t

    items: List[Dict[str, Any]] = []
    for i, (text, start) in enumerate(toks):
        nominal = start + char_seconds * len(text)
        if i + 1 < len(toks):
            nxt = toks[i + 1][1]
            end = nxt if nxt - nominal <= bridge_seconds else nominal
        else:
            end = nominal
        if audio_duration is not None:
            end = min(end, float(audio_duration))
        end = max(end, start)
        items.extend(_split_evenly(text, start, end, speaker))
    return items


def _split_evenly(text: Sequence[str], start: float, end: float,
                  speaker: str) -> List[Dict[str, Any]]:
    n = len(text)
    step = (end - start) / n
    out = []
    for k, ch in enumerate(text):
        s = start + k * step
        e = end if k == n - 1 else start + (k + 1) * step
        out.append({"speaker": speaker, "word": ch,
                    "start": round(s, 3), "end": round(e, 3)})
    return out
