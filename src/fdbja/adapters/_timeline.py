"""Response timeline shared by the streaming adapters.

Every streaming adapter receives the model's audio in chunks. How those chunks
are laid out on the time axis decides the response latency and the length of
each utterance that the benchmark scores, so all adapters use the same rules
here.

Rules (``build_timeline(rule=...)``):

``playout``
    What a real client plays. A chunk starts when it arrives, or when the
    audio queued before it has finished playing, whichever is later (servers
    may send audio faster than real time). When the server reports that the
    user interrupted the model, playback stops at the time the notice arrived
    and everything still waiting in the queue is discarded.
``arrival``
    The same queue, but interruption notices are ignored and every received
    chunk stays on the timeline (``gemini_live`` and ``openai_realtime`` up to
    0.2.0).
``concat``
    Chunks are joined in the order received, starting at time 0, without
    using arrival times (``moshi_ws`` up to 0.2.0).

Adapters take a ``timeline`` option: ``"playout"`` or ``"legacy"`` (the rule
that adapter used up to 0.2.0, for reproducing published values).

The event record (``ResponseTimeline.record()``) keeps the arrival time and
sample count of every chunk, the time of every notice, and when and why the
adapter stopped listening (the note ``client.receive_end``). Together with the
received audio (all chunks joined in order) it is enough to rebuild the
timeline under any rule without calling the model again
(``timeline_from_record``).
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

RULES = ("playout", "arrival", "concat")
TIMELINE_OPTIONS = ("playout", "legacy")
DEFAULT_TIMELINE = "playout"
RECORD_FORMAT = "fdbja.response_events/1"
#: note written by an adapter when it stops listening (``reason`` says why)
RECEIVE_END = "client.receive_end"
SESSION = "client.session"      # note at time 0: the settings the session was opened with


def text_id(text: str) -> str:
    """Short identifier of a text (first 12 hex digits of its SHA-256), for
    records that do not hold the text itself."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def resolve_rule(timeline: str, legacy_rule: str) -> str:
    """Map an adapter's ``timeline`` option to a rule name."""
    if timeline == "legacy":
        return legacy_rule
    if timeline == "playout":
        return "playout"
    raise ValueError(
        f"timeline must be one of {TIMELINE_OPTIONS}, got {timeline!r}")


class _Queue:
    """Playback queue of one response channel (sample positions only)."""

    def __init__(self, sample_rate: int, flush: bool):
        self.sample_rate = sample_rate
        self.flush = flush
        self.cursor = 0                 # where the queued audio ends
        self.spans: list[list] = []     # [start, n, source_offset, item]
        self.dead: set = set()          # items cut off by an interruption
        self._seen: set = set()

    def audio(self, t: float, n: int, offset: int, item=None) -> bool:
        """Queue ``n`` samples that arrived at ``t``. False if discarded."""
        if item is not None:
            self._seen.add(item)
            if item in self.dead:
                return False
        start = max(int(t * self.sample_rate), self.cursor)
        self.spans.append([start, n, offset, item])
        self.cursor = start + n
        return True

    def interrupt(self, t: float) -> dict:
        """Stop playback at ``t``. Returns ``{item: samples_played}`` for the
        items that lost audio (samples played before the stop)."""
        if not self.flush:
            return {}
        ts = int(t * self.sample_rate)
        cut: dict = {}
        kept = []
        for span in self.spans:
            start, n, _, item = span
            if start + n <= ts:
                kept.append(span)
                continue
            cut.setdefault(item, 0)
            if start < ts:
                span[1] = ts - start
                kept.append(span)
        for start, n, _, item in kept:
            if item in cut:
                cut[item] += n
        self.spans = kept
        self.cursor = min(self.cursor, ts)
        # A response that was audible or queued at the interruption is over:
        # audio for it that is still in flight must not be played.
        self.dead |= self._seen
        return cut


def build_timeline(events, received: np.ndarray, sample_rate: int,
                   rule: str = "playout", min_samples: int = 0,
                   max_samples: int | None = None) -> np.ndarray:
    """Lay the received audio out on the time axis.

    Args:
        events: the ``events`` list of a record (``type`` ``audio`` /
            ``interruption`` entries are used, in order; others are ignored).
        received: every received chunk joined in order.
        rule: ``playout`` / ``arrival`` / ``concat``.
        min_samples: pad the result with silence up to this length.
        max_samples: cut the result at this length (``None`` = keep all).
    """
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, got {rule!r}")
    received = np.asarray(received)
    if rule == "concat":
        n_audio = sum(e["n"] for e in events if e["type"] == "audio")
        out = received[:n_audio].copy()
        if len(out) < min_samples:
            out = np.concatenate(
                [out, np.zeros(min_samples - len(out), dtype=received.dtype)])
        return out if max_samples is None else out[:max_samples]

    queue = _Queue(sample_rate, flush=(rule == "playout"))
    offset = 0
    for e in events:
        if e["type"] == "audio":
            queue.audio(e["t"], e["n"], offset, e.get("item"))
            offset += e["n"]
        elif e["type"] == "interruption":
            queue.interrupt(e["t"])
    end = max([min_samples] + [s[0] + s[1] for s in queue.spans])
    out = np.zeros(end, dtype=received.dtype)
    for start, n, src, _ in queue.spans:
        out[start:start + n] = received[src:src + n]
    return out if max_samples is None else out[:max_samples]


class ResponseTimeline:
    """Collects what an adapter receives during one clip.

    Times are seconds on the client clock, measured from the moment the first
    input sample is due to be sent.
    """

    def __init__(self, sample_rate: int, adapter: str = ""):
        self.sample_rate = sample_rate
        self.adapter = adapter
        self.events: list[dict] = []
        self._chunks: list[np.ndarray] = []
        self._n_received = 0
        self._queue = _Queue(sample_rate, flush=True)

    def add_audio(self, t: float, pcm: np.ndarray, item=None, **info) -> None:
        pcm = np.asarray(pcm)
        t = round(float(t), 4)      # the saved value, so a rebuild is identical
        played = self._queue.audio(t, len(pcm), self._n_received, item)
        event = {"t": t, "type": "audio", "n": int(len(pcm))}
        if item is not None:
            event["item"] = item
        event.update({k: v for k, v in info.items() if v is not None})
        if not played:
            event["after_interruption"] = True
        self.events.append(event)
        self._chunks.append(pcm)
        self._n_received += len(pcm)

    def interrupt(self, t: float, source: str, **info) -> dict:
        """Record an interruption notice. Returns ``{item: milliseconds
        played}`` for the items whose remaining audio a playing client
        discards at this moment."""
        t = round(float(t), 4)
        cut = self._queue.interrupt(t)
        event = {"t": t, "type": "interruption", "source": source}
        event.update({k: v for k, v in info.items() if v is not None})
        self.events.append(event)
        return {item: int(n * 1000 // self.sample_rate)
                for item, n in cut.items()}

    def note(self, t: float, kind: str, **info) -> None:
        """Record any other server or client event (kept for analysis)."""
        event = {"t": round(float(t), 4), "type": "note", "event": kind}
        event.update({k: v for k, v in info.items() if v is not None})
        self.events.append(event)

    def received_audio(self) -> np.ndarray:
        if not self._chunks:
            return np.zeros(0, dtype=np.int16)
        return np.concatenate(self._chunks)

    def render(self, rule: str = "playout", min_samples: int = 0,
               max_samples: int | None = None) -> np.ndarray:
        return build_timeline(self.events, self.received_audio(),
                              self.sample_rate, rule=rule,
                              min_samples=min_samples, max_samples=max_samples)

    def record(self, timeline: str, rule: str, min_samples: int = 0,
               max_samples: int | None = None) -> dict:
        """JSON-serialisable record of this clip (no audio, no text)."""
        return {
            "format": RECORD_FORMAT,
            "adapter": self.adapter,
            "sample_rate": self.sample_rate,
            "timeline": timeline,
            "rule": rule,
            "min_samples": int(min_samples),
            "max_samples": None if max_samples is None else int(max_samples),
            "received_samples": int(self._n_received),
            "events": self.events,
        }


def timeline_from_record(record: dict, received: np.ndarray,
                         rule: str | None = None) -> np.ndarray:
    """Rebuild the response channel from a saved record and the received
    audio. ``rule=None`` reproduces the timeline that was written at
    recording time."""
    if record.get("format") != RECORD_FORMAT:
        raise ValueError(f"unknown record format: {record.get('format')!r}")
    same = rule is None or rule == record["rule"]
    return build_timeline(
        record["events"], received, record["sample_rate"],
        rule=rule or record["rule"], min_samples=record.get("min_samples", 0),
        max_samples=record.get("max_samples") if same else None)


def load_record(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
