"""OpenAI-Realtime-protocol adapter (GPT-Realtime and Grok Voice).

Speaks the OpenAI Realtime WebSocket protocol: paced ``input_audio_buffer.append``
frames in, ``response.output_audio.delta`` frames out. xAI's Grok Voice Agent
API (wss://api.x.ai/v1/realtime) implements the same event schema, so one
adapter covers both; only the ``session.update`` payload differs slightly
(``dialect``: "openai" nests voice/turn_detection under ``audio``, "xai" puts
them at the session top level).

Timeline (``timeline`` option, see ``fdbja.adapters._timeline``):

``playout`` (default)
    The response channel is what a real client plays. Audio is queued by
    arrival time (the server sends it faster than real time). When the server
    reports that the user started speaking (``input_audio_buffer.speech_started``),
    playback stops at the time the notice arrived, the audio still waiting in
    the queue is discarded, and the adapter sends ``conversation.item.truncate``
    with the milliseconds actually played, as the API guide asks WebSocket
    clients to do. The server marks the cut-off response with ``response.done``
    whose ``response.status`` is ``cancelled``; audio for that response that
    arrives after the notice is not played.
``legacy``
    The construction up to 0.2.0: interruption notices are ignored, nothing is
    sent back, and every received chunk stays on the timeline. A reply that
    the API cut short is then scored as continuous speech (for example a short
    start in a Pause can become an utterance over 1.5 s and TO = 1). Use it to
    reproduce values published with 0.2.0.

Each clip also returns an event record (arrival time and sample count of every
chunk, time of every notice) and the received audio, which ``fdbja infer``
saves as ``events.json`` / ``received.wav`` next to ``combined.wav``.

Requires ``websockets`` + ``librosa`` (extra ``[realtime]``) and an API key
(``OPENAI_API_KEY`` / ``XAI_API_KEY`` by default, or ``api_key`` in config).
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time

import numpy as np

from fdbja.adapters._timeline import (
    DEFAULT_TIMELINE,
    RECEIVE_END,
    ResponseTimeline,
    resolve_rule,
)
from fdbja.adapters.base import DialogueResponse, ModelAdapter

WIRE_SR = 24000            # both APIs default to 24 kHz PCM16
CHUNK_MS = 100
CHUNK_SAMPLES = int(WIRE_SR * CHUNK_MS / 1000)
AUDIO_DELTAS = ("response.output_audio.delta", "response.audio.delta")
SPEECH_STARTED = "input_audio_buffer.speech_started"
TRUNCATE_ID = "fdbja_truncate_"     # event_id prefix of the truncations we send
_NOTE_KEYS = ("item_id", "response_id", "content_index", "audio_start_ms",
              "audio_end_ms")

SYSTEM_INSTRUCTION = (
    "あなたは日本語の対話相手です。自然に日本語で応答してください。"
    "相槌（「うん」「はい」「なるほど」など）も適切に使ってください。"
    "必ず日本語のみで応答してください。"
)


def _retryable(e: Exception) -> bool:
    """Transient failures worth a backoff-retry: rate limits (HTTP 429),
    5xx rejections, dropped connections, and timeouts."""
    import websockets
    if isinstance(e, websockets.exceptions.InvalidStatus):
        code = getattr(getattr(e, "response", None), "status_code", None)
        return code == 429 or (code is not None and code >= 500)
    return isinstance(e, (websockets.exceptions.ConnectionClosedError,
                          asyncio.TimeoutError, OSError))


def _to_wire_i16(audio: np.ndarray, sr: int) -> np.ndarray:
    a = np.asarray(audio)
    if a.ndim == 2:
        a = a[0]
    if a.dtype != np.float32:
        a = a.astype(np.float32)
    if sr != WIRE_SR:
        import librosa
        a = librosa.resample(a, orig_sr=sr, target_sr=WIRE_SR)
    return np.clip(a * 32768, -32768, 32767).astype(np.int16)


class OpenAIRealtimeAdapter(ModelAdapter):
    name = "openai_realtime"

    def __init__(self, url: str = "wss://api.openai.com/v1/realtime",
                 model: str = "gpt-realtime", voice: str = "marin",
                 dialect: str | None = None, silence_after_s: float = 5.0,
                 instructions: str = SYSTEM_INSTRUCTION,
                 max_retries: int = 8, retry_wait_s: float = 15.0,
                 api_key: str | None = None, api_key_env: str | None = None,
                 timeline: str = DEFAULT_TIMELINE, send_truncate: bool = True,
                 **_):
        self.url = url
        self.model = model
        self.voice = voice
        self.dialect = dialect or ("xai" if "x.ai" in url else "openai")
        self.silence_after_s = silence_after_s
        self.instructions = instructions
        self.max_retries = max_retries
        self.retry_wait_s = retry_wait_s
        self.timeline = timeline
        self.rule = resolve_rule(timeline, legacy_rule="arrival")
        self.send_truncate = send_truncate
        env = api_key_env or ("XAI_API_KEY" if self.dialect == "xai"
                              else "OPENAI_API_KEY")
        self.api_key = api_key or os.environ.get(env)
        if not self.api_key:
            raise RuntimeError(f"{env} not set (adapter {self.name})")

    # ------------------------------------------------------------------
    def infer(self, input_audio: np.ndarray, sample_rate: int) -> DialogueResponse:
        input_pcm = _to_wire_i16(input_audio, sample_rate)
        for attempt in range(self.max_retries + 1):
            try:
                output_pcm, tl, events = asyncio.run(self._stream(input_pcm))
                break
            except Exception as e:  # noqa: BLE001
                if attempt >= self.max_retries or not _retryable(e):
                    raise
                wait = min(300.0, self.retry_wait_s * (2 ** attempt))
                print(f"[{self.name}] {type(e).__name__} "
                      f"(attempt {attempt + 1}/{self.max_retries}), "
                      f"retrying in {wait:.0f}s: {e}")
                time.sleep(wait)
        return DialogueResponse(
            input_audio=input_pcm.astype(np.float32) / 32768.0,
            response_audio=output_pcm.astype(np.float32) / 32768.0,
            sample_rate=WIRE_SR,
            events=events,
            received_audio=tl.received_audio().astype(np.float32) / 32768.0,
        )

    def _session_update(self) -> dict:
        fmt = {"type": "audio/pcm", "rate": WIRE_SR}
        if self.dialect == "xai":
            session = {
                "voice": self.voice,
                "instructions": self.instructions,
                "turn_detection": {"type": "server_vad"},
                "audio": {"input": {"format": fmt}, "output": {"format": fmt}},
            }
        else:
            session = {
                "type": "realtime",
                "instructions": self.instructions,
                "audio": {
                    "input": {"format": fmt,
                              "turn_detection": {"type": "server_vad"}},
                    "output": {"format": fmt, "voice": self.voice},
                },
            }
        return {"type": "session.update", "session": session}

    async def _stream(self, input_pcm: np.ndarray):
        import websockets

        total = len(input_pcm)
        total_dur = total / WIRE_SR
        deadline = total_dur + self.silence_after_s + 3.0   # recv grace
        tl = ResponseTimeline(WIRE_SR, adapter=self.name)
        playout = self.rule == "playout"
        content_index: dict = {}                            # item -> index
        n_truncate = 0
        send_done = [False]

        headers = {"Authorization": f"Bearer {self.api_key}"}
        uri = f"{self.url}?model={self.model}" if "?" not in self.url else self.url
        async with websockets.connect(uri, additional_headers=headers,
                                      max_size=16 * 1024 * 1024) as ws:
            await ws.send(json.dumps(self._session_update()))
            t0 = time.monotonic()

            async def send_audio():
                silence = np.zeros(CHUNK_SAMPLES, dtype=np.int16)
                n_sil = int(self.silence_after_s * 1000 / CHUNK_MS)
                n_chunks = (total + CHUNK_SAMPLES - 1) // CHUNK_SAMPLES
                for k in range(n_chunks + n_sil):
                    dt = k * CHUNK_MS / 1000 - (time.monotonic() - t0)
                    if dt > 0:
                        await asyncio.sleep(dt)
                    if k < n_chunks:
                        chunk = input_pcm[k * CHUNK_SAMPLES:(k + 1) * CHUNK_SAMPLES]
                        if len(chunk) < CHUNK_SAMPLES:
                            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
                    else:
                        chunk = silence
                    await ws.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk.tobytes()).decode(),
                    }))
                send_done[0] = True

            async def recv_audio():
                reason = await receive()
                tl.note(time.monotonic() - t0, RECEIVE_END, reason=reason)

            async def receive() -> str:
                """Handle server messages; returns why listening stopped."""
                nonlocal n_truncate
                while True:
                    remaining = deadline - (time.monotonic() - t0)
                    if remaining <= 0:
                        return "deadline"
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                    except asyncio.TimeoutError:
                        return "deadline"
                    except websockets.exceptions.ConnectionClosed:
                        return "closed"
                    msg = json.loads(raw)
                    mtype = msg.get("type", "")
                    at = time.monotonic() - t0
                    if mtype in AUDIO_DELTAS:
                        pcm = np.frombuffer(base64.b64decode(msg["delta"]),
                                            dtype=np.int16)
                        item = msg.get("item_id")
                        content_index[item] = msg.get("content_index", 0)
                        tl.add_audio(at, pcm, item=item,
                                     response=msg.get("response_id"))
                    elif mtype == SPEECH_STARTED:
                        # the user barged in: stop playback, drop the queue
                        cut = tl.interrupt(
                            at, source=mtype,
                            audio_start_ms=msg.get("audio_start_ms"))
                        if not (playout and self.send_truncate):
                            continue
                        for item, played_ms in cut.items():
                            if item is None:
                                continue
                            n_truncate += 1
                            await ws.send(json.dumps({
                                "type": "conversation.item.truncate",
                                "event_id": f"{TRUNCATE_ID}{n_truncate}",
                                "item_id": item,
                                "content_index": content_index.get(item, 0),
                                "audio_end_ms": played_ms,
                            }))
                            tl.note(time.monotonic() - t0,
                                    "conversation.item.truncate (sent)",
                                    item_id=item, audio_end_ms=played_ms)
                    elif mtype == "error":
                        err = msg.get("error", msg)
                        cause = err.get("event_id") if isinstance(err, dict) else None
                        if str(cause or "").startswith(TRUNCATE_ID):
                            # a refused truncation does not spoil the recording
                            tl.note(at, "error", event_id=cause,
                                    code=err.get("code"))
                            continue
                        raise RuntimeError(f"{self.name}: server error: {err}")
                    elif mtype == "response.done":
                        resp = msg.get("response") or {}
                        tl.note(at, mtype, response_id=resp.get("id"),
                                status=resp.get("status"),
                                reason=(resp.get("status_details") or {}).get("reason"))
                        if send_done[0]:
                            return "response_done"
                    elif not mtype.endswith(".delta"):
                        tl.note(at, mtype,
                                **{k: msg.get(k) for k in _NOTE_KEYS})

            await asyncio.gather(send_audio(), recv_audio())

        # lay the received audio out on the time axis
        n = int((total_dur + self.silence_after_s) * WIRE_SR)
        max_samples = None if playout else n    # up to 0.2.0: cut at n
        out = tl.render(self.rule, min_samples=n, max_samples=max_samples)
        events = tl.record(self.timeline, self.rule, min_samples=n,
                           max_samples=max_samples)
        return out, tl, events
