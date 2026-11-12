"""OpenAI-Realtime-protocol adapter (GPT-Realtime and Grok Voice).

Speaks the OpenAI Realtime WebSocket protocol: paced ``input_audio_buffer.append``
frames in, ``response.output_audio.delta`` frames out. xAI's Grok Voice Agent
API (wss://api.x.ai/v1/realtime) implements the same event schema, so one
adapter covers both; only the ``session.update`` payload differs slightly
(``dialect``: "openai" nests voice/turn_detection under ``audio``, "xai" puts
them at the session top level).

The response timeline is reconstructed by arrival time, preserving silence
gaps (same SynchronizedRecorder convention as the Gemini adapter) — the
timing of the response, not just its content, is what the benchmark scores.

Requires ``websockets`` + ``librosa`` (extra ``[realtime]``) and an API key
(``OPENAI_API_KEY`` / ``XAI_API_KEY`` by default, or ``api_key`` in config).

Known limitation (fairness across adapters, not yet fixed): this adapter
ignores the server's interruption notices (``input_audio_buffer.speech_started``,
``response.cancelled``) and keeps every
received audio chunk on the timeline. A real client stops playback when the
user barges in, so a reply that the API would have cut short is scored here as
continuous speech (e.g. a short start in a Pause can become an utterance over
1.5 s and TO = 1). The timeline is also built from wall-clock arrival times,
whereas ``moshi_ws`` concatenates the received frames in order, so response
latencies (``lat_median``) are not measured on exactly the same footing across
adapters. See README "既知の限界".
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time

import numpy as np

from fdbja.adapters.base import DialogueResponse, ModelAdapter

WIRE_SR = 24000            # both APIs default to 24 kHz PCM16
CHUNK_MS = 100
CHUNK_SAMPLES = int(WIRE_SR * CHUNK_MS / 1000)

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
                 **_):
        self.url = url
        self.model = model
        self.voice = voice
        self.dialect = dialect or ("xai" if "x.ai" in url else "openai")
        self.silence_after_s = silence_after_s
        self.instructions = instructions
        self.max_retries = max_retries
        self.retry_wait_s = retry_wait_s
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
                output_pcm = asyncio.run(self._stream(input_pcm))
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

    async def _stream(self, input_pcm: np.ndarray) -> np.ndarray:
        import websockets

        total = len(input_pcm)
        total_dur = total / WIRE_SR
        deadline = total_dur + self.silence_after_s + 3.0   # recv grace
        received: list[tuple[float, np.ndarray]] = []       # (arrival, pcm)
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
                while True:
                    remaining = deadline - (time.monotonic() - t0)
                    if remaining <= 0:
                        break
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                    except (asyncio.TimeoutError,
                            websockets.exceptions.ConnectionClosed):
                        break
                    msg = json.loads(raw)
                    mtype = msg.get("type", "")
                    if mtype in ("response.output_audio.delta",
                                 "response.audio.delta"):
                        at = time.monotonic() - t0
                        pcm = np.frombuffer(base64.b64decode(msg["delta"]),
                                            dtype=np.int16)
                        received.append((at, pcm))
                    elif mtype == "error":
                        err = msg.get("error", msg)
                        raise RuntimeError(f"{self.name}: server error: {err}")
                    elif mtype == "response.done" and send_done[0]:
                        break

            await asyncio.gather(send_audio(), recv_audio())

        # reconstruct timeline with silence by arrival time
        n = int((total_dur + self.silence_after_s) * WIRE_SR)
        out = np.zeros(n, dtype=np.int16)
        cursor = 0
        for at, pcm in received:
            start = max(int(at * WIRE_SR), cursor)
            end = start + len(pcm)
            if end > len(out):
                out = np.concatenate([out, np.zeros(end - len(out),
                                                    dtype=np.int16)])
            out[start:end] = pcm
            cursor = end
        return out[:n]
