"""Gemini Live API adapter.

Streams speaker-A audio (16 kHz) to the Gemini Live API and records the
model's 24 kHz audio response. Ported from ``gemini_live_inference.py``.

Requires ``google-genai`` (extra ``[gemini]``) and ``GOOGLE_API_KEY``.

Timeline (``timeline`` option, see ``fdbja.adapters._timeline``):

``playout`` (default)
    The response channel is what a real client plays. Audio is queued by
    arrival time (the API generates it as fast as it can, not in real time).
    When a message has ``server_content.interrupted`` set, playback stops at
    the time the message arrived and the audio still waiting in the queue is
    discarded, which is what the API reference tells a client that plays in
    real time to do. The adapter keeps receiving until the input and the
    trailing silence have been sent, so every model turn in the clip is
    recorded. The clock starts when the session is open, as in the other
    adapters.
``legacy``
    The construction up to 0.2.0, for reproducing published values. It differs
    in three ways: ``interrupted`` is ignored and every received chunk stays
    on the timeline; receiving stops at the first ``turn_complete`` (the SDK's
    ``session.receive()`` ends there), so later model turns in the same clip
    are not recorded; and the clock starts before the connection is opened.

Each clip also returns an event record (arrival time and sample count of every
chunk, time of every notice) and the received audio, which ``fdbja infer``
saves as ``events.json`` / ``received.wav`` next to ``combined.wav``.
"""

from __future__ import annotations

import asyncio
import os
import time

import numpy as np

from fdbja.adapters._timeline import (
    DEFAULT_TIMELINE,
    ResponseTimeline,
    resolve_rule,
)
from fdbja.adapters.base import DialogueResponse, ModelAdapter

INPUT_SR = 16000
OUTPUT_SR = 24000
CHUNK_MS = 100
CHUNK_SAMPLES = int(INPUT_SR * CHUNK_MS / 1000)
RECV_GRACE_S = 3.0      # keep receiving this long after the last input chunk

SYSTEM_INSTRUCTION = (
    "あなたは日本語の対話相手です。自然に日本語で応答してください。"
    "相槌（「うん」「はい」「なるほど」など）も適切に使ってください。"
    "必ず日本語のみで応答してください。"
)


def _to_16k_i16(audio: np.ndarray, sr: int) -> np.ndarray:
    a = np.asarray(audio)
    if a.ndim == 2:
        a = a[0]
    if a.dtype != np.float32:
        a = a.astype(np.float32)
    if sr != INPUT_SR:
        import librosa
        a = librosa.resample(a, orig_sr=sr, target_sr=INPUT_SR)
    return np.clip(a * 32768, -32768, 32767).astype(np.int16)


class GeminiLiveAdapter(ModelAdapter):
    name = "gemini_live"

    def __init__(self, model: str = "gemini-3.1-flash-live-preview",
                 voice: str = "Aoede", silence_after_s: float = 5.0,
                 api_key: str | None = None,
                 timeline: str = DEFAULT_TIMELINE, **_):
        self.model = model
        self.timeline = timeline
        self.rule = resolve_rule(timeline, legacy_rule="arrival")
        self.voice = voice
        self.silence_after_s = silence_after_s
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.sample_rate = OUTPUT_SR
        if not self.api_key:
            raise RuntimeError("GOOGLE_API_KEY not set")

    def infer(self, input_audio: np.ndarray, sample_rate: int) -> DialogueResponse:
        input_pcm = _to_16k_i16(input_audio, sample_rate)
        output_pcm, tl, events = asyncio.run(self._stream(input_pcm))
        import librosa
        in_f32 = input_pcm.astype(np.float32) / 32768.0
        in_24k = librosa.resample(in_f32, orig_sr=INPUT_SR, target_sr=OUTPUT_SR)
        return DialogueResponse(
            input_audio=in_24k,
            response_audio=output_pcm.astype(np.float32) / 32768.0,
            sample_rate=OUTPUT_SR,
            events=events,
            received_audio=tl.received_audio().astype(np.float32) / 32768.0,
        )

    async def _stream(self, input_pcm: np.ndarray):
        from google import genai
        from google.genai import types

        total = len(input_pcm)
        total_dur = total / INPUT_SR
        client = genai.Client(api_key=self.api_key)
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=self.voice)),
                language_code="ja-JP"),
            system_instruction=SYSTEM_INSTRUCTION,
        )
        tl = ResponseTimeline(OUTPUT_SR, adapter=self.name)
        playout = self.rule == "playout"
        send_done = [False]
        t0 = time.monotonic()   # up to 0.2.0 the clock started here

        async with client.aio.live.connect(model=self.model, config=config) as session:
            if playout:
                t0 = time.monotonic()   # session is open: same as other adapters

            async def send_audio():
                for i in range(0, total, CHUNK_SAMPLES):
                    dt = i / INPUT_SR - (time.monotonic() - t0)
                    if dt > 0:
                        await asyncio.sleep(dt)
                    chunk = input_pcm[i:i + CHUNK_SAMPLES]
                    if len(chunk) < CHUNK_SAMPLES:
                        chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
                    await session.send_realtime_input(audio=types.Blob(
                        data=chunk.tobytes(), mime_type=f"audio/pcm;rate={INPUT_SR}"))
                n_sil = int(self.silence_after_s * 1000 / CHUNK_MS)
                silence = np.zeros(CHUNK_SAMPLES, dtype=np.int16)
                for j in range(n_sil):
                    dt = total_dur + j * CHUNK_MS / 1000 - (time.monotonic() - t0)
                    if dt > 0:
                        await asyncio.sleep(dt)
                    await session.send_realtime_input(audio=types.Blob(
                        data=silence.tobytes(), mime_type=f"audio/pcm;rate={INPUT_SR}"))
                send_done[0] = True

            def handle(msg) -> bool:
                """Record one server message. True at the end of a turn."""
                sc = getattr(msg, "server_content", None)
                if not sc:
                    return False
                at = time.monotonic() - t0
                if sc.model_turn:
                    for part in sc.model_turn.parts:
                        if part.inline_data:
                            tl.add_audio(at, np.frombuffer(
                                part.inline_data.data, dtype=np.int16))
                if getattr(sc, "interrupted", None):
                    # the user barged in: stop playback, drop the queue
                    tl.interrupt(at, source="server_content.interrupted")
                if getattr(sc, "generation_complete", None):
                    tl.note(at, "server_content.generation_complete")
                if sc.turn_complete:
                    tl.note(at, "server_content.turn_complete")
                return bool(sc.turn_complete)

            async def recv_audio():
                # session.receive() ends after each completed turn
                while True:
                    n_msgs = 0
                    async for msg in session.receive():
                        n_msgs += 1
                        if handle(msg) and send_done[0]:
                            return
                    if not playout or send_done[0] or n_msgs == 0:
                        return

            if playout:
                await _run_until(send_audio(), recv_audio(), t0 + total_dur
                                 + self.silence_after_s + RECV_GRACE_S)
            else:
                await asyncio.gather(send_audio(), recv_audio())

        # lay the received audio out on the time axis
        n = int((total_dur + self.silence_after_s) * OUTPUT_SR)
        out = tl.render(self.rule, min_samples=n)
        events = tl.record(self.timeline, self.rule, min_samples=n)
        return out, tl, events


async def _run_until(send, recv, deadline: float) -> None:
    """Run both coroutines; stop whatever is still running at ``deadline``
    (``time.monotonic()``). An error in either one is raised."""
    tasks = [asyncio.ensure_future(send), asyncio.ensure_future(recv)]
    try:
        done, _ = await asyncio.wait(
            tasks, timeout=max(0.0, deadline - time.monotonic()),
            return_when=asyncio.FIRST_EXCEPTION)
        for task in done:
            task.result()
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
