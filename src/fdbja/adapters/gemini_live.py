"""Gemini Live API adapter.

Streams speaker-A audio (16 kHz) to the Gemini Live API and records the
model's 24 kHz audio response, preserving silence gaps by arrival time (the
SynchronizedRecorder convention). Ported from ``gemini_live_inference.py``.

Requires ``google-genai`` (extra ``[gemini]``) and ``GOOGLE_API_KEY``.
"""

from __future__ import annotations

import asyncio
import os
import time

import numpy as np

from fdbja.adapters.base import ModelAdapter, DialogueResponse

INPUT_SR = 16000
OUTPUT_SR = 24000
CHUNK_MS = 100
CHUNK_SAMPLES = int(INPUT_SR * CHUNK_MS / 1000)

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
                 api_key: str | None = None, **_):
        self.model = model
        self.voice = voice
        self.silence_after_s = silence_after_s
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.sample_rate = OUTPUT_SR
        if not self.api_key:
            raise RuntimeError("GOOGLE_API_KEY not set")

    def infer(self, input_audio: np.ndarray, sample_rate: int) -> DialogueResponse:
        input_pcm = _to_16k_i16(input_audio, sample_rate)
        output_pcm = asyncio.run(self._stream(input_pcm))
        import librosa
        in_f32 = input_pcm.astype(np.float32) / 32768.0
        in_24k = librosa.resample(in_f32, orig_sr=INPUT_SR, target_sr=OUTPUT_SR)
        return DialogueResponse(
            input_audio=in_24k,
            response_audio=output_pcm.astype(np.float32) / 32768.0,
            sample_rate=OUTPUT_SR,
        )

    async def _stream(self, input_pcm: np.ndarray) -> np.ndarray:
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
        received = []  # (arrival_time, pcm)
        send_done = [False]
        t0 = time.monotonic()

        async with client.aio.live.connect(model=self.model, config=config) as session:
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

            async def recv_audio():
                try:
                    async for msg in session.receive():
                        sc = getattr(msg, "server_content", None)
                        if not sc:
                            continue
                        if sc.model_turn:
                            at = time.monotonic() - t0
                            for part in sc.model_turn.parts:
                                if part.inline_data:
                                    received.append((at, np.frombuffer(
                                        part.inline_data.data, dtype=np.int16)))
                        if sc.turn_complete and send_done[0]:
                            break
                except asyncio.TimeoutError:
                    pass

            await asyncio.gather(send_audio(), recv_audio())

        # reconstruct timeline with silence by arrival time
        n = int((total_dur + self.silence_after_s) * OUTPUT_SR)
        out = np.zeros(n, dtype=np.int16)
        cursor = 0
        for at, pcm in received:
            start = max(int(at * OUTPUT_SR), cursor)
            end = start + len(pcm)
            if end > len(out):
                out = np.concatenate([out, np.zeros(end - len(out), dtype=np.int16)])
            out[start:end] = pcm
            cursor = end
        return out
