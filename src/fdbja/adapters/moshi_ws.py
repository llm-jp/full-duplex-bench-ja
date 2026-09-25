"""Moshi-style WebSocket adapter.

Streams speaker-A audio to a running Moshi-family server
(``uv run -m moshi.server --hf-repo ...``) and records the model response.
Works with J-Moshi and LLM-jp-Moshi-v1. Ported from the research client
``moshi_inference.py``.

Requires ``opuslib`` and ``websockets`` (install extra ``[infer]``). Needs a
running server; not exercised in CI without one.
"""

from __future__ import annotations

import asyncio
import time

import numpy as np

from fdbja.adapters._oggopus import OggDemuxer, OggOpusMuxer
from fdbja.adapters.base import DialogueResponse, ModelAdapter

MOSHI_SR = 24000
FRAME_DURATION_MS = 20
FRAME_SIZE = int(MOSHI_SR * FRAME_DURATION_MS / 1000)  # 480
SEND_INTERVAL_MS = 80


def _to_24k_i16(audio: np.ndarray, sr: int) -> np.ndarray:
    import torch
    import torchaudio.functional as AF
    wav = torch.from_numpy(np.asarray(audio, dtype=np.float32))
    if wav.ndim == 2:
        wav = wav[0]
    if sr != MOSHI_SR:
        wav = AF.resample(wav, sr, MOSHI_SR)
    return np.clip(wav.numpy() * 32768, -32768, 32767).astype(np.int16)


class MoshiWebSocketAdapter(ModelAdapter):
    name = "moshi_ws"

    def __init__(self, url: str = "ws://localhost:8998/api/chat",
                 trailing_silence_s: float = 3.0, **_):
        self.url = url
        self.trailing_silence_s = trailing_silence_s
        self.sample_rate = MOSHI_SR

    def infer(self, input_audio: np.ndarray, sample_rate: int) -> DialogueResponse:
        input_pcm = _to_24k_i16(input_audio, sample_rate)
        output_pcm = asyncio.run(self._stream(input_pcm))
        return DialogueResponse(
            input_audio=input_pcm.astype(np.float32) / 32768.0,
            response_audio=output_pcm.astype(np.float32) / 32768.0,
            sample_rate=MOSHI_SR,
        )

    async def _stream(self, input_pcm: np.ndarray) -> np.ndarray:
        import opuslib
        import websockets

        total = len(input_pcm)
        total_dur = total / MOSHI_SR
        encoder = opuslib.Encoder(MOSHI_SR, 1, opuslib.APPLICATION_AUDIO)
        decoder = opuslib.Decoder(MOSHI_SR, 1)
        muxer, demuxer = OggOpusMuxer(MOSHI_SR, 1), OggDemuxer()
        out_chunks = []

        async with websockets.connect(self.url, max_size=10 * 1024 * 1024) as ws:
            await ws.recv()  # handshake
            await ws.send(bytes([1]) + muxer.build_id_header())
            await ws.send(bytes([1]) + muxer.build_tags_header())

            send_pos = 0
            frames_per_send = SEND_INTERVAL_MS // FRAME_DURATION_MS
            t0 = time.monotonic()

            async def send_audio():
                nonlocal send_pos
                while send_pos < total:
                    audio_t = send_pos / MOSHI_SR
                    dt = audio_t - (time.monotonic() - t0)
                    if dt > 0:
                        await asyncio.sleep(dt)
                    for _ in range(frames_per_send):
                        frame = np.zeros(FRAME_SIZE, dtype=np.int16)
                        if send_pos < total:
                            end = min(send_pos + FRAME_SIZE, total)
                            frame[:end - send_pos] = input_pcm[send_pos:end]
                            send_pos = end
                        pkt = encoder.encode(frame.tobytes(), FRAME_SIZE)
                        await ws.send(bytes([1]) + muxer.build_audio_page(pkt, 960))
                n_sil = int(self.trailing_silence_s * 1000 / FRAME_DURATION_MS)
                for i in range(n_sil):
                    target = total_dur + i * FRAME_DURATION_MS / 1000
                    dt = target - (time.monotonic() - t0)
                    if dt > 0:
                        await asyncio.sleep(dt)
                    frame = np.zeros(FRAME_SIZE, dtype=np.int16)
                    pkt = encoder.encode(frame.tobytes(), FRAME_SIZE)
                    await ws.send(bytes([1]) + muxer.build_audio_page(pkt, 960))

            async def recv_audio():
                decode_sizes = [480, 960, 1920, 3840]
                try:
                    while True:
                        msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                        if not (isinstance(msg, bytes) and msg):
                            continue
                        if msg[0] == 1:
                            demuxer.push(msg[1:])
                            for pkt in demuxer.get_packets():
                                for fsize in decode_sizes:
                                    try:
                                        dec = decoder.decode(pkt, fsize)
                                        out_chunks.append(
                                            np.frombuffer(dec, dtype=np.int16))
                                        break
                                    except Exception:
                                        continue
                except (asyncio.TimeoutError,
                        websockets.exceptions.ConnectionClosed):
                    pass

            await asyncio.gather(send_audio(), recv_audio())

        if out_chunks:
            return np.concatenate(out_chunks)
        return np.zeros(total, dtype=np.int16)
