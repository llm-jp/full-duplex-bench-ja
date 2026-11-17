"""Moshi-style WebSocket adapter.

Streams speaker-A audio to a running Moshi-family server
(``uv run -m moshi.server --hf-repo ...``) and records the model response.
Works with J-Moshi and LLM-jp-Moshi-v1. Ported from the research client
``moshi_inference.py``.

Requires ``opuslib`` and ``websockets`` (install extra ``[infer]``). Needs a
running server; not exercised in CI without one.

Timeline (``timeline`` option, see ``fdbja.adapters._timeline``):

``playout`` (default)
    The response channel is what a real client plays: each decoded frame is
    placed at the time it arrived, or after the frames queued before it,
    whichever is later. This is the same rule as in the commercial-API
    adapters (``gemini_live``, ``openai_realtime``), so response latencies are
    measured on the same footing. The protocol has no interruption notice, so
    nothing is discarded.
``legacy``
    The construction up to 0.2.0: the received frames are joined in the order
    they arrive, starting at time 0, without using arrival times. Use it to
    reproduce values published with 0.2.0.

Each clip also returns an event record (arrival time and sample count of every
frame) and the received audio, which ``fdbja infer`` saves as ``events.json`` /
``received.wav`` next to ``combined.wav``.
"""

from __future__ import annotations

import asyncio
import time

import numpy as np

from fdbja.adapters._oggopus import OggDemuxer, OggOpusMuxer
from fdbja.adapters._timeline import (
    DEFAULT_TIMELINE,
    ResponseTimeline,
    resolve_rule,
)
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
                 trailing_silence_s: float = 3.0,
                 timeline: str = DEFAULT_TIMELINE, **_):
        self.url = url
        self.timeline = timeline
        self.rule = resolve_rule(timeline, legacy_rule="concat")
        self.trailing_silence_s = trailing_silence_s
        self.sample_rate = MOSHI_SR

    def infer(self, input_audio: np.ndarray, sample_rate: int) -> DialogueResponse:
        input_pcm = _to_24k_i16(input_audio, sample_rate)
        output_pcm, tl, events = asyncio.run(self._stream(input_pcm))
        return DialogueResponse(
            input_audio=input_pcm.astype(np.float32) / 32768.0,
            response_audio=output_pcm.astype(np.float32) / 32768.0,
            sample_rate=MOSHI_SR,
            events=events,
            received_audio=tl.received_audio().astype(np.float32) / 32768.0,
        )

    async def _stream(self, input_pcm: np.ndarray):
        import opuslib
        import websockets

        total = len(input_pcm)
        total_dur = total / MOSHI_SR
        encoder = opuslib.Encoder(MOSHI_SR, 1, opuslib.APPLICATION_AUDIO)
        decoder = opuslib.Decoder(MOSHI_SR, 1)
        muxer, demuxer = OggOpusMuxer(MOSHI_SR, 1), OggDemuxer()
        tl = ResponseTimeline(MOSHI_SR, adapter=self.name)

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
                            at = time.monotonic() - t0
                            demuxer.push(msg[1:])
                            for pkt in demuxer.get_packets():
                                for fsize in decode_sizes:
                                    try:
                                        dec = decoder.decode(pkt, fsize)
                                        tl.add_audio(at, np.frombuffer(
                                            dec, dtype=np.int16))
                                        break
                                    except Exception:
                                        continue
                except (asyncio.TimeoutError,
                        websockets.exceptions.ConnectionClosed):
                    pass

            await asyncio.gather(send_audio(), recv_audio())

        if self.rule == "playout":
            n = int((total_dur + self.trailing_silence_s) * MOSHI_SR)
        else:
            n = 0       # up to 0.2.0: exactly the received frames
        out = tl.render(self.rule, min_samples=n)
        if len(out) == 0:
            out = np.zeros(total, dtype=np.int16)
        return out, tl, tl.record(self.timeline, self.rule, min_samples=n)
