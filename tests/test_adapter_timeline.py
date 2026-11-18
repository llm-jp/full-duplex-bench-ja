"""Response timeline of the streaming adapters: playback queue, interruption
notices, the 0.2.0 construction (``timeline: legacy``) and the event record.

The adapters are run against scripted fake sessions (no network, no model)."""

import asyncio
import base64
import json
import os
import sys
import time
import types

import numpy as np
import pytest
import soundfile as sf

from fdbja.adapters import _timeline as tlm
from fdbja.adapters._oggopus import OggOpusMuxer
from fdbja.adapters._timeline import (
    ResponseTimeline,
    build_timeline,
    resolve_rule,
    timeline_from_record,
)
from fdbja.adapters.base import DialogueResponse
from fdbja.runner import _write_outputs

SR = 24000


def _tone(seconds, value):
    return np.full(int(seconds * SR), value, dtype=np.int16)


def _span(x):
    """(first, last + 1) sample index of the non-zero part, or None."""
    nz = np.flatnonzero(x)
    return (int(nz[0]), int(nz[-1]) + 1) if len(nz) else None


# --- the rules themselves ------------------------------------------------

def test_chunk_is_placed_at_its_arrival_time():
    tl = ResponseTimeline(SR)
    tl.add_audio(0.5, _tone(0.2, 7))
    for rule in ("playout", "arrival"):
        out = tl.render(rule, min_samples=SR)
        assert len(out) == SR
        assert _span(out) == (int(0.5 * SR), int(0.7 * SR))
    assert _span(tl.render("concat")) == (0, int(0.2 * SR))


def test_audio_arriving_faster_than_real_time_is_queued():
    # 1 s of audio arrives within 40 ms: it plays back to back, in real time
    tl = ResponseTimeline(SR)
    for k in range(5):
        tl.add_audio(0.5 + 0.01 * k, _tone(0.2, k + 1), item="a")
    out = tl.render("playout")
    assert _span(out) == (int(0.5 * SR), int(1.5 * SR))
    assert out[int(0.5 * SR) + int(0.2 * SR) * 4] == 5      # fifth chunk, in order
    assert np.array_equal(out, tl.render("arrival"))        # no notice: same thing


def test_interruption_discards_what_was_not_played_yet():
    tl = ResponseTimeline(SR)
    for k in range(5):
        tl.add_audio(0.5 + 0.01 * k, _tone(0.2, k + 1), item="a")
    cut = tl.interrupt(0.9, source="notice")
    assert cut == {"a": 400}                                # 0.5 s .. 0.9 s was heard
    assert _span(tl.render("playout")) == (int(0.5 * SR), int(0.9 * SR))
    # the 0.2.0 rule keeps everything
    assert _span(tl.render("arrival")) == (int(0.5 * SR), int(1.5 * SR))


def test_audio_of_the_interrupted_response_that_arrives_late_is_not_played():
    tl = ResponseTimeline(SR)
    tl.add_audio(0.5, _tone(1.0, 1), item="a")
    tl.interrupt(0.9, source="notice")
    tl.add_audio(0.95, _tone(0.2, 2), item="a")             # still in flight
    tl.add_audio(1.2, _tone(0.1, 3), item="b")              # the next response
    out = tl.render("playout")
    assert not np.any(out == 2)
    assert not np.any(out[int(0.9 * SR):int(1.2 * SR)])
    assert _span(out == 3) == (int(1.2 * SR), int(1.3 * SR))    # not behind the old queue
    assert tl.events[2]["after_interruption"] is True
    # the 0.2.0 rule: late chunk and next response queue behind the first one
    old = tl.render("arrival")
    assert _span(old == 2) == (int(1.5 * SR), int(1.7 * SR))
    assert _span(old == 3) == (int(1.7 * SR), int(1.8 * SR))


def test_several_notices_and_notices_with_nothing_queued():
    tl = ResponseTimeline(SR)
    assert tl.interrupt(0.1, source="notice") == {}         # nothing received yet
    tl.add_audio(0.2, _tone(1.0, 1), item="a")
    assert tl.interrupt(0.5, source="notice") == {"a": 300}
    assert tl.interrupt(0.6, source="notice") == {}         # queue already empty
    tl.add_audio(1.0, _tone(0.1, 2), item="b")              # fully played
    tl.add_audio(1.5, _tone(1.0, 3), item="c")
    tl.add_audio(1.5, _tone(1.0, 4), item="d")              # waits behind c
    assert tl.interrupt(1.8, source="notice") == {"c": 300, "d": 0}
    out = tl.render("playout")
    assert _span(out == 1) == (int(0.2 * SR), int(0.5 * SR))
    assert _span(out == 2) == (int(1.0 * SR), int(1.1 * SR))
    assert _span(out == 3) == (int(1.5 * SR), int(1.8 * SR))
    assert not np.any(out == 4)
    assert _span(out)[1] == int(1.8 * SR)


def test_record_rebuilds_every_rule_without_the_model():
    tl = ResponseTimeline(SR, adapter="x")
    tl.add_audio(0.5, _tone(1.0, 1), item="a", response="r1")
    tl.interrupt(0.9, source="notice", audio_start_ms=800)
    tl.note(0.91, "response.done", status="cancelled", reason=None)
    tl.add_audio(1.2, _tone(0.1, 3), item="b")
    n = 2 * SR
    record = json.loads(json.dumps(tl.record("playout", "playout", min_samples=n)))
    received = tl.received_audio()
    assert record["received_samples"] == len(received) == int(1.1 * SR)
    assert [e["type"] for e in record["events"]] == ["audio", "interruption", "note", "audio"]
    assert record["events"][1]["audio_start_ms"] == 800
    assert "reason" not in record["events"][2]              # empty fields are dropped
    assert np.array_equal(timeline_from_record(record, received), tl.render("playout", n))
    for rule in ("arrival", "concat"):
        assert np.array_equal(timeline_from_record(record, received, rule),
                              tl.render(rule, n))
    with pytest.raises(ValueError):
        timeline_from_record({"format": "other"}, received)


def test_rule_names():
    assert resolve_rule("playout", "concat") == "playout"
    assert resolve_rule("legacy", "concat") == "concat"
    assert resolve_rule("legacy", "arrival") == "arrival"
    with pytest.raises(ValueError):
        resolve_rule("fast", "arrival")
    with pytest.raises(ValueError):
        build_timeline([], np.zeros(0, dtype=np.int16), SR, rule="fast")
    assert tlm.DEFAULT_TIMELINE in tlm.TIMELINE_OPTIONS


# --- saving ---------------------------------------------------------------

def test_infer_saves_the_record_and_the_received_audio(tmp_path):
    tl = ResponseTimeline(SR, adapter="x")
    tl.add_audio(0.1, _tone(0.5, 1000), item="a")
    tl.interrupt(0.3, source="notice")
    n = SR
    out = tl.render("playout", n)
    resp = DialogueResponse(
        input_audio=np.zeros(n, dtype=np.float32),
        response_audio=out.astype(np.float32) / 32768.0, sample_rate=SR,
        events=tl.record("playout", "playout", min_samples=n),
        received_audio=tl.received_audio().astype(np.float32) / 32768.0)
    combined = tmp_path / "clip" / "combined.wav"
    _write_outputs(str(combined), resp, sf)
    d = combined.parent
    assert sorted(os.listdir(d)) == ["combined.wav", "events.json", "input.wav",
                                     "output.wav", "received.wav"]
    record = tlm.load_record(str(d / "events.json"))
    received, sr = sf.read(str(d / "received.wav"), dtype="int16")
    saved, _ = sf.read(str(d / "output.wav"), dtype="int16")
    assert sr == SR and len(received) == int(0.5 * SR)
    assert np.array_equal(timeline_from_record(record, received), saved)
    # the 0.2.0 timeline can be made from the same files
    assert _span(timeline_from_record(record, received, "arrival")) == (
        int(0.1 * SR), int(0.6 * SR))


def test_adapters_without_a_record_write_the_usual_files(tmp_path):
    resp = DialogueResponse(np.zeros(100, dtype=np.float32),
                            np.zeros(100, dtype=np.float32), SR)
    combined = tmp_path / "clip" / "combined.wav"
    _write_outputs(str(combined), resp, sf)
    assert sorted(os.listdir(combined.parent)) == ["combined.wav", "input.wav",
                                                   "output.wav"]


# --- fake sessions ----------------------------------------------------------

class _Script:
    """Server messages with the time (seconds after the session opened) at
    which each one is delivered."""

    def __init__(self, items):
        self.items = list(items)
        self.t0 = time.monotonic()
        self.sent = []

    async def next(self):
        if not self.items:
            await asyncio.sleep(3600)       # nothing more: wait to be stopped
        at, msg = self.items.pop(0)
        wait = at - (time.monotonic() - self.t0)
        if wait > 0:
            await asyncio.sleep(wait)
        return msg


def _arrivals(record, kind):
    return [e for e in record["events"] if e["type"] == kind]


# --- openai_realtime ---------------------------------------------------------

def _fake_websockets(script_items, monkeypatch):
    state = {}

    class ConnectionClosed(Exception):
        pass

    class _WS:
        def __init__(self):
            self.script = _Script(script_items)
            state["ws"] = self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def send(self, raw):
            self.script.sent.append(raw)

        async def recv(self):
            msg = await self.script.next()
            if msg is ConnectionClosed:
                raise ConnectionClosed()
            return msg

    mod = types.ModuleType("websockets")
    mod.connect = lambda *a, **k: _WS()
    mod.exceptions = types.SimpleNamespace(
        ConnectionClosed=ConnectionClosed, ConnectionClosedError=ConnectionClosed,
        InvalidStatus=type("InvalidStatus", (Exception,), {}))
    monkeypatch.setitem(sys.modules, "websockets", mod)
    return state, ConnectionClosed


def _delta(item, seconds, value, response="r"):
    return json.dumps({
        "type": "response.output_audio.delta", "item_id": item, "response_id": response,
        "content_index": 0,
        "delta": base64.b64encode(_tone(seconds, value).tobytes()).decode()})


def _realtime_script(extra=()):
    return [
        (0.10, _delta("item_a", 0.5, 1000)),
        (0.11, _delta("item_a", 0.5, 1000)),             # 1 s of audio by 0.11 s
        (0.30, json.dumps({"type": "input_audio_buffer.speech_started",
                           "audio_start_ms": 250, "item_id": "user_1"})),
        *extra,
        (0.32, json.dumps({"type": "response.done", "response": {
            "id": "r", "status": "cancelled",
            "status_details": {"type": "cancelled", "reason": "turn_detected"}}})),
        (0.33, _delta("item_a", 0.2, 3000)),             # late audio of the cancelled reply
        (0.50, _delta("item_b", 0.1, 2000, response="r2")),
        (0.95, json.dumps({"type": "response.done",
                           "response": {"id": "r2", "status": "completed"}})),
    ]


def _run_realtime(monkeypatch, script, **kw):
    from fdbja.adapters.openai_realtime import OpenAIRealtimeAdapter
    state, _ = _fake_websockets(script, monkeypatch)
    adapter = OpenAIRealtimeAdapter(api_key="test", silence_after_s=0.4, max_retries=0, **kw)
    resp = adapter.infer(np.zeros(int(0.4 * SR), dtype=np.float32), SR)
    sent = [json.loads(m) for m in state["ws"].script.sent]
    return resp, sent


def test_realtime_stops_playback_and_truncates_on_speech_started(monkeypatch):
    resp, sent = _run_realtime(monkeypatch, _realtime_script())
    rec = resp.events
    out = np.round(resp.response_audio * 32768).astype(np.int16)
    audio = _arrivals(rec, "audio")
    (notice,) = _arrivals(rec, "interruption")
    assert rec["timeline"] == rec["rule"] == "playout"
    assert notice["source"] == "input_audio_buffer.speech_started"
    assert notice["audio_start_ms"] == 250
    first, cut_at = int(audio[0]["t"] * SR), int(notice["t"] * SR)
    b_at = int(audio[3]["t"] * SR)
    # the first reply is heard from its arrival until the notice, and no further
    assert _span(out == 1000) == (first, cut_at)
    assert not np.any(out[cut_at:b_at])
    assert not np.any(out == 3000) and audio[2]["after_interruption"] is True
    assert _span(out == 2000) == (b_at, b_at + int(0.1 * SR))
    # the server is told how much was played, as the API guide asks
    truncates = [m for m in sent if m["type"] == "conversation.item.truncate"]
    assert truncates == [{"type": "conversation.item.truncate",
                          "event_id": "fdbja_truncate_1", "item_id": "item_a",
                          "content_index": 0,
                          "audio_end_ms": (cut_at - first) * 1000 // SR}]
    done = [e for e in rec["events"] if e.get("event") == "response.done"]
    assert [(e["status"], e.get("reason")) for e in done] == [
        ("cancelled", "turn_detected"), ("completed", None)]
    # the record and the received audio rebuild the same channel
    received = np.round(resp.received_audio * 32768).astype(np.int16)
    assert len(received) == int(1.3 * SR)
    assert np.array_equal(timeline_from_record(rec, received), out)
    assert not any("delta" in e.get("event", "") for e in rec["events"])


def test_realtime_legacy_keeps_everything_and_sends_nothing_back(monkeypatch):
    resp, sent = _run_realtime(monkeypatch, _realtime_script(), timeline="legacy")
    rec = resp.events
    out = np.round(resp.response_audio * 32768).astype(np.int16)
    assert rec["timeline"] == "legacy" and rec["rule"] == "arrival"
    assert not [m for m in sent if m["type"] == "conversation.item.truncate"]
    n = int(0.8 * SR)
    assert len(out) == n == rec["max_samples"]           # cut at input + trailing silence
    first = int(_arrivals(rec, "audio")[0]["t"] * SR)
    assert _span(out == 1000) == (first, n)              # plays through the notice
    # the notice is still on record, so the playout timeline can be made later
    assert len(_arrivals(rec, "interruption")) == 1
    received = np.round(resp.received_audio * 32768).astype(np.int16)
    again = timeline_from_record(rec, received, "playout")
    assert _span(again == 1000) == (first, int(_arrivals(rec, "interruption")[0]["t"] * SR))


def test_realtime_without_truncation_message(monkeypatch):
    resp, sent = _run_realtime(monkeypatch, _realtime_script(), send_truncate=False)
    assert not [m for m in sent if m["type"] == "conversation.item.truncate"]
    out = np.round(resp.response_audio * 32768).astype(np.int16)
    cut_at = int(_arrivals(resp.events, "interruption")[0]["t"] * SR)
    assert _span(out == 1000)[1] == cut_at               # playback still stops


def test_realtime_refused_truncation_does_not_fail_the_clip(monkeypatch):
    refused = (0.31, json.dumps({"type": "error", "error": {
        "type": "invalid_request_error", "code": "invalid_value",
        "message": "x", "event_id": "fdbja_truncate_1"}}))
    resp, _ = _run_realtime(monkeypatch, _realtime_script(extra=[refused]))
    errors = [e for e in resp.events["events"] if e.get("event") == "error"]
    assert errors and errors[0]["event_id"] == "fdbja_truncate_1"
    other = (0.31, json.dumps({"type": "error", "error": {
        "type": "server_error", "message": "x"}}))
    with pytest.raises(RuntimeError):
        _run_realtime(monkeypatch, _realtime_script(extra=[other]))


def test_realtime_no_interruption_is_the_same_under_both_options(monkeypatch):
    script = [(0.10, _delta("item_a", 0.3, 1000)),
              (0.95, json.dumps({"type": "response.done",
                                 "response": {"id": "r", "status": "completed"}}))]
    new, sent = _run_realtime(monkeypatch, list(script))
    assert not [m for m in sent if m["type"] == "conversation.item.truncate"]
    out = np.round(new.response_audio * 32768).astype(np.int16)
    first = int(_arrivals(new.events, "audio")[0]["t"] * SR)
    assert _span(out) == (first, first + int(0.3 * SR))
    received = np.round(new.received_audio * 32768).astype(np.int16)
    assert np.array_equal(timeline_from_record(new.events, received, "arrival"), out)


# --- gemini_live ---------------------------------------------------------------

def _fake_genai(script_items, monkeypatch):
    state = {}

    class _Session:
        def __init__(self):
            self.script = _Script(script_items)
            state["session"] = self

        async def send_realtime_input(self, audio):
            self.script.sent.append(audio)

        async def receive(self):
            # like the SDK: one call yields the messages of one turn
            while True:
                msg = await self.script.next()
                yield msg
                if msg.server_content.turn_complete:
                    return

    class _Connect:
        async def __aenter__(self):
            await asyncio.sleep(0.15)       # opening the session takes time
            return _Session()

        async def __aexit__(self, *a):
            return False

    ns = types.SimpleNamespace
    genai = types.ModuleType("google.genai")
    genai.Client = lambda api_key=None: ns(aio=ns(live=ns(connect=lambda **k: _Connect())))
    gtypes = types.ModuleType("google.genai.types")
    for name in ("LiveConnectConfig", "SpeechConfig", "VoiceConfig",
                 "PrebuiltVoiceConfig", "Blob"):
        setattr(gtypes, name, lambda **k: ns(**k))
    genai.types = gtypes
    google = types.ModuleType("google")
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", gtypes)
    resampler = types.ModuleType("librosa")
    resampler.resample = lambda a, orig_sr, target_sr: np.repeat(a, 3)[::2].copy()
    monkeypatch.setitem(sys.modules, "librosa", resampler)
    return state


def _gemini_msg(audio=None, interrupted=None, turn_complete=None):
    ns = types.SimpleNamespace
    turn = None
    if audio is not None:
        turn = ns(parts=[ns(inline_data=ns(data=audio.tobytes()))])
    return ns(server_content=ns(model_turn=turn, interrupted=interrupted,
                                turn_complete=turn_complete, generation_complete=None))


def _gemini_script():
    return [
        (0.10, _gemini_msg(audio=_tone(1.0, 1000))),     # generated faster than real time
        (0.30, _gemini_msg(interrupted=True)),
        (0.31, _gemini_msg(turn_complete=True)),
        (0.50, _gemini_msg(audio=_tone(0.1, 2000))),     # second turn
        (0.55, _gemini_msg(turn_complete=True)),
        (0.95, _gemini_msg(turn_complete=True)),         # after the input has ended
    ]


def _run_gemini(monkeypatch, script, **kw):
    from fdbja.adapters.gemini_live import GeminiLiveAdapter
    _fake_genai(script, monkeypatch)
    adapter = GeminiLiveAdapter(api_key="test", silence_after_s=0.4, **kw)
    resp = adapter.infer(np.zeros(int(0.4 * 16000), dtype=np.float32), 16000)
    return resp, np.round(resp.response_audio * 32768).astype(np.int16)


def test_gemini_stops_playback_on_interrupted_and_records_every_turn(monkeypatch):
    resp, out = _run_gemini(monkeypatch, _gemini_script())
    rec = resp.events
    audio = _arrivals(rec, "audio")
    (notice,) = _arrivals(rec, "interruption")
    assert notice["source"] == "server_content.interrupted"
    first, cut_at = int(audio[0]["t"] * SR), int(notice["t"] * SR)
    assert _span(out == 1000) == (first, cut_at)
    second = int(audio[1]["t"] * SR)
    assert _span(out == 2000) == (second, second + int(0.1 * SR))
    assert not np.any(out[cut_at:second])
    assert len(out) == int(0.8 * SR)
    # the clock starts when the session is open, not before
    assert abs(audio[0]["t"] - 0.10) < 0.08
    turns = [e for e in rec["events"] if e.get("event") == "server_content.turn_complete"]
    assert len(turns) == 3
    assert len(resp.input_audio) == int(0.4 * SR)


def test_gemini_legacy_reproduces_the_previous_construction(monkeypatch):
    resp, out = _run_gemini(monkeypatch, _gemini_script(), timeline="legacy")
    rec = resp.events
    assert rec["timeline"] == "legacy" and rec["rule"] == "arrival"
    audio = _arrivals(rec, "audio")
    # receiving stopped at the first completed turn: the second reply is missing
    assert len(audio) == 1 and not np.any(out == 2000)
    first = int(audio[0]["t"] * SR)
    assert _span(out == 1000) == (first, first + SR)      # plays through the notice
    # the clock started before the session was open (0.15 s in this fake)
    assert audio[0]["t"] > 0.2


def test_gemini_stops_waiting_after_the_grace_period(monkeypatch):
    from fdbja.adapters import gemini_live
    monkeypatch.setattr(gemini_live, "RECV_GRACE_S", 0.2)
    script = [(0.10, _gemini_msg(audio=_tone(0.2, 1000))),
              (0.12, _gemini_msg(turn_complete=True))]    # nothing after that
    t = time.monotonic()
    resp, out = _run_gemini(monkeypatch, script)
    assert time.monotonic() - t < 2.0
    assert _span(out)[1] - _span(out)[0] == int(0.2 * SR)


# --- moshi_ws -------------------------------------------------------------------

def _fake_moshi(frame_times, monkeypatch):
    from fdbja.adapters import moshi_ws
    items = []
    _, closed = _fake_websockets(items, monkeypatch)
    muxer = OggOpusMuxer(SR, 1)
    items += [(0.0, b"\x00"),                               # handshake
             (0.0, bytes([1]) + muxer.build_id_header()),
             (0.0, bytes([1]) + muxer.build_tags_header())]
    for k, at in enumerate(frame_times):
        pcm = np.full(1920, 100 * (k + 1), dtype=np.int16)         # 80 ms
        items.append((at, bytes([1]) + muxer.build_audio_page(pcm.tobytes(), 3840)))
    items.append((frame_times[-1] + 0.5, closed))

    class _Codec:
        def __init__(self, *a):
            pass

        def encode(self, pcm, frame_size):
            return b"\x00" * 8

        def decode(self, packet, frame_size):
            if len(packet) != frame_size * 2:
                raise ValueError("wrong frame size")
            return packet

    codec = types.ModuleType("opuslib")
    codec.Encoder = codec.Decoder = _Codec
    codec.APPLICATION_AUDIO = 0
    monkeypatch.setitem(sys.modules, "opuslib", codec)
    monkeypatch.setattr(moshi_ws, "_to_24k_i16",
                        lambda a, sr: np.asarray(a * 32768, dtype=np.int16))
    return moshi_ws


def _run_moshi(monkeypatch, frame_times, **kw):
    moshi_ws = _fake_moshi(frame_times, monkeypatch)
    adapter = moshi_ws.MoshiWebSocketAdapter(trailing_silence_s=0.2, **kw)
    resp = adapter.infer(np.zeros(int(0.4 * SR), dtype=np.float32), SR)
    return resp, np.round(resp.response_audio * 32768).astype(np.int16)


FRAME_TIMES = [0.10, 0.18, 0.26, 0.50]      # the fourth frame is late


def test_moshi_frames_are_placed_by_arrival_time(monkeypatch):
    resp, out = _run_moshi(monkeypatch, FRAME_TIMES)
    rec = resp.events
    audio = _arrivals(rec, "audio")
    assert rec["rule"] == "playout" and len(audio) == 4
    assert all(e["n"] == 1920 for e in audio)
    starts = [_span(out == 100 * (k + 1))[0] for k in range(4)]
    assert starts[0] == int(audio[0]["t"] * SR) > 0        # not at time 0
    for k in (1, 2):                                         # on time: back to back or later
        assert starts[k] == max(int(audio[k]["t"] * SR), starts[k - 1] + 1920)
    assert starts[3] == int(audio[3]["t"] * SR) > starts[2] + 1920   # the gap is kept
    assert len(out) >= int(0.6 * SR)


def test_moshi_legacy_joins_the_frames_in_order(monkeypatch):
    resp, out = _run_moshi(monkeypatch, FRAME_TIMES, timeline="legacy")
    assert resp.events["rule"] == "concat"
    assert len(out) == 4 * 1920
    assert [int(out[k * 1920]) for k in range(4)] == [100, 200, 300, 400]
    received = np.round(resp.received_audio * 32768).astype(np.int16)
    assert np.array_equal(received, out)
    # same recording, placed by arrival time
    again = timeline_from_record(resp.events, received, "playout")
    assert _span(again)[0] == int(_arrivals(resp.events, "audio")[0]["t"] * SR)


# --- why listening stopped ---------------------------------------------------

def _receive_end(record):
    (end,) = [e for e in record["events"] if e.get("event") == tlm.RECEIVE_END]
    return end


def test_realtime_records_why_listening_stopped(monkeypatch):
    resp, _ = _run_realtime(monkeypatch, _realtime_script())
    end = _receive_end(resp.events)
    assert end["reason"] == "response_done" and end["t"] >= 0.8    # after the input


def test_realtime_connection_closed_during_the_input_is_on_the_record(monkeypatch):
    from fdbja.adapters.openai_realtime import OpenAIRealtimeAdapter
    items = []
    _, closed = _fake_websockets(items, monkeypatch)
    items += [(0.10, _delta("item_a", 0.1, 1000)), (0.20, closed)]
    adapter = OpenAIRealtimeAdapter(api_key="test", silence_after_s=0.4, max_retries=0)
    resp = adapter.infer(np.zeros(int(0.4 * SR), dtype=np.float32), SR)
    end = _receive_end(resp.events)
    assert end["reason"] == "closed" and end["t"] < 0.4            # before the input ended


def test_gemini_records_why_listening_stopped(monkeypatch):
    from fdbja.adapters import gemini_live
    resp, _ = _run_gemini(monkeypatch, _gemini_script())
    end = _receive_end(resp.events)
    assert end["reason"] == "turn_complete" and end["t"] >= 0.8
    # up to 0.2.0: listening stopped with the first completed turn
    resp, _ = _run_gemini(monkeypatch, _gemini_script(), timeline="legacy")
    end = _receive_end(resp.events)
    assert end["reason"] == "stream_ended"
    assert end["t"] < 0.5 + 0.15                # before the second reply was sent
    # nothing more from the server: stopped by the client after the grace period
    monkeypatch.setattr(gemini_live, "RECV_GRACE_S", 0.2)
    script = [(0.10, _gemini_msg(audio=_tone(0.2, 1000))),
              (0.12, _gemini_msg(turn_complete=True))]
    resp, _ = _run_gemini(monkeypatch, script)
    assert _receive_end(resp.events)["reason"] == "deadline"


def test_moshi_records_why_listening_stopped(monkeypatch):
    resp, _ = _run_moshi(monkeypatch, FRAME_TIMES)
    end = _receive_end(resp.events)
    assert end["reason"] == "closed" and end["t"] >= 0.9


# --- the settings of the session ---------------------------------------------

def _session(record):
    (note,) = [e for e in record["events"] if e.get("event") == tlm.SESSION]
    return note


def test_commercial_adapters_record_model_voice_and_instruction_id(monkeypatch):
    from fdbja.adapters import gemini_live, openai_realtime
    resp, _ = _run_realtime(monkeypatch, _realtime_script())
    realtime = _session(resp.events)
    assert realtime["t"] == 0.0 and realtime["model"] == "gpt-realtime"
    assert realtime["voice"] == "marin"
    assert realtime["instructions"] == tlm.text_id(openai_realtime.SYSTEM_INSTRUCTION)
    resp, _ = _run_gemini(monkeypatch, _gemini_script())
    gemini = _session(resp.events)
    assert gemini["model"] == "gemini-3.1-flash-live-preview" and gemini["voice"] == "Aoede"
    # the two adapters send the same instruction text by default
    assert gemini_live.SYSTEM_INSTRUCTION == openai_realtime.SYSTEM_INSTRUCTION
    assert gemini["instructions"] == realtime["instructions"]
    # the record holds an identifier, not the text
    assert openai_realtime.SYSTEM_INSTRUCTION not in json.dumps(resp.events, ensure_ascii=False)
    # the Moshi-type adapter has no such settings
    resp, _ = _run_moshi(monkeypatch, FRAME_TIMES)
    assert not [e for e in resp.events["events"] if e.get("event") == tlm.SESSION]


def test_gemini_instruction_can_be_replaced(monkeypatch):
    from fdbja.adapters.gemini_live import SYSTEM_INSTRUCTION, GeminiLiveAdapter
    _fake_genai(_gemini_script(), monkeypatch)
    sent = {}
    gtypes = sys.modules["google.genai.types"]
    monkeypatch.setattr(gtypes, "LiveConnectConfig",
                        lambda **k: sent.update(k) or types.SimpleNamespace(**k))
    adapter = GeminiLiveAdapter(api_key="test", silence_after_s=0.4,
                                instructions="日本語で話してください。")
    resp = adapter.infer(np.zeros(int(0.4 * 16000), dtype=np.float32), 16000)
    assert sent["system_instruction"] == "日本語で話してください。"
    assert _session(resp.events)["instructions"] == tlm.text_id("日本語で話してください。")
    assert _session(resp.events)["instructions"] != tlm.text_id(SYSTEM_INSTRUCTION)


def test_record_says_which_version_recorded_it_and_when():
    import re

    import fdbja
    tl = ResponseTimeline(24000, adapter="x")
    record = tl.record("playout", "playout")
    assert record["fdbja"] == fdbja.__version__
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", record["recorded_at"])
