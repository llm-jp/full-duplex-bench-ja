import json
import os
import tempfile
import unittest

import numpy as np

from fdbja import runner
from fdbja.data import system_activity as sa
from fdbja.data.event_boundaries import voice_activity
from fdbja.metrics import tor

SR = 16000


def _signal(segments, total, sr=SR, noise_db=-80.0, seed=0):
    """Mono test signal: tones at -20 dBFS on ``segments`` [(start, end)], noise elsewhere."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(total * sr)) * 10 ** (noise_db / 20)
    t = np.arange(len(x)) / sr
    for s, e in segments:
        m = (t >= s) & (t < e)
        x[m] += 0.1 * np.sqrt(2) * np.sin(2 * np.pi * 220 * t[m])
    return x.astype(np.float32)


def _chars(text, start, end, speaker="B"):
    step = (end - start) / len(text)
    return [{"speaker": speaker, "word": w, "start": round(start + i * step, 3),
             "end": round(start + (i + 1) * step, 3)} for i, w in enumerate(text)]


class SpeechSegmentsTest(unittest.TestCase):
    def test_segments_from_synthetic_speech(self):
        va = voice_activity(_signal([(1.0, 2.0), (2.2, 3.0), (5.0, 6.5)], 8.0), SR)
        segs = sa.speech_segments(va, 0.02, merge_gap=0.3)
        self.assertEqual(len(segs), 2)                      # 0.2 s gap is merged
        self.assertAlmostEqual(segs[0][0], 1.0, delta=0.021)
        self.assertAlmostEqual(segs[0][1], 3.0, delta=0.021)
        self.assertAlmostEqual(segs[1][0], 5.0, delta=0.021)
        self.assertAlmostEqual(segs[1][1], 6.5, delta=0.021)

    def test_silent_channel_has_no_utterance(self):
        va = voice_activity(np.zeros(SR * 3, np.float32), SR)
        utts, stats = sa.utterances_from_activity(_chars("はい", 1.0, 1.4), va)
        self.assertEqual(utts, [])
        self.assertEqual(stats["unassigned_chars"], 2)


class StretchedCharacterTest(unittest.TestCase):
    """The last character of an ASR segment stretched over the trailing
    silence (2.08-5.30 s) must not count as talking through A's pause."""

    def setUp(self):
        # model says "うん" at 2.0-2.4 and "そうですね" at 5.3-6.3; A pauses 3.0-3.8
        self.va = voice_activity(_signal([(2.0, 2.4), (5.3, 6.3)], 8.0), SR)
        self.jchat = ([{"speaker": "B", "word": "う", "start": 2.0, "end": 2.079},
                       {"speaker": "B", "word": "ん", "start": 2.079, "end": 5.304}]
                      + _chars("そうですね", 5.304, 6.3))
        self.pause = [{"start": 3.0, "end": 3.8, "duration": 0.8}]

    def test_transcript_counts_takeover(self):
        from fdbja.data.event_detection import group_chars_to_utterances
        utts = group_chars_to_utterances(self.jchat, "B")
        self.assertEqual(tor.eval_pause(self.pause, utts)[0]["TO"], 1)

    def test_audio_keeps_silence(self):
        utts, stats = sa.utterances_from_activity(self.jchat, self.va)
        self.assertEqual([u["num_chars"] for u in utts], [2, 5])
        self.assertAlmostEqual(utts[0]["end"], 2.4, delta=0.021)
        self.assertEqual(stats["unassigned_chars"], 0)
        self.assertEqual(tor.eval_pause(self.pause, utts)[0]["TO"], 0)

    def test_char_outside_any_segment_is_dropped(self):
        jchat = self.jchat + _chars("あ", 7.2, 7.3)
        utts, stats = sa.utterances_from_activity(jchat, self.va)
        self.assertEqual(stats["unassigned_chars"], 1)
        self.assertEqual(sum(u["num_chars"] for u in utts), 7)


class ConfigTest(unittest.TestCase):
    def test_from_config(self):
        c = sa.SystemActivityConfig.from_config({"margin_db": 12.0, "merge_gap": 0.2})
        self.assertEqual(c.vad.margin_db, 12.0)
        self.assertEqual(c.merge_gap, 0.2)
        with self.assertRaises(ValueError):
            sa.SystemActivityConfig.from_config({"no_such_key": 1})

    def test_runner_default_is_transcript(self):
        gt = _chars("あ" * 10, 0.0, 3.0, "A") + _chars("あ" * 10, 3.6, 8.0, "A")
        asr = _chars("そうですねえ、ほんとに", 3.1, 3.7)
        events = [{"start": 3.0, "end": 3.6, "duration": 0.6}]
        a, b, stats = runner._tor_events("pause", gt, asr, events=events)
        self.assertIsNone(stats)
        self.assertEqual(b[0]["TO"], 1)
        with self.assertRaises(FileNotFoundError):
            runner._tor_events("pause", gt, asr, events=events,
                               system_activity="audio", model_wav="/nonexistent.wav")
        with self.assertRaises(ValueError):
            runner._tor_events("pause", gt, asr, events=events, system_activity="vad")

    def test_runner_audio_reads_model_channel(self):
        import soundfile as sf
        b = _signal([(2.0, 2.4)], 8.0)
        x = np.stack([np.zeros_like(b), b], axis=1)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "combined.wav")
            sf.write(p, x, SR)
            jchat = [{"speaker": "B", "word": "う", "start": 2.0, "end": 2.079},
                     {"speaker": "B", "word": "ん", "start": 2.079, "end": 5.304}]
            events = [{"start": 3.0, "end": 3.8, "duration": 0.8}]
            _, res_t, _ = runner._tor_events("pause", [], jchat, events=events)
            _, res_a, stats = runner._tor_events(
                "pause", [], jchat, events=events, system_activity="audio", model_wav=p)
            self.assertEqual(res_t[0]["TO"], 1)
            self.assertEqual(res_a[0]["TO"], 0)
            self.assertEqual(stats["chars"], 2)
            json.dumps(stats)


if __name__ == "__main__":
    unittest.main()
