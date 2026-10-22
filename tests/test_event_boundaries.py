import unittest

import numpy as np

from fdbja.data import event_boundaries as eb

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


class VoiceActivityTest(unittest.TestCase):
    def test_threshold_between_noise_and_speech(self):
        x = _signal([(1.0, 3.0), (4.0, 6.0)], 8.0)
        thr = eb.threshold_db(eb.frame_db(x, SR))
        self.assertGreater(thr, -70)
        self.assertLess(thr, -25)

    def test_short_blip_dropped_and_short_hole_filled(self):
        # 40 ms blip in silence, 40 ms hole in speech
        x = _signal([(1.0, 2.0), (2.04, 3.0), (4.0, 4.04)], 6.0)
        va = eb.voice_activity(x, SR)
        f = eb.DEFAULT_VAD.frame
        self.assertTrue(va[int(2.01 / f)])           # hole filled
        self.assertFalse(va[int(4.01 / f)])          # blip dropped

    def test_digital_zero_channel_is_silent(self):
        va = eb.voice_activity(np.zeros(SR * 3, np.float32), SR)
        self.assertFalse(va.any())


class RefineTest(unittest.TestCase):
    def test_pause_moves_onto_audio_gap(self):
        # A speaks until 5.3, silent 5.3-5.8, speaks again from 5.8;
        # the transcript says the gap is 5.0-5.65 (both sides early)
        va_a = eb.voice_activity(_signal([(0.5, 5.3), (5.8, 9.0)], 10.0), SR)
        ev = {"start": 5.0, "end": 5.65, "duration": 0.65}
        out = eb.refine_events("pause", [ev], va_a)[0]
        self.assertAlmostEqual(out["start"], 5.3, delta=0.021)
        self.assertAlmostEqual(out["end"], 5.8, delta=0.021)
        self.assertAlmostEqual(out["duration"], 0.5, delta=0.041)
        self.assertEqual(out["boundary"]["asr"], {"start": 5.0, "end": 5.65})
        self.assertEqual(out["boundary"]["flags"], [])
        self.assertEqual(ev["start"], 5.0)           # input not modified

    def test_pause_without_silence_is_flagged_not_dropped(self):
        va_a = eb.voice_activity(_signal([(0.5, 9.0)], 10.0), SR)
        out = eb.refine_events("pause", [{"start": 5.0, "end": 5.6, "duration": 0.6}], va_a)
        self.assertEqual(len(out), 1)
        self.assertIn("no_silence", out[0]["boundary"]["flags"])
        self.assertEqual((out[0]["start"], out[0]["end"]), (5.0, 5.6))

    def test_smooth_tt_offset_and_onset(self):
        va_a = eb.voice_activity(_signal([(1.0, 5.25)], 10.0), SR)
        va_b = eb.voice_activity(_signal([(5.5, 9.5)], 10.0, seed=1), SR)
        ev = {"a_end": 5.0, "b_start": 5.3, "gap": 0.3, "latency": 0.3}
        out = eb.refine_events("smooth_tt", [ev], va_a, va_b)[0]
        self.assertAlmostEqual(out["a_end"], 5.25, delta=0.021)
        self.assertAlmostEqual(out["b_start"], 5.5, delta=0.021)
        self.assertAlmostEqual(out["gap"], 0.25, delta=0.041)

    def test_smooth_tt_offset_ignores_short_internal_pause(self):
        # a 0.2 s dip inside A's turn (< min_silence) is not an offset
        va_a = eb.voice_activity(_signal([(1.0, 4.9), (5.1, 5.3)], 10.0), SR)
        self.assertAlmostEqual(eb.refine_offset(va_a, 5.0), 5.3, delta=0.021)

    def test_offset_not_found_keeps_transcript_value(self):
        va_a = eb.voice_activity(_signal([], 10.0), SR)
        out = eb.refine_events("smooth_tt", [{"a_end": 5.0, "b_start": 5.2, "gap": 0.2,
                                              "latency": 0.2}], va_a)[0]
        self.assertEqual(out["a_end"], 5.0)
        self.assertIn("a_end_not_found", out["boundary"]["flags"])

    def test_bc_utterance_bounds(self):
        va_a = eb.voice_activity(_signal([(2.2, 6.3)], 10.0), SR)
        ev = {"a_utt_start": 2.0, "a_utt_end": 6.0, "a_duration": 4.0}
        out = eb.refine_events("bc", [ev], va_a)[0]
        self.assertAlmostEqual(out["a_utt_start"], 2.2, delta=0.021)
        self.assertAlmostEqual(out["a_utt_end"], 6.3, delta=0.021)


if __name__ == "__main__":
    unittest.main()
