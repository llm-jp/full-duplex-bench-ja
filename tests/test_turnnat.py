import unittest

import numpy as np

from fdbja.metrics.turnnat import (
    aggregate,
    boundary_times,
    future_pattern,
    nll_series,
    pattern_to_class,
    tbu_scores,
    voice_activity,
)

FR = 20  # frames per second


def _va(spec):
    """Binary VA array from (start, end) second pairs over a 10 s timeline."""
    va = np.zeros(10 * FR, dtype=bool)
    for start, end in spec:
        va[int(start * FR):int(end * FR)] = True
    return va


class VoiceActivityTest(unittest.TestCase):
    def test_all_zero_channel_is_all_inactive(self):
        va = voice_activity(np.zeros(16000), 16000, FR)
        self.assertEqual(len(va), FR)
        self.assertFalse(va.any())

    def test_detects_a_burst_at_the_right_frames(self):
        audio = np.zeros(16000)
        audio[8000:12000] = 0.5          # 0.5-0.75 s
        va = voice_activity(audio, 16000, FR)
        self.assertTrue(va[10:14].all())
        self.assertFalse(va[:10].any())


class FuturePatternTest(unittest.TestCase):
    def test_bins_follow_the_speaker_and_horizon(self):
        # A speaks only 0.1 s ahead of frame 0; B only 1.5 s ahead.
        va_a = _va([(0.05, 0.15)])
        va_b = _va([(1.4, 1.6)])
        bits = future_pattern(va_a, va_b, FR)
        self.assertEqual(bits.shape[1], 8)
        self.assertTrue(bits[0, 0])          # A, bin [0, 200] ms
        self.assertFalse(bits[0, 1:4].any())  # A, later bins
        self.assertTrue(bits[0, 7])          # B, bin [1200, 2000] ms
        self.assertFalse(bits[0, 4:7].any())

    def test_horizon_frames_are_dropped(self):
        va = np.ones(10 * FR, dtype=bool)
        bits = future_pattern(va, va, FR)
        self.assertEqual(len(bits), 10 * FR - int(2.0 * FR))


class ClassIndexTest(unittest.TestCase):
    def test_bit_order_is_explicit_and_reversible(self):
        bits = np.array([[True] + [False] * 7])
        self.assertEqual(pattern_to_class(bits, msb_first=True)[0], 128)
        self.assertEqual(pattern_to_class(bits)[0], 1)   # default: LSB-first
        self.assertEqual(pattern_to_class(bits, msb_first=False)[0], 1)

    def test_range_is_within_the_codebook(self):
        rng = np.random.default_rng(0)
        bits = rng.random((50, 8)) > 0.5
        cls = pattern_to_class(bits)
        self.assertTrue((cls >= 0).all() and (cls < 256).all())


class NllTest(unittest.TestCase):
    def test_confident_correct_prediction_scores_near_zero(self):
        probs = np.full((3, 256), 1e-6)
        probs[np.arange(3), [7, 7, 7]] = 1.0
        nll = nll_series(probs, np.array([7, 7, 7]))
        self.assertTrue(np.allclose(nll, 0.0, atol=1e-6))

    def test_surprising_observation_scores_high(self):
        probs = np.full((1, 256), 1e-6)
        probs[0, 3] = 1.0
        self.assertGreater(nll_series(probs, np.array([9]))[0], 10.0)


class TbuTest(unittest.TestCase):
    def test_boundaries_are_onsets_and_offsets_of_both_speakers(self):
        va_a = _va([(1.0, 2.0)])
        va_b = _va([(3.0, 4.0)])
        self.assertEqual(boundary_times(va_a, va_b, FR), [1.0, 2.0, 3.0, 4.0])

    def test_short_regions_are_discarded(self):
        va_a = _va([(1.0, 1.1)])         # 100 ms < 200 ms minimum
        self.assertEqual(boundary_times(va_a, np.zeros_like(va_a), FR), [])

    def test_unit_score_averages_the_preceding_context(self):
        nll = np.concatenate([np.zeros(FR), np.ones(FR)])   # 0-1 s: 0, 1-2 s: 1
        units = tbu_scores(nll, [2.0], FR, context=2.0)
        self.assertAlmostEqual(units[0], 0.5)

    def test_boundary_outside_scored_frames_is_dropped(self):
        self.assertEqual(len(tbu_scores(np.zeros(10), [0.0], FR)), 0)


class AggregateTest(unittest.TestCase):
    def test_tail_makes_one_bad_boundary_dominate(self):
        ordinary = np.full(20, 1.0)
        with_outlier = np.concatenate([np.full(19, 1.0), [10.0]])
        self.assertLess(aggregate(with_outlier, top_k=1),
                        aggregate(ordinary, top_k=1))

    def test_higher_score_means_more_natural(self):
        self.assertGreater(aggregate(np.full(5, 0.5)), aggregate(np.full(5, 5.0)))

    def test_empty_input_is_nan_not_a_crash(self):
        self.assertTrue(np.isnan(aggregate(np.asarray([]))))


if __name__ == "__main__":
    unittest.main()
