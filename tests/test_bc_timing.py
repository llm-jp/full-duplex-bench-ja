import unittest

from fdbja.data.event_detection import detect_backchannels, group_chars_to_utterances
from fdbja.metrics.tor import eval_bc, timing_jsd


def _chars(speaker, text, start, step=0.2):
    return [
        {"speaker": speaker, "word": ch,
         "start": start + i * step, "end": start + (i + 1) * step}
        for i, ch in enumerate(text)
    ]


class BackchannelTimingTest(unittest.TestCase):
    def test_timing_jsd_bounds(self):
        self.assertAlmostEqual(timing_jsd([1.0, 2.0], [1.0, 2.0], 0.0, 4.0), 0.0)
        self.assertAlmostEqual(timing_jsd([0.1], [3.9], 0.0, 4.0), 1.0)
        self.assertEqual(timing_jsd([1.0], [], 0.0, 4.0), 1.0)
        v = timing_jsd([1.0, 2.0], [1.0, 3.0], 0.0, 4.0)
        self.assertTrue(0.0 < v < 1.0)

    def test_event_carries_human_onsets_and_eval_reports_jsd(self):
        a = _chars("A", "abcdefghijklmnopqrst", 0.0)  # 4 s utterance
        events = detect_backchannels(a + _chars("B", "うん", 1.0))
        self.assertEqual(events[0]["human_bc_times"], [1.0])
        # model backchannels at the same time -> JSD 0, no takeover
        model_b = group_chars_to_utterances(_chars("B", "はい", 1.0), "B")
        res = eval_bc(events, model_b)
        self.assertEqual(res[0]["TO"], 0)
        self.assertEqual(res[0]["bc_count"], 1)
        self.assertAlmostEqual(res[0]["jsd"], 0.0)
        # model silent -> JSD 1
        self.assertEqual(eval_bc(events, [])[0]["jsd"], 1.0)
        # model takes over -> TO 1
        long_b = group_chars_to_utterances(_chars("B", "とてもながいはつわです", 1.0), "B")
        self.assertEqual(eval_bc(events, long_b)[0]["TO"], 1)


if __name__ == "__main__":
    unittest.main()
