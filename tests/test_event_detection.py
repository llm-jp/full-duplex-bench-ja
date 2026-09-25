import unittest

from fdbja.data.event_detection import detect_backchannels


def _chars(speaker, text, start, step=0.2):
    return [
        {"speaker": speaker, "word": ch,
         "start": start + i * step, "end": start + (i + 1) * step}
        for i, ch in enumerate(text)
    ]


class EventDetectionTest(unittest.TestCase):
    def test_backchannel_requires_observed_human_backchannel(self):
        a = _chars("A", "abcdefghijklmnopqrst", 0.0)
        self.assertEqual(detect_backchannels(a), [])

        dialogue = a + _chars("B", "うん", 1.0)
        events = detect_backchannels(dialogue)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["human_bc_count"], 1)
