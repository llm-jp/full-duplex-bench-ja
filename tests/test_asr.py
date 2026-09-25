"""ASR layer tests that run without any speech model.

Covers (a) subword -> character time assignment, (b) jchat validation,
(c) the hand-off to ``group_chars_to_utterances`` / TOR, and (d) the
``transcribe`` wiring through a stub backend.
"""

import json
import os
import tempfile
import unittest

import numpy as np
import soundfile as sf

from fdbja import asr as asr_pkg
from fdbja.asr._subword import subwords_to_chars
from fdbja.asr.base import ASRModel, validate_jchat
from fdbja.config import Config
from fdbja.data.event_detection import group_chars_to_utterances, is_backchannel
from fdbja.metrics import tor
from fdbja.runner import run_transcribe, transcribe_file


class SubwordToCharsTest(unittest.TestCase):
    def test_multi_char_subword_is_split_evenly_and_capped_by_next_token(self):
        # "はい" at 1.0, "そう" at 1.1: first token is capped at the next onset
        items = subwords_to_chars([("はい", 1.0), ("そう", 1.1), ("です", 1.3)],
                                  char_seconds=0.1)
        self.assertEqual([c["word"] for c in items], list("はいそうです"))
        self.assertEqual(items[0]["start"], 1.0)
        self.assertEqual(items[0]["end"], 1.05)
        self.assertEqual(items[1]["start"], 1.05)
        self.assertEqual(items[1]["end"], 1.1)      # == next token onset
        # last token keeps the nominal duration (2 chars * 0.1 s)
        self.assertEqual(items[-1]["end"], 1.5)
        for c in items:
            self.assertEqual(c["speaker"], "B")
            self.assertLessEqual(c["start"], c["end"])

    def test_token_before_a_pause_does_not_stretch_across_it(self):
        items = subwords_to_chars([("はい", 1.0), ("ええ", 5.0)], char_seconds=0.1)
        self.assertEqual(items[1]["end"], 1.2)      # nominal, not 5.0
        self.assertEqual(items[2]["start"], 5.0)

    def test_short_gaps_are_bridged_long_ones_are_not(self):
        # nominal end of "あ" is 1.1; next onset 1.5 (gap 0.4 <= 0.5) is bridged
        items = subwords_to_chars([("あ", 1.0), ("い", 1.5), ("う", 2.2)],
                                  char_seconds=0.1, bridge_seconds=0.5)
        self.assertEqual(items[0]["end"], 1.5)
        self.assertEqual(items[1]["end"], 1.6)      # gap 0.6 > 0.5: nominal
        # bridging off -> nominal everywhere
        items = subwords_to_chars([("あ", 1.0), ("い", 1.5)], char_seconds=0.1,
                                  bridge_seconds=0.0)
        self.assertEqual(items[0]["end"], 1.1)

    def test_cleans_sentencepiece_markers_and_clips_to_audio_length(self):
        items = subwords_to_chars([("▁", 0.5), ("▁あ", 0.5), (" ", 0.6),
                                   ("の", 0.7)], audio_duration=0.75)
        self.assertEqual([c["word"] for c in items], ["あ", "の"])
        self.assertEqual(items[-1]["end"], 0.75)

    def test_non_monotonic_timestamps_are_clamped(self):
        items = subwords_to_chars([("あ", 1.0), ("い", 0.8), ("う", 1.2)])
        self.assertEqual([c["start"] for c in items], [1.0, 1.0, 1.2])
        self.assertEqual(validate_jchat(items), items)


class ValidateJchatTest(unittest.TestCase):
    def test_sorts_and_accepts_zero_length_items(self):
        items = [{"speaker": "B", "word": "い", "start": 1.0, "end": 1.0},
                 {"speaker": "B", "word": "あ", "start": 0.5, "end": 0.7}]
        out = validate_jchat(items)
        self.assertEqual([c["word"] for c in out], ["あ", "い"])

    def test_rejects_missing_key_and_negative_span(self):
        with self.assertRaises(ValueError):
            validate_jchat([{"speaker": "B", "word": "あ", "start": 0.0}])
        with self.assertRaises(ValueError):
            validate_jchat([{"speaker": "B", "word": "あ", "start": 1.0, "end": 0.5}])
        with self.assertRaises(ValueError):
            validate_jchat([{"speaker": "B", "word": "", "start": 0.0, "end": 0.1}])


class HandOffToTORTest(unittest.TestCase):
    def test_subword_output_groups_into_utterances_and_scores_a_pause(self):
        # a 2-char backchannel at 1.0 s, then a 7-char turn at 4.0 s
        subwords = [("うん", 1.0), ("それ", 4.0), ("で", 4.2), ("ど", 4.3), ("う", 4.4),
                    ("し", 4.5), ("た", 4.6)]
        jchat = subwords_to_chars(subwords)
        utts = group_chars_to_utterances(jchat, "B")
        self.assertEqual(len(utts), 2)
        self.assertEqual(utts[0]["text"], "うん")
        self.assertTrue(is_backchannel(utts[0]))
        self.assertEqual(utts[1]["num_chars"], 7)
        self.assertFalse(is_backchannel(utts[1]))
        # pause events: one under the backchannel (TO=0), one under the turn (TO=1)
        events = [{"start": 0.9, "end": 1.5}, {"start": 4.0, "end": 4.8}]
        res = tor.eval_pause(events, utts)
        self.assertEqual([r["TO"] for r in res], [0, 1])


class _StubASR(ASRModel):
    """Emits one character per 0.5 s of non-silent audio; records calls."""
    name = "stub"
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def transcribe(self, audio, sample_rate, speaker="B"):
        _StubASR.calls.append((len(audio), sample_rate, speaker))
        n = int(round(len(audio) / sample_rate / 0.5))
        return [{"speaker": speaker, "word": "あ", "start": 0.5 * i, "end": 0.5 * i + 0.4}
                for i in range(n) if np.abs(audio[int(0.5 * i * sample_rate):
                                                  int((0.5 * i + 0.5) * sample_rate)]).max() > 0]


class TranscribeWiringTest(unittest.TestCase):
    def setUp(self):
        asr_pkg.register_backend("stub", _StubASR)
        _StubASR.calls.clear()

    def tearDown(self):
        asr_pkg.BACKENDS.pop("stub", None)

    def test_registry_reports_unknown_backend(self):
        with self.assertRaises(ValueError) as cm:
            asr_pkg.load_asr("no_such_backend")
        self.assertIn("reazonspeech_nemo", str(cm.exception))
        self.assertIn(asr_pkg.DEFAULT_BACKEND, asr_pkg.BACKENDS)

    def test_transcribe_file_uses_response_channel_and_writes_jchat(self):
        with tempfile.TemporaryDirectory() as d:
            sr = 24000
            n = 2 * sr
            a = np.zeros(n, dtype=np.float32)           # ch0: speaker A (ignored)
            a[:] = 0.3
            b = np.zeros(n, dtype=np.float32)           # ch1: response, 1st second only
            b[:sr] = 0.2
            wav = os.path.join(d, "combined.wav")
            sf.write(wav, np.stack([a, b], axis=1), sr)
            out = os.path.join(d, "text", "clip.json")
            items = transcribe_file(asr_pkg.load_asr("stub"), wav, out)
            self.assertEqual(_StubASR.calls, [(n, sr, "B")])
            self.assertEqual(len(items), 2)             # 1 s of speech -> 2 chars
            with open(out, encoding="utf-8") as f:
                self.assertEqual(json.load(f), items)

    def test_run_transcribe_follows_the_layout_and_skips_existing(self):
        with tempfile.TemporaryDirectory() as d:
            eval_dir = os.path.join(d, "eval")
            os.makedirs(os.path.join(eval_dir, "pause"))
            for clip in ("c1", "c2"):
                with open(os.path.join(eval_dir, "pause", clip + ".json"), "w") as f:
                    json.dump([], f)
                odir = os.path.join(eval_dir, "pause_m_output", clip)
                os.makedirs(odir)
                sig = np.full((16000, 2), 0.1, dtype=np.float32)
                sf.write(os.path.join(odir, "combined.wav"), sig, 16000)
            cfg = Config({"corpus": {"eval_dir": eval_dir}, "tasks": ["pause"],
                          "models": [{"name": "m"}, {"name": "absent"}],
                          "asr": {"backend": "stub", "options": {"foo": 1}}})
            n = run_transcribe(cfg, log=lambda *_: None)
            self.assertEqual(n, 2)
            out = os.path.join(eval_dir, "pause_m_asr", "text", "c1.json")
            self.assertTrue(os.path.exists(out))
            # second run keeps the existing files
            self.assertEqual(run_transcribe(cfg, log=lambda *_: None), 0)
            self.assertEqual(run_transcribe(cfg, overwrite=True, models=["m"],
                                            log=lambda *_: None), 2)


if __name__ == "__main__":
    unittest.main()
