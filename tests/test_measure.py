"""``fdbja measure`` / ``fdbja make-profile``: synthetic two-channel signals."""

import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf

import fdbja
from fdbja import cli, load_profile, make_profile, measure
from fdbja import measurement as ms

SR = 16000
TOTAL = 40.0
D = 0.021           # one 20 ms frame

# user (channel 0) and system (channel 1) speech, seconds
USER = [(1.0, 4.0),                     # turn 1
        (9.0, 12.0), (12.7, 16.0),      # turn 2 with a 0.7 s pause
        (20.0, 23.0), (23.8, 27.0),     # turn 3 with a 0.8 s pause
        (30.0, 33.0)]                   # turn 4
SYSTEM = [(4.5, 7.0),                   # response to turn 1, +0.5 s
          (12.2, 12.6),                 # short vocalisation in the pause of turn 2
          (15.8, 18.0),                 # response to turn 2, 0.2 s before its end
          (23.3, 26.0),                 # starts in the pause of turn 3 and keeps going
          (31.0, 31.4)]                 # short vocalisation while the user speaks (turn 4)
# a human-human dialogue: A (channel 0) and B (channel 1) alternate
HUMAN_A = [(1.0, 4.0), (7.6, 10.6), (14.2, 17.0)]
HUMAN_B = [(4.3, 7.0), (11.0, 14.0)]    # latencies B: +0.3, +0.4; A: +0.6, +0.2


def _signal(segments, total=TOTAL, sr=SR, seed=0):
    """Tones at -20 dBFS on ``segments``, noise at -80 dBFS elsewhere."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(total * sr)) * 1e-4
    t = np.arange(len(x)) / sr
    for s, e in segments:
        m = (t >= s) & (t < e)
        x[m] += 0.1 * np.sqrt(2) * np.sin(2 * np.pi * 220 * t[m])
    return x.astype(np.float32)


def _stereo(a, b, total=TOTAL, sr=SR):
    return np.stack([_signal(a, total, sr, 0), _signal(b, total, sr, 1)], axis=1)


class _Files(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def wav(self, name, data, sr=SR):
        path = os.path.join(self.dir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sf.write(path, data, sr, subtype="PCM_16")
        return path


class EventsAndStatsTest(_Files):
    def setUp(self):
        super().setUp()
        self.path = self.wav("rec.wav", _stereo(USER, SYSTEM))
        self.m = measure(self.path)

    def test_turn_ends_and_latency(self):
        ends = [e for e in self.m.events if e["type"] == "turn_end"]
        self.assertEqual([e["outcome"] for e in ends], ["after", "early", "no_response"])
        self.assertAlmostEqual(ends[0]["time"], 4.0, delta=D)
        self.assertAlmostEqual(ends[0]["latency"], 0.5, delta=2 * D)
        self.assertAlmostEqual(ends[1]["latency"], -0.2, delta=2 * D)   # the sign is kept
        te = self.m.stats["turn_end"]
        self.assertEqual((te["n"], te["n_response"]), (3, 2))
        self.assertAlmostEqual(te["response_rate"], 2 / 3, places=3)
        self.assertAlmostEqual(te["no_response_rate"], 1 / 3, places=3)
        self.assertAlmostEqual(te["latency_median"], 0.15, delta=2 * D)
        self.assertAlmostEqual(te["latency_q1"], -0.025, delta=2 * D)
        self.assertAlmostEqual(te["latency_q3"], 0.325, delta=2 * D)
        self.assertEqual(te["early_start_rate"], 0.5)
        # turn 3 ends while the system is already speaking: not a turn end
        self.assertEqual(te["skipped"]["system_speaking"], 1)

    def test_pauses_split_by_length(self):
        pauses = [e for e in self.m.events if e["type"] == "pause"]
        self.assertEqual([p["outcome"] for p in pauses], ["short", "long"])
        self.assertAlmostEqual(pauses[0]["start"], 12.0, delta=D)
        self.assertAlmostEqual(pauses[0]["end"], 12.7, delta=D)
        pa = self.m.stats["pause"]
        self.assertEqual((pa["n"], pa["n_short"], pa["n_long"]), (2, 1, 1))
        self.assertEqual((pa["voiced_rate"], pa["short_rate"], pa["long_rate"]), (1.0, 0.5, 0.5))

    def test_starts_during_user_speech_overlap_and_speaking_time(self):
        st = self.m.stats["start_in_user_speech"]
        # the early response to turn 2 is listed under n_at_turn_end
        self.assertEqual((st["n"], st["n_short"], st["n_long"], st["n_at_turn_end"]),
                         (1, 1, 0, 1))
        self.assertAlmostEqual(st["per_minute_of_user_speech"], 60 / 18.5, delta=0.05)
        ov, sp = self.m.stats["overlap"], self.m.stats["speaking"]
        self.assertAlmostEqual(ov["seconds"], 2.8, delta=0.1)
        self.assertAlmostEqual(ov["share_of_recording"], 2.8 / 40, delta=0.003)
        self.assertAlmostEqual(sp["user_seconds"], 18.5, delta=0.15)
        self.assertAlmostEqual(sp["system_seconds"], 8.2, delta=0.15)
        self.assertAlmostEqual(sp["system_share_of_speech"], 8.2 / 26.7, delta=0.01)

    def test_no_score_without_profile(self):
        self.assertIsNone(self.m.score)
        self.assertIsNone(self.m.profile)

    def test_thresholds_can_be_changed(self):
        # turns must last 4 s: turn 1 (3 s) and turn 4 (3 s) are no longer turn ends
        m = measure(self.path, config={"turn_min_seconds": 4.0})
        self.assertEqual(m.stats["turn_end"]["n"], 1)
        # a short vocalisation of at most 0.3 s: the 0.4 s one in the pause is now long
        m = measure(self.path, config={"short_max_seconds": 0.3})
        self.assertEqual(m.stats["pause"]["n_long"], 2)
        # pauses of 0.75 s and more only
        m = measure(self.path, config={"pause_min_gap": 0.75})
        self.assertEqual(m.stats["pause"]["n"], 1)

    def test_swapped_channels(self):
        m = measure(self.path, user_channel=1)
        self.assertAlmostEqual(m.stats["speaking"]["user_seconds"], 8.2, delta=0.15)
        self.assertIn("ch:user=1,sys=0", m.signature)

    def test_report_text(self):
        text = str(self.m)
        self.assertIn("user turn ends (n = 3)", text)
        self.assertIn("by length only", text)
        self.assertIn("signature: fdbja:", text)
        self.assertIn("not comparable with the leaderboard", text)


class InputFormsTest(_Files):
    def test_mono_pair_arrays_and_other_sampling_rate(self):
        x = _stereo(USER, SYSTEM)
        ref = measure(self.wav("rec.wav", x)).stats
        u, s = self.wav("user.wav", x[:, 0]), self.wav("system.wav", x[:, 1])
        m = measure(u, s)
        self.assertEqual(m.stats, ref)
        self.assertIn("ch:user=input1,sys=input2", m.signature)
        # float arrays instead of 16-bit files: the same events
        for got in (measure(x, sample_rate=SR), measure(x[:, 0], x[:, 1], sample_rate=SR)):
            self.assertEqual(got.stats["turn_end"]["n"], ref["turn_end"]["n"])
            self.assertEqual(got.stats["pause"], ref["pause"])
        # 8 kHz input is used as it is (no resampling)
        m8 = measure(_stereo(USER, SYSTEM, sr=8000), sample_rate=8000)
        self.assertEqual(m8.stats["turn_end"]["n"], 3)
        self.assertAlmostEqual(m8.stats["turn_end"]["latency_median"], 0.15, delta=2 * D)

    def test_directory_is_pooled(self):
        self.wav("set/a.wav", _stereo(USER, SYSTEM))
        self.wav("set/b.wav", _stereo(USER, SYSTEM[:1]))
        m = measure(os.path.join(self.dir, "set"))
        self.assertEqual([r["name"] for r in m.recordings], ["a.wav", "b.wav"])
        self.assertEqual(m.stats["n_recordings"], 2)
        self.assertEqual(m.stats["turn_end"]["n"],
                         sum(r["stats"]["turn_end"]["n"] for r in m.recordings))
        self.assertAlmostEqual(m.stats["duration_seconds"], 80.0, delta=0.1)
        self.assertEqual({e["recording"] for e in m.events}, {0, 1})
        self.assertIn("ALL", str(m))

    def test_bad_input(self):
        x = _stereo(USER, SYSTEM)
        with self.assertRaises(ValueError):
            measure(self.wav("mono.wav", x[:, 0]))          # one channel, no system input
        with self.assertRaises(ValueError):
            measure(x.T, sample_rate=SR)                     # (channels, samples)
        with self.assertRaises(ValueError):
            measure(x)                                       # arrays need sample_rate
        with self.assertRaises(ValueError):
            measure(x, sample_rate=SR, config={"no_such_key": 1})
        with self.assertRaises(ValueError):
            measure(x, sample_rate=SR, config={"vad": {"search": 1.0}})

    def test_silent_system_gives_warning_not_error(self):
        m = measure(_stereo(USER, []), sample_rate=SR)
        self.assertEqual(m.stats["turn_end"]["response_rate"], 0.0)
        self.assertIsNone(m.stats["turn_end"]["latency_median"])
        self.assertTrue(any("system channel" in w for w in m.warnings))


class ProfileTest(_Files):
    def test_generic_profile_is_rule_constants_only(self):
        p = load_profile("generic")
        self.assertEqual(p["norms"]["response_latency"]["low"], -0.5)
        self.assertEqual(p["norms"]["response_latency"]["high"], 2.0)
        self.assertEqual(p["meta"]["n_recordings"], 0)
        self.assertIsNone(p["reference"])
        self.assertIn("Not derived from human data", p["description"])
        m = measure(_stereo(USER, SYSTEM), sample_rate=SR, profile="generic")
        sc = m.score
        # the tolerance range is the response window, so this is the response rate
        self.assertEqual(sc["latency_in_tolerance_rate"], m.stats["turn_end"]["response_rate"])
        self.assertEqual(sc["latency_deviation_mean"], 0.0)
        # pause of turn 2: brief reaction; pause of turn 3: speech that keeps going
        self.assertEqual((sc["pause_no_reaction_rate"], sc["pause_barge_in_rate"],
                          sc["pause_reaction_penalty"]), (0.0, 0.5, 0.5))
        self.assertIn("no human reference", sc["note"])
        self.assertIn("|profile:generic|", m.signature)
        # names of the published benchmark columns are not reused
        published = {"TOR", "pause_response", "pause_no_response_rate",
                     "pause_interrupt_rate", "lat_median", "overlap_rate"}
        self.assertFalse(published & set(sc))
        self.assertFalse(published & {k for part in m.stats.values()
                                      if isinstance(part, dict) for k in part})

    def test_make_profile_round_trip(self):
        human = self.wav("human/h1.wav", _stereo(HUMAN_A, HUMAN_B, total=20.0))
        profile = make_profile(os.path.dirname(human), language="ja",
                               relationship="strangers", situation="chat",
                               medium="video call", source="synthetic test signal",
                               tolerance_percentiles=(0, 100))
        meta = profile["meta"]
        self.assertEqual((meta["language"], meta["relationship"], meta["situation"],
                          meta["medium"]), ("ja", "strangers", "chat", "video call"))
        self.assertEqual(meta["n_recordings"], 1)
        self.assertEqual(meta["n_events"]["responses"], 4)       # both speakers as the user
        self.assertEqual(meta["detection"], ms.DEFAULT_MEASURE.to_dict())
        self.assertTrue(meta["warnings"])                          # only 4 responses
        tol = profile["norms"]["response_latency"]
        self.assertAlmostEqual(tol["low"], 0.2, delta=2 * D)
        self.assertAlmostEqual(tol["high"], 0.6, delta=2 * D)
        self.assertEqual(profile["reference"]["turn_end"]["n_response"], 4)

        path = ms.write_profile(profile, os.path.join(self.dir, "out", "profile.json"))
        self.assertEqual(load_profile(path), profile)
        m = measure(_stereo(USER, SYSTEM), sample_rate=SR, profile=path)
        sc = m.score
        # +0.5 s is inside 0.2-0.6 s, -0.2 s is 0.4 s outside, one turn end has no response
        self.assertAlmostEqual(sc["latency_in_tolerance_rate"], 1 / 3, places=3)
        self.assertAlmostEqual(sc["latency_deviation_mean"], 0.2, delta=2 * D)
        self.assertEqual(sc["reference"]["turn_end"]["n_response"], 4)
        self.assertRegex(m.signature, r"\|profile:ja-strangers-chat-video_call@[0-9a-f]{8}\|")
        self.assertEqual(m.profile["meta"]["medium"], "video call")
        self.assertIn("profile reference", str(m))

    def test_profile_carries_its_detection_settings(self):
        human = self.wav("h.wav", _stereo(HUMAN_A, HUMAN_B, total=20.0))
        profile = make_profile(human, language="ja", relationship="strangers",
                               situation="chat", medium="video call",
                               config={"turn_min_seconds": 2.5})
        x = _stereo(USER, SYSTEM)
        m = measure(x, sample_rate=SR, profile=profile)
        self.assertEqual(m.config["turn_min_seconds"], 2.5)         # taken from the profile
        self.assertIn("events:custom(turn_min_seconds=2.5)", m.signature)
        self.assertEqual(m.warnings, [])
        m = measure(x, sample_rate=SR, profile=profile, config={})
        self.assertTrue(any("differ" in w for w in m.warnings))

    def test_malformed_profile(self):
        with self.assertRaises(ValueError):
            load_profile({"format": "something-else"})
        bad = load_profile("generic")
        bad["norms"]["response_latency"] = {"low": 1.0, "high": 0.0}
        with self.assertRaises(ValueError):
            load_profile(bad)
        with self.assertRaises(ValueError):                         # no response at all
            make_profile(self.wav("s.wav", _stereo(USER, [])), language="ja",
                         relationship="x", situation="x", medium="x")


class SignatureTest(unittest.TestCase):
    def test_signature_follows_the_settings(self):
        v = fdbja.__version__
        self.assertEqual(
            ms.signature(),
            f"fdbja:{v}|measure|vad:energy(p56)|events:default|profile:none|ch:user=0,sys=1")
        cfg = ms.MeasureConfig.from_dict({"vad": {"margin_db": 12.0}})
        self.assertIn("|vad:energy(p56;margin_db=12.0)|events:default|", ms.signature(cfg))
        cfg = ms.MeasureConfig.from_dict({"vad": {"mode": "fixed"}, "turn_min_seconds": 4.0,
                                          "response_window": [0.0, 3.0]})
        self.assertIn("|vad:energy(fixed)|events:custom(response_window=0,3;turn_min_seconds=4.0)|",
                      ms.signature(cfg))
        self.assertIn("|profile:generic|", ms.signature(profile=load_profile("generic")))
        edited = load_profile("generic")
        edited["norms"]["response_latency"]["high"] = 1.0
        self.assertRegex(ms.signature(profile=edited), r"\|profile:generic@[0-9a-f]{8}\|")

    def test_settings_from_text(self):
        got = ms.parse_settings(["turn_min_seconds=4", "vad.margin_db=12.5",
                                 "response_window=-0.5,2.5", "vad.mode=floor"])
        self.assertEqual(got, {"turn_min_seconds": 4, "response_window": [-0.5, 2.5],
                               "vad": {"margin_db": 12.5, "mode": "floor"}})
        self.assertEqual(ms.MeasureConfig.from_dict(got).response_window, (-0.5, 2.5))

    def test_text_hash(self):
        self.assertRegex(ms.text_hash("a prompt"), r"^sha256:[0-9a-f]{12}$")
        self.assertNotEqual(ms.text_hash("a prompt"), ms.text_hash("another prompt"))


class CommandLineTest(_Files):
    def test_measure_and_make_profile(self):
        rec = self.wav("rec.wav", _stereo(USER, SYSTEM))
        human = self.wav("human/h1.wav", _stereo(HUMAN_A, HUMAN_B, total=20.0))
        prompt = os.path.join(self.dir, "prompt.txt")
        with open(prompt, "w", encoding="utf-8") as f:
            f.write("you are a helpful voice agent")
        prof = os.path.join(self.dir, "profile.json")
        self.assertEqual(cli.main(["make-profile", os.path.dirname(human), "--language", "ja",
                                   "--relationship", "strangers", "--situation", "chat",
                                   "--medium", "video call", "-o", prof]), 0)
        out = os.path.join(self.dir, "out.json")
        self.assertEqual(cli.main(["measure", rec, "--profile", prof,
                                   "--condition", "model=example-1", "--condition", "voice=a",
                                   "--condition-file", f"prompt={prompt}",
                                   "--set", "turn_min_seconds=2", "--json", out]), 0)
        with open(out, encoding="utf-8") as f:
            res = json.load(f)
        self.assertEqual(res["conditions"]["model"], "example-1")
        self.assertRegex(res["conditions"]["prompt"], r"^sha256:[0-9a-f]{12}$")
        self.assertNotIn("helpful", json.dumps(res))                # the prompt itself is not stored
        self.assertIn("events:custom(turn_min_seconds=2)", res["signature"])
        self.assertIn("leaderboard", res["comparability"])
        self.assertEqual(res["stats"]["turn_end"]["n"], 3)
        self.assertEqual(res["profile"]["meta"]["language"], "ja")
        self.assertEqual(cli.main(["measure", rec, "--condition", "novalue"]), 2)


class BaseDependenciesOnlyTest(unittest.TestCase):
    def test_measure_imports_no_optional_dependency(self):
        heavy = ["torch", "torchaudio", "whisperx", "espnet", "nemo", "transformers",
                 "matplotlib", "scipy", "pandas", "librosa", "sklearn"]
        code = ("import sys; import fdbja.cli; from fdbja import measure, load_profile, "
                "make_profile; import fdbja.measurement; "
                f"print([m for m in {heavy!r} if m in sys.modules])")
        got = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             check=True, env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)})
        self.assertEqual(got.stdout.strip(), "[]")

    def test_import_fdbja_is_lazy(self):
        code = "import sys, fdbja; print('fdbja.measurement' in sys.modules)"
        got = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             check=True, env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)})
        self.assertEqual(got.stdout.strip(), "False")


class FaintNoiseTest(unittest.TestCase):
    """A channel in which nobody speaks must come out silent, whatever its noise."""

    def _faint(self):
        # a system channel that never rises above -80 dBFS: noise at -100 dBFS
        # with louder stretches at -84 dBFS (what a codec leaves in a silent
        # output). A threshold relative to the channel marks the louder
        # stretches as voice.
        rng = np.random.default_rng(3)
        x = rng.standard_normal(int(TOTAL * SR)) * 1e-5
        t = np.arange(len(x)) / SR
        for s, e in [(4.5, 7.0), (12.2, 13.4), (23.3, 26.0)]:
            m = (t >= s) & (t < e)
            x[m] = rng.standard_normal(int(m.sum())) * 6e-5
        return x.astype(np.float32)

    def test_relative_threshold_alone_marks_noise(self):
        cfg = ms.MeasureConfig.from_dict({"min_level_db": -120.0})
        va, _dt = ms._activity(self._faint(), SR, cfg)
        self.assertGreater(va.mean(), 0.1)

    def test_default_keeps_a_faint_channel_silent(self):
        va, _dt = ms._activity(self._faint(), SR, ms.DEFAULT_MEASURE)
        self.assertEqual(int(va.sum()), 0)
        m = measure(_signal(USER), self._faint(), sample_rate=SR)
        self.assertEqual(m.stats["speaking"]["system_seconds"], 0.0)
        self.assertEqual(m.stats["pause"]["voiced_rate"], 0.0)
        self.assertEqual(m.stats["turn_end"]["no_response_rate"], 1.0)
        self.assertEqual(m.stats["turn_end"]["skipped"]["system_speaking"], 0)

    def test_speech_at_normal_level_is_unchanged(self):
        x = _stereo(USER, SYSTEM)
        off = ms.MeasureConfig.from_dict({"min_level_db": -120.0})
        for ch in (0, 1):
            a, _ = ms._activity(x[:, ch], SR, ms.DEFAULT_MEASURE)
            b, _ = ms._activity(x[:, ch], SR, off)
            self.assertTrue(np.array_equal(a, b))

    def test_setting_is_in_the_signature_when_changed(self):
        cfg = ms.MeasureConfig.from_dict({"min_level_db": -70.0})
        self.assertIn("min_level_db=-70.0", ms.signature(cfg))
        self.assertIn("events:default", ms.signature(ms.DEFAULT_MEASURE))


if __name__ == "__main__":
    unittest.main()
