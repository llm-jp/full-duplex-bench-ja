"""Recording quality estimates and warnings of ``fdbja measure``: synthetic
two-channel signals, each degraded in one way."""

import os
import tempfile
import unittest

import numpy as np
import soundfile as sf

from fdbja import measure
from fdbja import measurement as ms
from fdbja import recording_quality as rq
from fdbja.data.event_boundaries import DEFAULT_VAD, voice_activity, voice_activity_levels

SR = 16000
TOTAL = 40.0
USER = [(1.0, 4.0), (9.0, 12.0), (12.7, 16.0), (20.0, 23.0), (23.8, 27.0), (30.0, 33.0)]
SYSTEM = [(4.5, 7.0), (16.5, 19.0), (27.5, 29.5), (33.5, 37.0)]


def _voice(segments, seed, total=TOTAL, sr=SR, level=0.1):
    """Speech-like bursts (low-passed noise with a 4 Hz envelope, about
    -23 dBFS) on ``segments``, noise at -80 dBFS elsewhere. Unlike tones, two
    such signals are uncorrelated, as two speakers are."""
    rng = np.random.default_rng(seed)
    n = int(total * sr)
    x = rng.standard_normal(n) * 1e-4
    t = np.arange(n) / sr
    for a, b in segments:
        i, j = int(a * sr), int(b * sr)
        w = np.convolve(rng.standard_normal(j - i), np.ones(8) / 8, mode="same")
        x[i:j] += level * 3 * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t[i:j])) * w
    return x


def _pair(user=USER, system=SYSTEM, **kw):
    return _voice(user, 1, **kw), _voice(system, 2, **kw)


def _delayed(x, seconds, sr=SR):
    k = int(round(seconds * sr))
    y = np.zeros_like(x)
    y[k:] = x[:len(x) - k]
    return y


def _db(g):
    return 10.0 ** (g / 20.0)


def _measure(u, s, **kw):
    return measure(np.stack([u, s], axis=1).astype(np.float32), sample_rate=SR, **kw)


def _flags(m):
    return m.recordings[0]["quality"]["flags"]


class CleanRecordingTest(unittest.TestCase):
    def setUp(self):
        self.m = _measure(*_pair())

    def test_no_warning_and_no_flag(self):
        self.assertEqual(self.m.warnings, [])
        self.assertEqual(self.m.notes, [])
        self.assertEqual(_flags(self.m), [])
        self.assertEqual(self.m.quality["flags"], {})

    def test_estimates_are_reported(self):
        q = self.m.recordings[0]["quality"]
        for ch in ("user", "system"):
            self.assertAlmostEqual(q[ch]["speech_level_db"], -23.0, delta=1.5)
            self.assertGreater(q[ch]["snr_db"], 50.0)
            self.assertLess(q[ch]["clipped_share"], 1e-4)
            self.assertEqual(q[ch]["threshold_set_by"], "speech_level")
            self.assertEqual(q[ch]["digital_silence_share"], 0.0)
        self.assertAlmostEqual(q["user"]["voiced_share"], 18.3 / TOTAL, delta=0.02)
        for d in ("system_to_user", "user_to_system"):
            self.assertLess(q["leak"][d]["level_db"], -45.0)
        d = self.m.to_dict()
        self.assertEqual(d["quality"]["n_recordings"], 1)
        self.assertEqual(d["quality"]["thresholds"], rq.THRESHOLDS)
        self.assertEqual(d["notes"], [])
        self.assertIn("recording quality", str(self.m))

    def test_timing_does_not_depend_on_the_estimates(self):
        u, s = _pair()
        x = np.stack([u, s], axis=1).astype(np.float32)
        cfg = ms.DEFAULT_MEASURE
        va_u = voice_activity(x[:, 0], SR, cfg.vad, cfg.min_level_db)
        va_s = voice_activity(x[:, 1], SR, cfg.vad, cfg.min_level_db)
        np.testing.assert_array_equal(
            va_u, voice_activity_levels(x[:, 0], SR, cfg.vad, cfg.min_level_db)[0])
        a = ms.analyse(va_u, va_s, cfg.vad.frame, cfg)
        self.assertEqual(self.m.stats, ms.summarise([a]))


class NoiseTest(unittest.TestCase):
    def _noisy(self, snr, channel):
        u, s = _pair()
        noise = np.random.default_rng(7).standard_normal(len(u)) * _db(-23.0 - snr)
        return _measure(u + noise, s) if channel == "user" else _measure(u, s + noise)

    def test_estimate_follows_the_noise(self):
        for snr in (30, 20, 10):
            q = self._noisy(snr, "user").recordings[0]["quality"]["user"]
            self.assertAlmostEqual(q["snr_db"], snr, delta=3.0)

    def test_warning_below_the_limit_only(self):
        self.assertEqual(self._noisy(30, "user").warnings, [])
        self.assertEqual(self._noisy(26, "user").warnings, [])     # checked down to 25 dB
        self.assertEqual(_flags(self._noisy(18, "user")), ["noise:user"])
        for ch in ("user", "system"):
            m = self._noisy(12, ch)
            self.assertEqual(_flags(m), [f"noise:{ch}"])
            self.assertEqual(len(m.warnings), 1)
            self.assertIn(f"the speech on the {ch} channel is only", m.warnings[0])
            self.assertIn("no-response rate", m.warnings[0])       # which way it moves
            self.assertIn("warning: array: the speech", str(m))


class LeakTest(unittest.TestCase):
    def test_level_and_lag_are_recovered(self):
        u, s = _pair()
        for gain in (-35, -25, -15):
            q = _measure(u + _db(gain) * _delayed(s, 0.05), s).recordings[0]["quality"]["leak"]
            self.assertAlmostEqual(q["system_to_user"]["level_db"], gain, delta=1.5)
            self.assertAlmostEqual(q["system_to_user"]["lag_seconds"], 0.05, delta=0.002)
            self.assertLess(q["user_to_system"]["level_db"], -40.0)

    def test_warning_names_the_direction(self):
        u, s = _pair()
        m = _measure(u + _db(-20) * _delayed(s, 0.05), s)
        self.assertEqual(_flags(m), ["leak:user"])
        self.assertIn("the system's voice seems to leak into the user channel", m.warnings[0])
        self.assertIn("more overlap", m.warnings[0])
        m = _measure(u, s + _db(-20) * _delayed(u, 0.03))
        self.assertEqual(_flags(m), ["leak:system"])
        self.assertIn("the user's voice seems to leak into the system channel", m.warnings[0])

    def test_weak_leak_gives_no_warning(self):
        u, s = _pair()
        m = _measure(u + _db(-40) * _delayed(s, 0.05), s + _db(-40) * _delayed(u, 0.05))
        self.assertEqual(m.warnings, [])

    def test_weak_leak_into_a_quiet_channel_gives_a_warning(self):
        # the same -40 dB, but the system channel has no voice of its own: its
        # threshold is low and the leaked voice is close to it
        u, _ = _pair()
        s = _voice([], 2)
        m = _measure(u, s + _db(-40) * _delayed(u, 0.05))
        q = m.recordings[0]["quality"]
        self.assertIn("leak:system", _flags(m))
        self.assertNotIn("leak:user", _flags(m))
        self.assertAlmostEqual(q["leak"]["user_to_system"]["level_db"], -40.0, delta=1.5)
        self.assertGreaterEqual(q["leak"]["user_to_system"]["above_threshold_db"],
                                rq.LEAK_MAX_ABOVE_THRESHOLD_DB)
        self.assertTrue(any("The system channel is quiet" in w for w in m.warnings))
        # with a system that speaks, the same leak is far below its threshold
        u, s = _pair()
        q = _measure(u, s + _db(-40) * _delayed(u, 0.05)).recordings[0]["quality"]
        self.assertLess(q["leak"]["user_to_system"]["above_threshold_db"],
                        rq.LEAK_MAX_ABOVE_THRESHOLD_DB)

    def test_a_channel_that_is_only_a_leak_is_not_its_source(self):
        # the system channel holds nothing but the user's voice at -25 dB: the
        # leak is user -> system, and the user's channel is not warned
        u, _ = _pair()
        m = _measure(u, _voice([], 2) + _db(-25) * _delayed(u, 0.05))
        self.assertIn("leak:system", _flags(m))
        self.assertNotIn("leak:user", _flags(m))

    def test_simultaneous_speech_is_not_a_leak(self):
        # the two speakers talk over each other for most of the recording
        u = _voice([(1.0, 9.0), (12.0, 20.0), (24.0, 33.0)], 1)
        s = _voice([(2.0, 10.0), (13.0, 22.0), (25.0, 36.0)], 2)
        self.assertFalse(any("leak" in w for w in _measure(u, s).warnings))

    def test_long_recordings_are_sampled(self):
        u, s = _pair()
        x = u + _db(-20) * _delayed(s, 0.05)
        act_u = voice_activity_levels(x, SR, DEFAULT_VAD, -60.0)
        act_s = voice_activity_levels(s, SR, DEFAULT_VAD, -60.0)
        full = rq.leak_estimates(x, s, SR, DEFAULT_VAD.frame, act_u[1], act_s[1],
                                 act_u[0], act_s[0])
        old = rq.LEAK_FFT, rq.LEAK_MAX_BLOCKS
        rq.LEAK_FFT, rq.LEAK_MAX_BLOCKS = 1 << 13, 8      # 8 blocks of about 0.6 s
        try:
            few = rq.leak_estimates(x, s, SR, DEFAULT_VAD.frame, act_u[1], act_s[1],
                                    act_u[0], act_s[0])
        finally:
            rq.LEAK_FFT, rq.LEAK_MAX_BLOCKS = old
        self.assertLess(few["system_to_user"]["seconds"], 3.0)
        self.assertGreater(full["system_to_user"]["seconds"], 8.0)
        self.assertAlmostEqual(few["system_to_user"]["level_db"], -20.0, delta=2.0)


class ClippingTest(unittest.TestCase):
    def test_warning_when_clipped(self):
        u, s = _pair()
        m = _measure(np.clip(u * 8, -1, 1), s)
        self.assertEqual(_flags(m), ["clipping:user"])
        self.assertIn("samples of the user channel are at the peak value", m.warnings[0])
        self.assertIn("fewer pauses and turn ends", m.warnings[0])
        self.assertGreater(m.recordings[0]["quality"]["user"]["clipped_share"], 0.01)

    def test_a_few_clipped_samples_give_no_warning(self):
        u, s = _pair()
        peak = np.sort(np.abs(u))[-20]          # all but 19 samples below the limit
        m = _measure(np.clip(u / peak, -1, 1), s)
        self.assertEqual(m.warnings, [])


class LowLevelTest(unittest.TestCase):
    def test_warning_when_the_fixed_limit_sets_the_threshold(self):
        u, s = _pair()
        m = _measure(u * _db(-28), s)                    # speech at about -51 dBFS
        self.assertEqual(_flags(m), ["low_level:user"])
        self.assertEqual(m.recordings[0]["quality"]["user"]["threshold_set_by"], "min_level_db")
        self.assertIn("min_level_db = -60 dBFS", m.warnings[0])
        self.assertIn("--set min_level_db=-84", m.warnings[0])
        self.assertIn("more pauses and turn ends", m.warnings[0])

    def test_no_warning_once_the_limit_is_lowered(self):
        u, s = _pair()
        m = _measure(u * _db(-28), s, config={"min_level_db": -84.0})
        self.assertEqual(m.warnings, [])
        self.assertEqual(m.stats, _measure(u, s).stats)   # the level alone changes nothing


class QuietChannelTest(unittest.TestCase):
    def test_note_not_warning(self):
        u, _ = _pair()
        s = _voice([(4.5, 4.8)], 2)                      # one short sound in 40 s
        m = _measure(u, s)
        self.assertEqual(_flags(m), ["quiet_channel:system"])
        self.assertFalse(any("almost no voice" in w for w in m.warnings))
        self.assertEqual(len(m.notes), 1)
        self.assertIn("the system channel has almost no voice", m.notes[0])
        self.assertIn("note: array: the system channel has almost no voice", str(m))
        self.assertEqual(m.to_dict()["notes"], m.notes)

    def test_digital_silence_gives_no_note(self):
        u, s = _pair()
        m = _measure(u, np.zeros_like(s))
        self.assertEqual(m.notes, [])
        self.assertTrue(any("no voice detected on the system channel" in w for w in m.warnings))
        self.assertIsNone(m.recordings[0]["quality"]["system"]["snr_db"])


class SeveralRecordingsTest(unittest.TestCase):
    def test_one_line_per_kind_with_the_count(self):
        u, s = _pair()
        noise = np.random.default_rng(7).standard_normal(len(u)) * _db(-23.0 - 12)
        with tempfile.TemporaryDirectory() as d:
            for name, a in (("a.wav", u), ("b.wav", u + noise), ("c.wav", u + 2 * noise)):
                sf.write(os.path.join(d, name), np.stack([a, s], axis=1), SR, subtype="PCM_16")
            m = measure(d)
        self.assertEqual([r["quality"]["flags"] for r in m.recordings],
                         [[], ["noise:user"], ["noise:user"]])
        self.assertEqual(m.quality["flags"], {"noise:user": 2})
        self.assertEqual(m.quality["n_recordings"], 3)
        self.assertEqual(len(m.warnings), 1)
        self.assertTrue(m.warnings[0].startswith("2 of 3 recordings (b.wav, c.wav): "))
        snr = m.quality["user"]["snr_db"]
        self.assertEqual(snr["n"], 3)
        self.assertLess(snr["min"], 10.0)
        self.assertGreater(snr["max"], 50.0)
        self.assertIn("median [min, max] over the recordings", str(m))

    def test_recordings_with_the_same_file_name_are_given_by_position(self):
        u, s = _pair()
        noise = np.random.default_rng(7).standard_normal(len(u)) * _db(-23.0 - 12)
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for clip, a in (("c1", u), ("c2", u + noise), ("c3", u + 2 * noise)):
                os.makedirs(os.path.join(d, clip))
                paths.append(os.path.join(d, clip, "combined.wav"))
                sf.write(paths[-1], np.stack([a, s], axis=1), SR, subtype="PCM_16")
            m = measure(paths)
        self.assertTrue(m.warnings[0].startswith("2 of 3 recordings (no. 2, 3): "))


class EdgeCasesTest(unittest.TestCase):
    def test_short_silent_and_unequal_inputs(self):
        u, s = _pair()
        for a, b in ((u[:SR // 2], s[:SR // 2]),            # half a second
                     (np.zeros(SR), np.zeros(SR)),          # digital silence
                     (u[:100], s[:100])):                   # shorter than one frame
            m = _measure(a, b)
            self.assertEqual(m.quality["n_recordings"], 1)
        m = measure(u.astype(np.float32), s[:len(s) - 1234].astype(np.float32), sample_rate=SR)
        self.assertEqual(_flags(m), [])

    def test_other_sampling_rates(self):
        for sr in (8000, 48000):
            u, s = _pair(total=20.0, sr=sr, user=USER[:3], system=SYSTEM[:2])
            x = np.stack([u + _db(-20) * _delayed(s, 0.05, sr), s], axis=1).astype(np.float32)
            q = measure(x, sample_rate=sr).recordings[0]["quality"]
            self.assertEqual(q["flags"], ["leak:user"])
            self.assertAlmostEqual(q["leak"]["system_to_user"]["level_db"], -20.0, delta=1.5)


if __name__ == "__main__":
    unittest.main()
