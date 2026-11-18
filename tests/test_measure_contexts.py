"""``fdbja measure-contexts``: one system in several dialogue contexts,
on synthetic two-channel signals."""

import contextlib
import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf

import fdbja
from fdbja import cli, make_profile
from fdbja import contexts as cx
from fdbja import measurement as ms

SR = 16000
D = 0.05                    # tolerance on a time measured on 20 ms frames
N = 6                       # recordings per set
OFFSETS = [(i - 2.5) * 0.04 for i in range(N)]      # -0.10 ... +0.10 s, per recording


def _signal(segments, total, seed):
    """Tones at -20 dBFS on ``segments``, noise at -80 dBFS elsewhere."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(total * SR)) * 1e-4
    t = np.arange(len(x)) / SR
    for s, e in segments:
        m = (t >= s) & (t < e)
        x[m] += 0.1 * np.sqrt(2) * np.sin(2 * np.pi * 220 * t[m])
    return x.astype(np.float32)


def _dialogue(latency_b, latency_a, turns=3, turn=3.0):
    """A (channel 0) and B (channel 1) alternate; B starts ``latency_b``
    after A's turns, A ``latency_a`` after B's."""
    a, b, t = [], [], 1.0
    for _ in range(turns):
        a.append((t, t + turn))
        t += turn + latency_b
        b.append((t, t + turn))
        t += turn + latency_a
    total = b[-1][1] + 1.0
    return np.stack([_signal(a, total, 0), _signal(b, total, 1)], axis=1)


class _Sets(unittest.TestCase):
    """Human recordings of two contexts (``chat``: latency 0.3 s, ``task``:
    0.9 s) and systems that shift with them, not at all, or the other way."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.dir = cls._tmp.name
        cls.addClassCleanup(cls._tmp.cleanup)
        for name, latency in (("human_chat", 0.3), ("human_task", 0.9)):
            cls._write(name, latency, latency)
        # the user (A) answers the system after 0.5 s in every set
        for name, latency in (("sys_04", 0.4), ("sys_05", 0.5), ("sys_08", 0.8)):
            cls._write(name, latency, 0.5)
        cls.chat = make_profile(cls.path("human_chat"), language="ja",
                                relationship="strangers", situation="chat",
                                medium="video call")
        cls.task = make_profile(cls.path("human_task"), language="ja",
                                relationship="strangers", situation="task",
                                medium="video call")
        cls.profiles = {"chat": cls.chat, "task": cls.task}

    @classmethod
    def path(cls, name):
        return os.path.join(cls.dir, name)

    @classmethod
    def _write(cls, name, latency_b, latency_a):
        os.makedirs(cls.path(name))
        for i, off in enumerate(OFFSETS):
            sf.write(os.path.join(cls.path(name), f"r{i}.wav"),
                     _dialogue(latency_b + off, latency_a + off), SR, subtype="PCM_16")

    def run_contexts(self, chat, task, **kw):
        kw.setdefault("n_boot", 400)
        return cx.measure_contexts({"chat": self.path(chat), "task": self.path(task)},
                                   self.profiles, **kw)

    @staticmethod
    def latency(result):
        return result.shifts[0]["features"]["latency_median"]


class ProfileRowsTest(_Sets):
    def test_one_row_per_recording(self):
        rows = self.chat["reference_recordings"]
        self.assertEqual(len(rows), N)
        self.assertEqual(set(rows[0]), set(cx.FEATURES))
        for row, off in zip(rows, OFFSETS):
            self.assertAlmostEqual(row["latency_median"], 0.3 + off, delta=D)
        self.assertNotIn("r0", json.dumps(rows))          # no file names

    def test_profile_round_trip_and_older_profiles(self):
        path = ms.write_profile(self.chat, os.path.join(self.dir, "chat.json"))
        self.assertEqual(ms.load_profile(path), self.chat)
        old = {k: v for k, v in self.chat.items() if k != "reference_recordings"}
        self.assertEqual(ms.load_profile(old), old)       # still valid
        bad = {**self.chat, "reference_recordings": "rows"}
        with self.assertRaises(ValueError):
            ms.load_profile(bad)

    def test_generic_profile_has_no_rows(self):
        self.assertIsNone(ms.generic_profile()["reference_recordings"])


class ShiftTest(_Sets):
    def test_system_that_follows(self):
        res = self.run_contexts("sys_04", "sys_08")
        d = self.latency(res)
        self.assertEqual(d["verdict"], "followed")
        self.assertAlmostEqual(d["human"]["value"], 0.6, delta=D)
        self.assertAlmostEqual(d["system"]["value"], 0.4, delta=D)
        self.assertAlmostEqual(d["ratio"], 0.4 / 0.6, delta=0.1)
        for side in ("human", "system"):
            lo, hi = d[side]["interval"]
            self.assertTrue(0 < lo < d[side]["value"] < hi)
            self.assertTrue(d[side]["clear"])
            self.assertEqual(d[side]["n"], [N, N])
        self.assertEqual((res.shifts[0]["from"], res.shifts[0]["to"]), ("chat", "task"))
        s = res.summary
        self.assertEqual((s["n_pairs"], s["n_clear_human_shifts"], s["n_followed"],
                          s["n_opposite"], s["n_no_clear_system_shift"]), (1, 1, 1, 0, 0))
        self.assertEqual(s["followed_rate"], 1.0)
        self.assertEqual(s["n_features"], len(cx.SHIFT_FEATURES))
        counted = sum(s[k] for k in ("n_followed", "n_opposite", "n_no_clear_system_shift",
                                     "n_system_only", "n_no_clear_human_shift",
                                     "n_not_assessed"))
        self.assertEqual(counted, s["n_features"])

    def test_system_that_does_not_shift(self):
        res = self.run_contexts("sys_05", "sys_05")
        d = self.latency(res)
        self.assertEqual(d["verdict"], "no_clear_system_shift")
        self.assertAlmostEqual(d["system"]["value"], 0.0, delta=1e-9)
        lo, hi = d["system"]["interval"]
        self.assertTrue(lo < 0 < hi)
        self.assertAlmostEqual(d["ratio"], 0.0, delta=1e-9)
        self.assertEqual(res.summary["followed_rate"], 0.0)
        # close to neither context's human recordings, and both gaps are clear
        gaps = {n: res.contexts[n]["features"]["latency_median"]["gap"]
                for n in ("chat", "task")}
        self.assertAlmostEqual(gaps["chat"]["value"], 0.2, delta=D)
        self.assertAlmostEqual(gaps["task"]["value"], -0.4, delta=D)
        self.assertTrue(gaps["chat"]["clear"] and gaps["task"]["clear"])

    def test_system_that_shifts_the_other_way(self):
        res = self.run_contexts("sys_08", "sys_04")
        d = self.latency(res)
        self.assertEqual(d["verdict"], "opposite")
        self.assertLess(d["ratio"], 0)
        self.assertEqual((res.summary["n_followed"], res.summary["n_opposite"]), (0, 1))

    def test_features_without_a_human_shift(self):
        feats = self.run_contexts("sys_04", "sys_08").shifts[0]["features"]
        self.assertEqual(set(feats), set(cx.SHIFT_FEATURES))
        self.assertNotIn("pause_reaction_penalty", feats)
        # the same value in every recording: an interval, and it contains 0
        d = feats["overlap_share"]
        self.assertEqual(d["verdict"], "no_clear_human_shift")
        self.assertIsNone(d["ratio"])
        self.assertFalse(d["human"]["clear"])
        # no pause in these recordings: nothing to compare
        d = feats["pause_voiced_rate"]
        self.assertEqual(d["verdict"], "not_assessed")
        self.assertIsNone(d["human"]["value"])
        self.assertEqual(d["human"]["n"], [0, 0])

    def test_system_only(self):
        res = cx.measure_contexts(
            {"a": self.path("sys_04"), "b": self.path("sys_08")},
            {"a": self.chat, "b": self.chat}, n_boot=400)
        self.assertEqual(self.latency(res)["verdict"], "system_only")
        self.assertEqual(res.summary["n_system_only"], 1)
        self.assertIsNone(res.summary["followed_rate"])   # no clear human shift


class ContextTableTest(_Sets):
    def test_values_per_context(self):
        res = self.run_contexts("sys_04", "sys_08")
        chat = res.contexts["chat"]
        self.assertEqual(chat["n_recordings"], N)
        self.assertEqual(chat["profile"]["id"], ms.profile_id(self.chat))
        self.assertEqual(chat["profile"]["n_recordings"], N)
        f = chat["features"]["latency_median"]
        self.assertAlmostEqual(f["system"]["value"], 0.4, delta=D)
        self.assertAlmostEqual(f["human"]["value"], 0.3, delta=D)
        self.assertAlmostEqual(f["gap"]["value"], 0.1, delta=D)
        self.assertEqual((f["system"]["n"], f["human"]["n"]), (N, N))
        lo, hi = f["system"]["interval"]
        self.assertTrue(lo < f["system"]["value"] < hi)
        self.assertEqual(set(chat["features"]), set(cx.FEATURES))
        # the pooled measurement of `fdbja measure` for the context
        self.assertEqual(chat["stats"]["n_recordings"], N)
        self.assertIn("latency_in_tolerance_rate", chat["score"])
        self.assertNotIn("reference", chat["score"])
        self.assertIn("|measure|", chat["signature"])
        self.assertIn(f"profile:{ms.profile_id(self.chat)}", chat["signature"])

    def test_same_settings_and_signature_basis_in_every_context(self):
        res = self.run_contexts("sys_04", "sys_08")
        self.assertEqual(res.config, ms.DEFAULT_MEASURE.to_dict())
        self.assertEqual(res.warnings, [])


class FewRecordingsTest(_Sets):
    def test_no_interval_below_the_minimum(self):
        few = {name: [os.path.join(self.path(d), f"r{i}.wav") for i in range(3)]
               for name, d in (("chat", "sys_04"), ("task", "sys_08"))}
        self.assertLess(3, cx.FEW_RECORDINGS)
        res = cx.measure_contexts(few, self.profiles, n_boot=400)
        d = self.latency(res)
        self.assertEqual(d["verdict"], "not_assessed")
        self.assertIsNotNone(d["system"]["value"])        # the value is still given
        self.assertIsNone(d["system"]["interval"])
        self.assertIsNone(d["system"]["clear"])
        self.assertTrue(d["human"]["clear"])
        est = res.contexts["chat"]["features"]["latency_median"]
        self.assertEqual(est["system"]["n"], 3)
        self.assertIsNone(est["system"]["interval"])
        self.assertIsNone(est["gap"]["interval"])
        self.assertEqual(res.summary["n_not_assessed"], len(cx.SHIFT_FEATURES))
        self.assertIn("system: 3 recording(s) (fewer than 5: no interval)", str(res))

    def test_no_bootstrap(self):
        res = self.run_contexts("sys_04", "sys_08", n_boot=0)
        d = self.latency(res)
        self.assertEqual(d["verdict"], "not_assessed")
        self.assertAlmostEqual(d["system"]["value"], 0.4, delta=D)
        self.assertEqual(res.bootstrap["n"], 0)


class InputsTest(_Sets):
    def test_profile_without_rows(self):
        old = {k: v for k, v in self.chat.items() if k != "reference_recordings"}
        res = cx.measure_contexts(
            {"chat": self.path("sys_04"), "task": self.path("sys_08")},
            {"chat": old, "task": self.task}, n_boot=400)
        f = res.contexts["chat"]["features"]["latency_median"]
        self.assertIsNone(f["human"])
        self.assertIsNone(f["gap"])
        self.assertIsNotNone(res.contexts["task"]["features"]["latency_median"]["human"])
        d = self.latency(res)
        self.assertIsNone(d["human"])
        self.assertEqual(d["verdict"], "not_assessed")
        self.assertTrue(any("chat" in w and "make-profile" in w for w in res.warnings))

    def test_human_recordings_only(self):
        res = cx.measure_contexts(profiles=self.profiles, n_boot=400)
        self.assertIsNone(res.contexts["chat"]["n_recordings"])
        self.assertIsNone(res.contexts["chat"]["features"]["latency_median"]["system"])
        d = self.latency(res)
        self.assertTrue(d["human"]["clear"])
        self.assertIsNone(d["system"])
        self.assertEqual(d["verdict"], "not_assessed")
        self.assertIn("human", str(res))

    def test_system_recordings_only(self):
        res = cx.measure_contexts({"chat": self.path("sys_04"), "task": self.path("sys_08")},
                                  n_boot=400)
        d = self.latency(res)
        self.assertIsNone(d["human"])
        self.assertTrue(d["system"]["clear"])
        self.assertEqual(d["verdict"], "not_assessed")
        self.assertIsNone(res.contexts["chat"]["profile"])
        # without a profile only the layer compared with listeners is scored
        self.assertIn("pause_reaction_penalty", res.contexts["chat"]["score"])
        self.assertNotIn("latency_in_tolerance_rate", res.contexts["chat"]["score"])

    def test_profiles_with_different_settings(self):
        other = make_profile(self.path("human_task"), language="ja",
                             relationship="strangers", situation="task",
                             medium="video call", config={"turn_min_seconds": 2.0})
        with self.assertRaisesRegex(ValueError, "different detection settings"):
            cx.measure_contexts(profiles={"chat": self.chat, "task": other})
        # with explicit settings the run goes through, with a warning
        res = cx.measure_contexts(profiles={"chat": self.chat, "task": other},
                                  config={"turn_min_seconds": 2.0}, n_boot=0)
        self.assertEqual(res.config["turn_min_seconds"], 2.0)
        self.assertTrue(any("detection settings differ" in w for w in res.warnings))

    def test_errors(self):
        with self.assertRaisesRegex(ValueError, "at least two contexts"):
            cx.measure_contexts({"chat": self.path("sys_04")}, {"chat": self.chat})
        with self.assertRaisesRegex(ValueError, "baseline"):
            cx.measure_contexts(profiles=self.profiles, baseline="interview")
        with self.assertRaises(FileNotFoundError):
            cx.measure_contexts({"chat": self.path("missing"), "task": self.path("sys_08")})


class PairsAndDrawsTest(_Sets):
    def setUp(self):
        self.three = {"chat": self.chat, "task": self.task,
                      "interview": copy.deepcopy(self.task)}

    def test_every_pair_or_against_a_baseline(self):
        res = cx.measure_contexts(profiles=self.three, n_boot=200)
        self.assertEqual([(s["from"], s["to"]) for s in res.shifts],
                         [("chat", "task"), ("chat", "interview"), ("task", "interview")])
        self.assertEqual(res.summary["n_pairs"], 3)
        res = cx.measure_contexts(profiles=self.three, baseline="task", n_boot=200)
        self.assertEqual([(s["from"], s["to"]) for s in res.shifts],
                         [("task", "chat"), ("task", "interview")])
        self.assertEqual(res.baseline, "task")

    def test_same_seed_same_result(self):
        a = self.run_contexts("sys_04", "sys_08", seed=3).to_dict()
        b = self.run_contexts("sys_04", "sys_08", seed=3).to_dict()
        self.assertEqual(a, b)
        c = self.run_contexts("sys_04", "sys_08", seed=4).to_dict()
        value = lambda r: r["shifts"][0]["features"]["latency_median"]["system"]["value"]  # noqa: E731
        self.assertEqual(value(a), value(c))              # the values do not depend on it

    def test_draws_do_not_depend_on_the_other_contexts(self):
        two = cx.measure_contexts(profiles=self.profiles, n_boot=200)
        three = cx.measure_contexts(profiles=self.three, n_boot=200)
        self.assertEqual(two.shifts[0]["features"], three.shifts[0]["features"])
        back = cx.measure_contexts(profiles={"task": self.task, "chat": self.chat},
                                   n_boot=200)
        lo, hi = self.latency(two)["human"]["interval"]
        lo2, hi2 = self.latency(back)["human"]["interval"]
        self.assertAlmostEqual(lo2, -hi, places=3)
        self.assertAlmostEqual(hi2, -lo, places=3)


class OutputTest(_Sets):
    def test_dict(self):
        res = self.run_contexts("sys_04", "sys_08", factor="situation")
        d = json.loads(json.dumps(res.to_dict()))
        self.assertEqual(d["format"], cx.CONTEXTS_FORMAT)
        self.assertEqual(d["factor"], "situation")
        self.assertEqual(d["comparability"], ms.NOT_COMPARABLE)
        self.assertEqual(set(d["layers"]), {"context_gap", "context_shift",
                                            "listener_judgement"})
        self.assertEqual(set(d["verdicts"]), set(cx.VERDICTS))
        self.assertTrue(d["features"]["latency_median"]["shift"])
        self.assertFalse(d["features"]["pause_reaction_penalty"]["shift"])
        self.assertEqual(d["bootstrap"]["min_recordings"], cx.FEW_RECORDINGS)
        # names of the published benchmark columns are not reused
        published = {"TOR", "pause_response", "pause_no_response_rate",
                     "pause_interrupt_rate", "lat_median", "overlap_rate"}
        self.assertFalse(published & set(cx.FEATURES))
        self.assertFalse(published & set(d["summary"]))

    def test_report(self):
        text = str(self.run_contexts("sys_04", "sys_08", factor="situation"))
        text = " ".join(text.split())                     # notes are wrapped
        self.assertIn("2 contexts by situation: chat, task", text)
        self.assertIn("shift: chat -> task", text)
        self.assertIn("followed (ratio +0.6", text)
        self.assertIn("followed 1 of 1 clear human shift(s)", text)
        self.assertIn("Without a clear human shift: 3 (the system shifts in 0); "
                      "not assessed 1", text)
        self.assertIn("not compared with listeners' judgement", text)
        self.assertIn("not comparable with the leaderboard", text)
        self.assertNotIn("warning:", text)

    def test_cli(self):
        prof = {}
        for name, profile in self.profiles.items():
            prof[name] = ms.write_profile(profile, os.path.join(self.dir, f"cli_{name}.json"))
        out = os.path.join(self.dir, "contexts.json")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = cli.main(["measure-contexts",
                             f"chat={self.path('sys_04')}", f"task={self.path('sys_08')}",
                             "--profile", f"chat={prof['chat']}",
                             "--profile", f"task={prof['task']}",
                             "--factor", "situation", "--baseline", "chat",
                             "--bootstrap", "300", "--seed", "1", "--json", out])
        self.assertEqual(code, 0)
        self.assertIn("shift: chat -> task", buf.getvalue())
        with open(out, encoding="utf-8") as f:
            res = json.load(f)
        self.assertEqual(res["bootstrap"]["n"], 300)
        self.assertEqual(res["bootstrap"]["seed"], 1)
        self.assertEqual(res["baseline"], "chat")
        self.assertEqual(res["shifts"][0]["features"]["latency_median"]["verdict"],
                         "followed")

    def test_cli_errors(self):
        with contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(cli.main(["measure-contexts", self.path("sys_04")]), 2)
            self.assertEqual(cli.main(["measure-contexts",
                                       f"chat={self.path('sys_04')}"]), 2)
            self.assertEqual(cli.main(["measure-contexts", "--profile", "chat=a.json",
                                       "--profile", "chat=b.json"]), 2)
            self.assertEqual(cli.main(["measure-contexts", f"chat={self.path('missing')}",
                                       f"task={self.path('sys_08')}"]), 2)
        self.assertIn("NAME=RECORDINGS", err.getvalue())
        self.assertIn("at least two contexts", err.getvalue())

    def test_cli_repeated_name_adds_recordings(self):
        files = [os.path.join(self.path("sys_04"), f"r{i}.wav") for i in range(N)]
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            code = cli.main(["measure-contexts", *[f"chat={p}" for p in files],
                             f"task={self.path('sys_08')}", "--bootstrap", "100"])
        self.assertEqual(code, 0)
        self.assertIn(f"context: chat   system: {N} recording(s)   human: -", buf.getvalue())


class ExportTest(unittest.TestCase):
    def test_exported(self):
        self.assertIs(fdbja.measure_contexts, cx.measure_contexts)
        self.assertIs(fdbja.ContextMeasurement, cx.ContextMeasurement)

    def test_import_fdbja_is_lazy(self):
        code = "import sys, fdbja; print('fdbja.contexts' in sys.modules)"
        got = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             check=True,
                             env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)})
        self.assertEqual(got.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()
