"""Reproduce the human-judgement analysis (2026-11) from its intermediate tables.

Skipped unless ``FDBJA_ANALYSIS_DIR`` points at the directory holding the
analysis tables (not distributed: they are derived from the evaluation set):

  * ``frames.csv``   per event x system x frame (20 Hz, the listened stretch):
                     ``t``, ``t_rel`` (t - t0), ``dkl_acad`` (d_t), ``va_A``, ``va_Bm``
                     (voice activity of A and of the system)
  * ``features2.csv`` boundary-weighted D_KL, tau 2.0 (``acad_exp2.0`` / ``acad_exp2.0_gs``)
  * ``features6.csv`` Pause: ``gap_len``, ``resp_02`` (reaction >= 0.2 s), ``take`` (interrupt), ``pre_ov``

The toolkit functions get the same inputs (no audio, no VAP inference) and
must give the same values.
"""

import collections
import csv
import os

import numpy as np
import pytest

from fdbja.metrics import vap_distance as vd
from fdbja.metrics.pause_response import judge_event
from fdbja.summary import pause_response_stats

D = os.environ.get("FDBJA_ANALYSIS_DIR")
pytestmark = pytest.mark.skipif(
    not (D and os.path.exists(os.path.join(D, "frames.csv"))),
    reason="FDBJA_ANALYSIS_DIR with the analysis tables not set")


def _read(name):
    with open(os.path.join(D, name), encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def frames():
    out = collections.defaultdict(list)
    for r in _read("frames.csv"):
        out[(r["task"], r["stim_id"], r["system"])].append(r)
    return out


def test_boundary_weighted_dkl_matches_features2(frames):
    f2 = {(r["task"], r["stim_id"], r["system"]): r for r in _read("features2.csv")}
    n = 0
    for key, rows in frames.items():
        ref = f2.get(key, {}).get("acad_exp2.0_gs" if key[0] == "pause" else "acad_exp2.0", "")
        if ref == "":
            continue
        t_rel = np.array([float(r["t_rel"]) for r in rows])
        dkl = np.array([float(r["dkl_acad"]) for r in rows])
        # frames.csv covers the listened stretch: the window of boundary_window
        lo, hi, t0 = vd.boundary_window(key[0], {"start": 0.0, "end": 0.0, "a_end": 0.0})
        assert t_rel.min() >= lo - 0.06
        if key[0] == "smooth_tt":
            assert t_rel.max() <= hi + 1e-6
        got = vd.weighted_window_stats(t_rel, dkl, -np.inf, np.inf, t0, 2.0)["mean_dkl"]
        # d_t in frames.csv is rounded to 4 decimals
        assert got == pytest.approx(float(ref), rel=1e-3, abs=1e-3), key
        n += 1
    assert n >= 500


def test_pause_response_matches_features6(frames):
    f6 = {(r["stim_id"], r["system"]): r for r in _read("features6.csv")}
    n = 0
    differ = []
    for (task, stim, system), rows in frames.items():
        if task != "pause" or (stim, system) not in f6:
            continue
        r6 = f6[(stim, system)]
        t = np.array([float(r["t"]) for r in rows])
        t_rel = np.array([float(r["t_rel"]) for r in rows])
        gs = round(float(t[0] - t_rel[0]), 3)
        ge = gs + float(r6["gap_len"])
        va_a = np.array([int(r["va_A"]) for r in rows]) > 0
        va_s = np.array([int(r["va_Bm"]) for r in rows]) > 0
        j = judge_event(va_s, va_a, t, gs, ge)
        n += 1
        assert j["interrupt"] == int(r6["take"]), (stim, system)
        assert j["pre_overlap"] == pytest.approx(float(r6["pre_ov"]), abs=0.051)
        if 1 - j["no_response"] != int(r6["resp_02"]):
            differ.append((stim, system, gs, ge, t, va_s))
    assert n >= 270
    # The analysis kept the VAP time grid in float32, so a voiced part of
    # exactly 0.2 s inside the gap came out as 0.1999996 s and was not counted.
    # The toolkit counts it (">= 0.2 s"). Those are the only differences.
    assert len(differ) <= 3
    for stim, system, gs, ge, t, va_s in differ:
        dt = 0.05
        in_gap = []
        i = 0
        while i < len(va_s):
            if va_s[i]:
                k = i
                while k < len(va_s) and va_s[k]:
                    k += 1
                in_gap.append(max(0.0, min(t[k - 1] + dt, ge) - max(t[i], gs)))
                i = k
            else:
                i += 1
        assert max(in_gap) == pytest.approx(0.2, abs=1e-6), (stim, system)


def test_system_values_match_the_report():
    """System value = event mean of no response + interrupt over the 100 Pause
    events (the values of the analysis report, 9 systems)."""
    by = collections.defaultdict(list)
    for r in _read("features6.csv"):
        by[r["system"]].append({"no_response": 1 - int(r["resp_02"]), "interrupt": int(r["take"]),
                                "pause_response": 1 - int(r["resp_02"]) + int(r["take"])})
    got = sorted(round(pause_response_stats(v, n_boot=0)["pause_response"], 2) for v in by.values())
    assert all(len(v) == 100 for v in by.values())
    assert got == sorted([0.48, 1.07, 0.95, 1.06, 1.00, 1.04, 1.08, 1.09, 1.15])
