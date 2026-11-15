"""Check an installed wheel away from the repository (run by publish.yml).

Writes a synthetic two-channel recording into the current directory, runs the
installed ``fdbja measure`` on it and compares the result with the values the
signal was built to give. Needs the base dependencies only.
"""

import json
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

import fdbja

SR, TOTAL = 16000, 40.0
# user (channel 0) and system (channel 1) speech, seconds: four turn ends answered
# after 0.5 / 0.4 / 0.5 / 0.3 s, and one 0.7 s pause in which the system stays silent
USER = [(1.0, 4.0), (9.0, 12.0), (12.7, 16.0), (20.0, 23.0), (30.0, 33.0)]
SYSTEM = [(4.5, 7.0), (16.4, 18.0), (23.5, 26.0), (33.3, 36.0)]


def channel(segments, seed):
    """Tones at -20 dBFS on ``segments``, noise at -80 dBFS elsewhere."""
    t = np.arange(int(SR * TOTAL)) / SR
    x = np.random.default_rng(seed).standard_normal(len(t)) * 1e-4
    for s, e in segments:
        m = (t >= s) & (t < e)
        x[m] += 0.1 * np.sqrt(2) * np.sin(2 * np.pi * 220 * t[m])
    return x.astype(np.float32)


def main() -> int:
    here = os.path.dirname(os.path.abspath(fdbja.__file__))
    assert "site-packages" in here, f"fdbja was imported from the source tree: {here}"
    sf.write("rec.wav", np.stack([channel(USER, 0), channel(SYSTEM, 1)], axis=1), SR)
    exe = os.path.join(os.path.dirname(sys.executable), "fdbja")
    subprocess.run([exe, "measure", "rec.wav", "--json", "measure.json"], check=True)
    with open("measure.json", encoding="utf-8") as f:
        res = json.load(f)
    stats, score = res["stats"], res["score"]
    assert res["signature"].startswith(f"fdbja:{fdbja.__version__}|measure|"), res["signature"]
    assert res["target"] == "listener-judgement", res["target"]
    assert stats["turn_end"]["n"] == 4, stats["turn_end"]
    assert abs(stats["turn_end"]["latency_median"] - 0.45) <= 0.05, stats["turn_end"]
    assert stats["pause"]["n"] == 1, stats["pause"]
    assert score["pause_reaction_penalty"] == 1.0, score
    print(f"fdbja {fdbja.__version__}: the installed wheel measures the recording as expected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
