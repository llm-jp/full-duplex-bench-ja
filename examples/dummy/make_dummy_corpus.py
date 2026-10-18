"""Write a tiny synthetic corpus in the LLM-jp-Zoom1 layout.

The real evaluation set is rebuilt from LLM-jp-Zoom1, which is not
redistributed (see DATASET.md). This script lets you run the whole pipeline
(``reconstruct`` -> ``infer`` -> ``transcribe`` -> ``score-offline``) without
it. The "speech" is amplitude-modulated noise and every character is a
placeholder, so **the resulting numbers mean nothing**; the point is only to
check that the toolkit is installed and wired correctly.

Layout written under ``--out`` (the same as ``corpus.zoom1_root``)::

    text/<session>/<session>.json          # jchat: [{"speaker", "word", "start", "end"}, ...]
    audio_merged/<session>/<session>.wav   # stereo 16 kHz, ch0 = speaker A, ch1 = speaker B

Each session repeats one pattern: A speaks, pauses briefly (a Pause Handling
event: B stays silent), speaks again, and B takes the turn after a short gap
(a Smooth Turn-Taking event).

Usage::

    uv run python examples/dummy/make_dummy_corpus.py --out data/dummy_zoom1
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import soundfile as sf

SR = 16000
CHAR_STEP = 0.1     # one placeholder character every 0.1 s of speech
CHAR_LEN = 0.08


def _chars(speaker: str, start: float, end: float) -> list:
    out = []
    t = start
    while t + CHAR_LEN <= end + 1e-9:
        out.append({"speaker": speaker, "word": "あ" if speaker == "A" else "い",
                    "start": round(t, 3), "end": round(t + CHAR_LEN, 3)})
        t += CHAR_STEP
    return out


def _speechlike(rng: np.random.Generator, n: int, level: float = 0.1) -> np.ndarray:
    """Low-passed noise with a ~5 Hz syllable-like envelope."""
    x = rng.normal(0.0, 1.0, n).astype(np.float32)
    k = 8
    x = np.convolve(x, np.ones(k, dtype=np.float32) / k, mode="same")
    t = np.arange(n) / SR
    env = 0.6 + 0.4 * np.sin(2 * np.pi * 5.0 * t + rng.uniform(0, 2 * np.pi))
    x = x * env
    return (level * x / (np.abs(x).max() + 1e-9)).astype(np.float32)


def make_session(rng: np.random.Generator, duration: float = 60.0):
    """Return (jchat, stereo audio) for one synthetic session."""
    segs = []   # (speaker, start, end)
    t = 0.5
    while True:
        a1 = rng.uniform(2.8, 3.4)
        pause = rng.uniform(0.45, 0.9)      # Pause Handling: 0.4-1.0 s, B silent
        a2 = rng.uniform(4.2, 5.0)          # >= 4 s so the A->B switch qualifies
        gap = rng.uniform(0.1, 0.3)         # Smooth Turn-Taking: < 0.4 s
        b = rng.uniform(4.3, 5.0)           # >= 4 s
        rest = rng.uniform(0.6, 1.2)
        end = t + a1 + pause + a2 + gap + b
        if end + rest > duration:
            break
        segs.append(("A", t, t + a1))
        t += a1 + pause
        segs.append(("A", t, t + a2))
        t += a2 + gap
        segs.append(("B", t, t + b))
        t += b + rest

    jchat = []
    audio = np.zeros((int(duration * SR), 2), dtype=np.float32)
    for spk, s, e in segs:
        jchat.extend(_chars(spk, s, e))
        ch = 0 if spk == "A" else 1
        i0, i1 = int(s * SR), int(e * SR)
        audio[i0:i1, ch] = _speechlike(rng, i1 - i0)
    jchat.sort(key=lambda c: (c["start"], c["speaker"]))
    return jchat, audio


def make_dummy_corpus(out: str, n_sessions: int = 10, seed: int = 0,
                      duration: float = 60.0) -> list:
    rng = np.random.default_rng(seed)
    ids = []
    for i in range(n_sessions):
        sid = f"dummy_s{i + 1:02d}"
        jchat, audio = make_session(rng, duration)
        tdir = os.path.join(out, "text", sid)
        adir = os.path.join(out, "audio_merged", sid)
        os.makedirs(tdir, exist_ok=True)
        os.makedirs(adir, exist_ok=True)
        with open(os.path.join(tdir, f"{sid}.json"), "w", encoding="utf-8") as f:
            json.dump(jchat, f, ensure_ascii=False)
        sf.write(os.path.join(adir, f"{sid}.wav"), audio, SR)
        ids.append(sid)
    return ids


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", default="data/dummy_zoom1")
    p.add_argument("--sessions", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    ids = make_dummy_corpus(args.out, args.sessions, args.seed)
    print(f"wrote {len(ids)} synthetic sessions to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
