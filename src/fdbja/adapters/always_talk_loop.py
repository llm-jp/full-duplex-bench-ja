"""The speech loop of the ``always_talk`` trivial baseline, generated on first use.

The loop is synthesised with Open JTalk (``pyopenjtalk``, Modified BSD) and the
HTS voice "Mei" (Nagoya Institute of Technology, CC BY 3.0; attribution in
``fdbja/assets/README.md``) from nine sentences written for this toolkit, and
cached as a 16 kHz mono FLAC so that no audio is stored in the repository.

Why speech and why a fixed loop: binary TOR reads an ASR transcript, so noise
is not transcribed, and a loop cut from the input clip makes the ASR split it
at the seams (smooth_tt TOR 0.20-0.40 instead of ~0). With this loop the
held-out check (20 pause / 10 smooth_tt clips, reazonspeech_whisperx) gave
Pause TOR 1.000 / Smooth TT TOR 0.000, the level of the published v4_eval
baseline (0.992 / 0.005).
"""

from __future__ import annotations

import os

import numpy as np

SAMPLE_RATE = 16000
LEVEL = 0.035          # RMS, = the v4_eval loop (0.0345)
GAP_S = 0.08           # silence between sentences
FILE_NAME = "always_talk_loop_ja.flac"
INSTALL_HINT = ("pip install pyopenjtalk  (or uv pip install pyopenjtalk; it downloads the "
                "Open JTalk dictionary on first use), or pass source_wav=<speech file>")

TEXT = (
    "今日は朝から少し肌寒くて、駅まで歩くあいだに上着を持ってくればよかったと思いました",
    "そういえば先週話していた新しいお店ですが、昼休みに行ってみたら思ったより空いていました",
    "メニューは定食が中心で、魚の煮付けがとてもおいしかったので、また今度一緒に行きましょう",
    "それから来月の予定なんですけど、週末のどこかで一日くらい遠出ができたらいいなと考えています",
    "電車で一時間くらいのところに、景色のきれいな公園があるそうなので、そこも候補に入れています",
    "天気が良ければお弁当を持っていって、のんびり散歩するのも楽しそうですよね",
    "あとは仕事のほうも少し落ち着いてきたので、前から読みたかった本をやっと読み始めました",
    "最初は難しそうだと思っていたんですが、読んでみると説明が丁寧で、思ったより進みが早いです",
    "週に一冊くらいのペースで読めたら、年末までにかなりの数になるんじゃないかと思っています",
)


def cache_path() -> str:
    """``$FDBJA_CACHE_DIR`` or ``~/.cache/fdbja``, plus the file name."""
    root = os.environ.get("FDBJA_CACHE_DIR") or os.path.join(
        os.path.expanduser("~"), ".cache", "fdbja")
    return os.path.join(root, FILE_NAME)


def _decimate3(x: np.ndarray) -> np.ndarray:
    """48 kHz -> 16 kHz: windowed-sinc low-pass (cut-off 7.2 kHz, 255 taps), keep every 3rd sample."""
    n = np.arange(-127, 128)
    fc = 7200.0 / 48000.0
    h = 2 * fc * np.sinc(2 * fc * n) * np.blackman(len(n))
    h /= h.sum()
    return np.convolve(x, h, mode="same")[::3]


def synthesize() -> np.ndarray:
    """Return the loop (float32, 16 kHz, RMS ``LEVEL``). Needs ``pyopenjtalk``."""
    try:
        import pyopenjtalk
    except ImportError as e:
        raise RuntimeError(
            "always_talk needs Open JTalk to build its speech loop on first use: "
            + INSTALL_HINT) from e
    segs = []
    for t in TEXT:
        x, sr = pyopenjtalk.tts(t)          # 48 kHz, voice "Mei"
        if sr != 48000:
            raise RuntimeError(f"unexpected Open JTalk sample rate {sr}")
        x = np.asarray(x, dtype=np.float64) / 32768.0
        on = np.abs(x) > 0.01 * np.abs(x).max()
        i, j = int(np.argmax(on)), len(on) - int(np.argmax(on[::-1]))
        segs += [x[i:j], np.zeros(int(GAP_S * sr))]
    y = _decimate3(np.concatenate(segs))
    y *= LEVEL / np.sqrt((y ** 2).mean())
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def ensure_loop(path: str = None) -> str:
    """Path of the cached loop, synthesising it first if it does not exist."""
    path = path or cache_path()
    if not os.path.exists(path):
        import soundfile as sf
        y = synthesize()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp.flac"
        sf.write(tmp, y, SAMPLE_RATE, subtype="PCM_16")
        os.replace(tmp, path)
    return path
