import numpy as np
import pytest

from fdbja.adapters.baselines import AlwaysTalkAdapter, SilenceAdapter
from fdbja.asr.base import frame_rms

SR = 16000


def _speechy(seconds=10.0, on=((1, 3), (5, 6.5)), level=0.2):
    x = np.zeros(int(seconds * SR), dtype=np.float32)
    rng = np.random.default_rng(0)
    for s, e in on:
        x[int(s * SR):int(e * SR)] = rng.normal(0, level, int((e - s) * SR))
    return x


def test_silence_is_digital_zero():
    x = _speechy()
    r = SilenceAdapter().infer(x, SR)
    assert r.response_audio.shape == x.shape and not r.response_audio.any()


def test_always_talk_default_loops_input_speech_without_gaps():
    x = _speechy()
    r = AlwaysTalkAdapter().infer(x, SR)
    y = r.response_audio
    assert y.shape == x.shape
    rms = frame_rms(y, SR, 0.05)
    assert (rms > 0.005).all()                     # active in every frame
    assert abs(float(np.sqrt((y.astype(np.float64) ** 2).mean())) - 0.035) < 1e-3


def test_always_talk_from_wav(tmp_path):
    sf = pytest.importorskip("soundfile")
    src = tmp_path / "loop.wav"
    sf.write(src, _speechy(4.0, on=((0.5, 2.0),)), SR)
    y = AlwaysTalkAdapter(source_wav=str(src)).infer(np.zeros(8 * SR, np.float32), SR).response_audio
    assert (frame_rms(y, SR, 0.05) > 0.005).all()


def test_always_talk_rejects_a_silent_source():
    with pytest.raises(ValueError):
        AlwaysTalkAdapter().infer(np.zeros(SR, np.float32), SR)
    with pytest.raises(ValueError):
        AlwaysTalkAdapter(source="wav")


def test_noise_mode_is_still_available():
    y = AlwaysTalkAdapter(source="noise", level=0.05).infer(np.zeros(SR, np.float32), SR).response_audio
    assert y.std() > 0


@pytest.mark.parametrize("join", ["concat", "crossfade", "longest"])
def test_always_talk_join_modes_fill_the_clip(join):
    x = _speechy()
    y = AlwaysTalkAdapter(join=join).infer(x, SR).response_audio
    assert y.shape == x.shape
    assert (frame_rms(y, SR, 0.05)[1:-1] > 0.002).all()


def test_unknown_join_is_rejected():
    with pytest.raises(ValueError):
        AlwaysTalkAdapter(join="splice")
