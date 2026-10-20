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


def test_always_talk_default_uses_the_cached_loop(tmp_path, monkeypatch):
    sf = pytest.importorskip("soundfile")
    monkeypatch.setenv("FDBJA_CACHE_DIR", str(tmp_path))
    sf.write(tmp_path / "always_talk_loop_ja.flac", _speechy(4.0, on=((0.5, 3.5),)), SR)
    a = AlwaysTalkAdapter()
    assert a.source == "wav" and a.source_wav == str(tmp_path / "always_talk_loop_ja.flac")
    x = _speechy()
    y = a.infer(x, SR).response_audio
    assert y.shape == x.shape
    rms = frame_rms(y, SR, 0.05)
    assert (rms > 0.005).all()                     # active in every frame
    assert abs(float(np.sqrt((y.astype(np.float64) ** 2).mean())) - 0.035) < 1e-3


def test_always_talk_default_without_open_jtalk_asks_for_source_wav(tmp_path, monkeypatch):
    import builtins
    monkeypatch.setenv("FDBJA_CACHE_DIR", str(tmp_path))      # empty cache
    real_import = builtins.__import__

    def no_pyopenjtalk(name, *args, **kwargs):
        if name == "pyopenjtalk":
            raise ImportError("No module named 'pyopenjtalk'")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_pyopenjtalk)
    with pytest.raises(RuntimeError, match="source_wav"):
        AlwaysTalkAdapter()


def test_loop_synthesis_matches_the_spec(tmp_path):
    pytest.importorskip("pyopenjtalk")
    from fdbja.adapters.always_talk_loop import ensure_loop
    sf = pytest.importorskip("soundfile")
    y, sr = sf.read(ensure_loop(str(tmp_path / "loop.flac")))
    assert sr == 16000 and 50 < len(y) / sr < 70
    assert abs(float(np.sqrt((y ** 2).mean())) - 0.035) < 2e-3


def test_always_talk_from_wav(tmp_path):
    sf = pytest.importorskip("soundfile")
    src = tmp_path / "loop.wav"
    sf.write(src, _speechy(4.0, on=((0.5, 2.0),)), SR)
    y = AlwaysTalkAdapter(source_wav=str(src)).infer(np.zeros(8 * SR, np.float32), SR).response_audio
    assert (frame_rms(y, SR, 0.05) > 0.005).all()


def test_always_talk_rejects_a_silent_source():
    with pytest.raises(ValueError):
        AlwaysTalkAdapter(source="input_loop").infer(np.zeros(SR, np.float32), SR)
    with pytest.raises(ValueError):
        AlwaysTalkAdapter(source="wav")


def test_noise_mode_is_still_available():
    y = AlwaysTalkAdapter(source="noise", level=0.05).infer(np.zeros(SR, np.float32), SR).response_audio
    assert y.std() > 0


@pytest.mark.parametrize("join", ["concat", "crossfade", "longest"])
def test_always_talk_join_modes_fill_the_clip(join):
    x = _speechy()
    y = AlwaysTalkAdapter(source="input_loop", join=join).infer(x, SR).response_audio
    assert y.shape == x.shape
    assert (frame_rms(y, SR, 0.05)[1:-1] > 0.002).all()


def test_unknown_join_is_rejected():
    with pytest.raises(ValueError):
        AlwaysTalkAdapter(join="splice")
