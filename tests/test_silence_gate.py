import numpy as np

from fdbja.asr.base import silence_gate

SR = 16000


def _hallucinating_asr():
    calls = []

    def fn():
        calls.append(1)
        return [{"speaker": "B", "word": "は", "start": 0.1, "end": 9.0},
                {"speaker": "B", "word": "い", "start": 9.0, "end": 19.6}]
    return fn, calls


def test_digital_silence_skips_the_asr():
    fn, calls = _hallucinating_asr()
    assert silence_gate(np.zeros(20 * SR, np.float32), SR, fn) == []
    assert calls == []


def test_dither_level_also_counts_as_silence():
    fn, calls = _hallucinating_asr()
    x = np.random.default_rng(0).normal(0, 1e-5, 20 * SR).astype(np.float32)
    assert silence_gate(x, SR, fn) == [] and calls == []


def test_real_signal_is_transcribed_unchanged_by_default():
    fn, calls = _hallucinating_asr()
    x = np.zeros(20 * SR, np.float32)
    x[SR:2 * SR] = 0.1
    assert len(silence_gate(x, SR, fn)) == 2 and calls == [1]


def test_drop_unvoiced_items_removes_characters_over_silence():
    x = np.zeros(20 * SR, np.float32)
    x[SR:2 * SR] = 0.1                    # voiced only 1-2 s

    def fn():
        return [{"speaker": "B", "word": "あ", "start": 1.2, "end": 1.4},
                {"speaker": "B", "word": "い", "start": 10.0, "end": 12.0}]
    out = silence_gate(x, SR, fn, drop_unvoiced_items=True)
    assert [it["word"] for it in out] == ["あ"]
