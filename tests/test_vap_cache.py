"""VAP probability cache of score-offline: keyed by the input audio and the checkpoint."""

import numpy as np
import pytest
import soundfile as sf

from fdbja.runner import _vap_cache_tag, _vap_distance_events, vap_cache_key


class _CountingVAP:
    """Stand-in backend: probabilities from the channel energies; counts calls."""

    def __init__(self):
        self.calls = 0

    def probs(self, left, right):
        self.calls += 1
        hop = 800
        n = min(len(left), len(right)) // hop
        e = np.sqrt((right[:n * hop].reshape(n, hop) ** 2).mean(1))
        p = np.full((n, 256), 1.0)
        p[:, 255] += 50 * e
        return np.arange(n) / 20 + 0.05, p / p.sum(1, keepdims=True)


def _wav(path, b_level, sr=16000, sec=3.0):
    rng = np.random.default_rng(0)
    a = 0.1 * rng.standard_normal(int(sr * sec))
    b = b_level * rng.standard_normal(int(sr * sec))
    sf.write(path, np.stack([a, b], 1).astype(np.float32), sr)


EV = [{"start": 1.0, "end": 2.0, "duration": 1.0}]
CFG = {"vap_model": "x.pt", "cpc_model": "c.pt"}


def _run(backend, tmp_path, tag):
    return _vap_distance_events(backend, str(tmp_path / "gt.wav"), str(tmp_path / "m.wav"),
                                "pause", EV, str(tmp_path / "cache"), "m", "clip",
                                cache_tag=tag)


def test_cache_is_reused_while_inputs_are_unchanged(tmp_path):
    _wav(tmp_path / "gt.wav", 0.0)
    _wav(tmp_path / "m.wav", 0.1)
    vap = _CountingVAP()
    tag = _vap_cache_tag(CFG)
    first, _ = _run(vap, tmp_path, tag)
    assert vap.calls == 2
    again, _ = _run(vap, tmp_path, tag)
    assert vap.calls == 2 and again == first
    assert (tmp_path / "cache" / "m" / "pause" / "clip.npz").exists()


def test_cache_is_recomputed_when_the_output_changes(tmp_path):
    _wav(tmp_path / "gt.wav", 0.0)
    _wav(tmp_path / "m.wav", 0.1)
    vap = _CountingVAP()
    tag = _vap_cache_tag(CFG)
    first, _ = _run(vap, tmp_path, tag)
    _wav(tmp_path / "m.wav", 0.0)        # re-run infer: the model now stays silent
    second, _ = _run(vap, tmp_path, tag)
    assert vap.calls == 3                # only the model side is recomputed
    assert second[0]["mean_dkl"] < first[0]["mean_dkl"]


def test_cache_is_recomputed_when_the_checkpoint_changes(tmp_path):
    _wav(tmp_path / "gt.wav", 0.0)
    _wav(tmp_path / "m.wav", 0.1)
    ck1, ck2 = tmp_path / "a.pt", tmp_path / "b.pt"
    ck1.write_bytes(b"weights-1")
    ck2.write_bytes(b"weights-2")
    vap = _CountingVAP()
    _run(vap, tmp_path, _vap_cache_tag(CFG, str(ck1)))
    _run(vap, tmp_path, _vap_cache_tag(CFG, str(ck2)))
    assert vap.calls == 4
    _run(vap, tmp_path, _vap_cache_tag({**CFG, "vap_revision": "fe24ac6"}, str(ck2)))
    assert vap.calls == 6


def test_trust_mode_reuses_legacy_files(tmp_path):
    _wav(tmp_path / "gt.wav", 0.0)
    _wav(tmp_path / "m.wav", 0.1)
    vap = _CountingVAP()
    assert _vap_cache_tag({**CFG, "cache": "trust"}) is None
    _run(vap, tmp_path, None)
    _wav(tmp_path / "m.wav", 0.0)
    _run(vap, tmp_path, None)            # stale file is used as before 0.2.0
    assert vap.calls == 2
    assert (tmp_path / "cache" / "m" / "clip.npz").exists()
    # a file without a key (written before 0.2.0) is recomputed in validate mode
    _run(vap, tmp_path, _vap_cache_tag(CFG))
    assert vap.calls == 4


def test_key_depends_on_bytes_and_tag(tmp_path):
    _wav(tmp_path / "a.wav", 0.1)
    _wav(tmp_path / "b.wav", 0.2)
    assert vap_cache_key(str(tmp_path / "a.wav"), "t") == vap_cache_key(str(tmp_path / "a.wav"), "t")
    assert vap_cache_key(str(tmp_path / "a.wav"), "t") != vap_cache_key(str(tmp_path / "b.wav"), "t")
    assert vap_cache_key(str(tmp_path / "a.wav"), "t") != vap_cache_key(str(tmp_path / "a.wav"), "u")


def test_unknown_cache_mode_raises():
    with pytest.raises(ValueError, match="validate\\|trust"):
        _vap_cache_tag({**CFG, "cache": "always"})


def test_analyze_pause_reads_the_per_task_layout(tmp_path):
    from fdbja.pause_analysis import _cached_dkl

    p = np.full((40, 256), 1 / 256)
    root = tmp_path / "vap_cache" / "x"
    for sub in ("gt/pause", "m/pause"):
        (root / sub).mkdir(parents=True)
        np.savez_compressed(root / sub / "c.npz", times=np.arange(40) * 0.05 + 0.05, probs=p)
    (d,) = _cached_dkl(root, "m", "c", [{"start": 0.5, "end": 1.0}])
    assert d == pytest.approx(0.0, abs=1e-6)
