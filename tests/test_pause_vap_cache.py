"""analyze-pause reads the VAP cache where score-offline writes it."""

import numpy as np
import pytest

from fdbja.pause_analysis import _cached_dkl, vap_cache_root


def _write(path, probs):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, times=np.arange(len(probs)) * 0.05 + 0.05, probs=probs)


def test_reads_the_checkpoint_keyed_cache(tmp_path):
    p = np.full((40, 256), 1 / 256)
    root = tmp_path / "vap_cache" / "vap_state_dict_jp_kyoto_20hz_2500msec"
    _write(root / "gt" / "pause" / "c.npz", p)
    _write(root / "m" / "c.npz", p)
    cache = vap_cache_root(tmp_path)
    assert cache == root
    (d,) = _cached_dkl(cache, "m", "c", [{"start": 0.5, "end": 1.0}])
    assert d == pytest.approx(0.0, abs=1e-6)


def test_several_checkpoints_need_a_choice(tmp_path):
    for stem in ("a", "b"):
        (tmp_path / "vap_cache" / stem / "gt" / "pause").mkdir(parents=True)
    with pytest.raises(ValueError, match="--vap-checkpoint"):
        vap_cache_root(tmp_path)
    assert vap_cache_root(tmp_path, "b.pt").name == "b"
