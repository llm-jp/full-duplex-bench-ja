import numpy as np

from fdbja.metrics.tor import eval_smooth_tt
from fdbja.summary import bootstrap_ci, stream_seed, timing_stats


def test_bootstrap_ci_is_deterministic_and_brackets_the_mean():
    x = np.r_[np.ones(30), np.zeros(70)]
    a = bootstrap_ci(x, 2000, seed=1)
    assert a == bootstrap_ci(x, 2000, seed=1)
    assert a[0] < 0.3 < a[1] and 0.15 < a[0] and a[1] < 0.45
    assert bootstrap_ci([], 100) is None
    assert bootstrap_ci([1, 1, 1], 100) == [1.0, 1.0]


def test_stream_seed_is_per_task_model():
    assert stream_seed(0, "pause", "a") == stream_seed(0, "pause", "a")
    assert stream_seed(0, "pause", "a") != stream_seed(0, "pause", "b")


def test_signed_onset_keeps_overlaps():
    ev = [{"a_end": 5.0}, {"a_end": 10.0}, {"a_end": 15.0}]
    utts = [{"start": 4.7, "end": 6.0, "num_chars": 10},     # starts 0.3 s before A ends
            {"start": 10.4, "end": 12.0, "num_chars": 10}]
    res = eval_smooth_tt(ev, utts)
    assert [r["TO"] for r in res] == [1, 1, 0]
    assert res[0]["latency"] == 0.0 and abs(res[0]["onset"] + 0.3) < 1e-9
    assert abs(res[1]["onset"] - 0.4) < 1e-9 and res[2]["onset"] is None
    st = timing_stats([{**r} for r in res])
    assert st["lat_n"] == 2 and st["overlap_rate"] == 0.5
    assert st["onset_q1"] < 0 < st["onset_q3"]


def test_scoring_conditions_records_checkpoint_hash(tmp_path):
    import hashlib

    from fdbja.config import Config
    from fdbja.summary import scoring_conditions

    ck = tmp_path / "ckpt.pt"
    ck.write_bytes(b"weights")
    cfg = Config({"detection": {"boundaries": "audio"},
                  "metrics": {"vap_distance": {"enabled": True, "vap_model": "ckpt.pt"}}})
    c = scoring_conditions(cfg, {"pause/m": "my_asr", "pause/x": None}, str(ck))
    assert c["boundaries"] == "audio" and c["system_activity"] == "transcript"
    assert c["vap_checkpoint"] == f"ckpt.pt (sha256 {hashlib.sha256(b'weights').hexdigest()[:12]})"
    assert c["asr"].startswith("my_asr; unknown for 1")
