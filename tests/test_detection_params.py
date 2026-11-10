"""``detection.<task>`` thresholds in the config reach the detectors."""

import pytest

from fdbja.config import Config
from fdbja.data import event_detection as ed


def _chars(speaker, text, start, step=0.2):
    return [{"speaker": speaker, "word": ch, "start": start + i * step,
             "end": start + (i + 1) * step} for i, ch in enumerate(text)]


def _dialogue():
    # A: 3.0 s, 0.5 s gap, 3.0 s -> one 6.5 s turn with a 0.5 s pause
    return _chars("A", "あいうえおかきくけこさしすせそ", 0.0) + \
        _chars("A", "たちつてとなにぬねのはひふへほ", 3.5)


def test_defaults_are_the_published_definition():
    cfg = Config({})
    assert cfg.detector_params("pause") == {}
    assert len(ed.detect("pause", _dialogue(), None, cfg.detector_params("pause"))) == 1


def test_config_thresholds_change_the_events():
    cfg = Config({"detection": {"pause": {"min_gap": 0.6}}})
    assert ed.detect("pause", _dialogue(), None, cfg.detector_params("pause")) == []


def test_unknown_threshold_keys_raise():
    cfg = Config({"detection": {"pause": {"min_gapp": 0.6}}})
    with pytest.raises(ValueError, match="min_gapp"):
        cfg.detector_params("pause")


def test_default_config_file_matches_code_defaults():
    cfg = Config.load("configs/default.yaml")
    for task in ("pause", "smooth_tt", "bc"):
        assert ed.detect(task, _dialogue(), None, cfg.detector_params(task)) == \
            ed.detect(task, _dialogue())
