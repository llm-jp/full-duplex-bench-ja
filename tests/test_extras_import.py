"""The optional extras import and reach their entry points (skipped when the
extra is not installed; the CI job ``extras`` installs the light ones)."""

import importlib

import pytest


@pytest.mark.parametrize("extra, modules, entry", [
    ("analysis", ["matplotlib", "pandas"], ("fdbja.turn_profile", "run")),
    ("always-talk", ["pyopenjtalk"], ("fdbja.adapters.always_talk_loop", None)),
    ("vap", ["torch", "torchaudio", "einops"], ("fdbja.vap.vap_realtime", "VAPRealtimeBackend")),
])
def test_extra_imports(extra, modules, entry):
    for m in modules:
        pytest.importorskip(m, reason=f"extra [{extra}] not installed")
    mod = importlib.import_module(entry[0])
    if entry[1]:
        assert hasattr(mod, entry[1])


def test_lazy_modules_import_without_extras():
    # modules that use an extra must import it lazily so the core install works
    for name in ("fdbja.turn_profile", "fdbja.adapters.always_talk_loop",
                 "fdbja.vap.vap_realtime", "fdbja.adapters.openai_realtime",
                 "fdbja.adapters.gemini_live", "fdbja.adapters.moshi_ws",
                 "fdbja.pause_analysis", "fdbja.analyze_human_eval", "fdbja.human_eval"):
        importlib.import_module(name)
