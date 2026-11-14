"""Full-Duplex-Bench-JA: evaluation toolkit for Japanese full-duplex
spoken dialogue models (turn-taking)."""

__version__ = "0.1.0"

#: public entry points of ``fdbja.measurement``, imported on first use so that
#: ``import fdbja`` stays free of numpy
_MEASUREMENT = ("measure", "load_profile", "make_profile", "Measurement", "MeasureConfig")
__all__ = ["__version__", *_MEASUREMENT]


def __getattr__(name):
    if name in _MEASUREMENT:
        from fdbja import measurement
        return getattr(measurement, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
