"""Is this recording inside the conditions ``fdbja measure`` was checked for?

``measure`` finds speech from the frame energy alone. Noise, the other
speaker's voice leaking into a channel, clipping and a very low recording
level all move its numbers, each in a known direction (README, "Supported
recording conditions"). This module estimates those properties from the
recording itself, per channel, and turns them into warnings. It reads the
recording and the detector's own voice activity; it changes nothing that
``measure`` reports about the timing.

Estimates (:func:`recording_quality`), per channel:

  * ``active_level_db``: the ITU-T P.56 style active level of the channel,
    the level the voice threshold is derived from;
  * ``background_level_db``: the median level of the frames the detector
    calls silent, and ``snr_db`` = active level - background level;
  * ``voiced_share``, ``digital_silence_share``: the share of frames the
    detector calls voice, and the share of frames that are digital zero;
  * ``clipped_share``: the share of samples stuck at the channel's peak value;
  * ``threshold_db``, ``threshold_set_by``: the voice threshold and which
    clause set it (``speech_level`` = the P.56 threshold, ``noise_floor`` =
    the noise floor + ``min_snr_db``, ``min_level_db`` = the fixed lower
    limit);

and for each direction of leak between the channels (``system_to_user`` = the
system's voice in the user's channel), :func:`leak_estimates`:

  * ``level_db``: the gain of the delayed copy of the other channel, from the
    cross-correlation of the two channels within +-``LEAK_MAX_LAG`` seconds;
  * ``correlation``: the normalised maximum of that cross-correlation;
  * ``lag_seconds``, ``seconds``: where the maximum is, and how much audio the
    estimate rests on;
  * ``above_threshold_db``: the level a leak of that size would have in the
    receiving channel (the source's speech level + ``level_db``) minus the
    receiving channel's voice threshold. A channel with little voice has a
    low threshold, so that a weak leak reaches it.

The cross-correlation is taken where a leak can be seen and the two speakers
cannot be confused: over the frames in which the *other* channel has just been
voiced and *this* channel is at least ``LEAK_MASK_MARGIN_DB`` quieter than it
(single talk). Two people who speak at the same time are correlated by chance
over a short recording, and without this restriction that chance correlation
is read as a leak.

Every threshold is in one place below, with what it rests on. All of them come
from recordings degraded artificially (added stationary noise, a delayed copy
of the other channel at a fixed gain, a fixed gain with clipping); none was
checked on the noise, reverberation or coding of real rooms and lines.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from fdbja.data.event_boundaries import (
    DB_FLOOR,
    VADConfig,
    _runs,
    active_level_db,
    threshold_db,
)

# ---------------------------------------------------------------------------
# How the leak is estimated (not thresholds)
# ---------------------------------------------------------------------------

#: the cross-correlation is searched within this many seconds either way
LEAK_MAX_LAG = 0.2
#: it is computed on the channels averaged down to about this sampling rate
#: (cost; a leak is a broadband copy, so the band below 4 kHz shows it)
LEAK_RATE = 8000.0
#: first-order pre-emphasis before the correlation (flattens the speech
#: spectrum, which sharpens the correlation peak)
LEAK_PREEMPHASIS = 0.97
#: single talk: this channel is at least this much quieter than the other
#: channel's recent voiced frames
LEAK_MASK_MARGIN_DB = 6.0
#: no estimate from less single talk than this
LEAK_MIN_SECONDS = 1.0
#: FFT length of one block and the number of blocks used at most. A leak is a
#: property of the set-up, so a long recording is sampled in blocks spread
#: evenly over it instead of being read in full (about 8 s per block at 8 kHz)
LEAK_FFT = 1 << 16
LEAK_MAX_BLOCKS = 64
#: a sample within this relative distance of the channel's peak is "at the peak"
CLIP_PEAK_TOLERANCE = 1e-4

CHANNELS = ("user", "system")
LEAK_DIRECTIONS = ("system_to_user", "user_to_system")


def _r(x: Optional[float], digits: int = 2) -> Optional[float]:
    return None if x is None or not np.isfinite(x) else round(float(x), digits)


# ---------------------------------------------------------------------------
# One channel
# ---------------------------------------------------------------------------

def _clipping(x: np.ndarray) -> Tuple[float, Optional[float]]:
    """(share of samples at the peak value, peak in dBFS)."""
    n = len(x)
    if n == 0:
        return 0.0, None
    peak = max(float(x.max()), -float(x.min()))
    if not peak > 0:
        return 0.0, None
    lim = peak * (1.0 - CLIP_PEAK_TOLERANCE)
    stuck = int(np.count_nonzero(x >= lim)) + int(np.count_nonzero(x <= -lim))
    return stuck / n, 20.0 * float(np.log10(peak))


def _mean_level(db: np.ndarray) -> float:
    """Level of the mean power of frames given in dB."""
    return 10.0 * float(np.log10(np.mean(10.0 ** (db / 10.0))))


def channel_quality(x: np.ndarray, va: np.ndarray, db: np.ndarray, thr: float,
                    vad: VADConfig, min_level_db: Optional[float]) -> Dict[str, Any]:
    """Estimates for one channel. ``va``, ``db``, ``thr`` are what
    ``voice_activity_levels`` returned for it."""
    n = len(db)
    n_live = int(np.count_nonzero(db > DB_FLOOR + 1.0))
    clipped, peak = _clipping(np.asarray(x))
    out: Dict[str, Any] = {
        "speech_level_db": None, "background_level_db": None, "snr_db": None,
        "active_level_db": None,
        "voiced_share": _r(float(va.mean()), 4) if n else 0.0,
        "digital_silence_share": _r(1.0 - n_live / n, 4) if n else None,
        "clipped_share": _r(clipped, 6), "peak_db": _r(peak),
        "threshold_db": None, "threshold_set_by": None,
    }
    if not n_live:
        return out
    act, p56 = active_level_db(db, vad.margin_db)
    detector = threshold_db(db, vad)
    if min_level_db is not None and float(min_level_db) > detector:
        by = "min_level_db"
    elif vad.mode == "fixed":
        by = "fixed_db"
    elif vad.mode == "floor":
        by = "noise_floor"
    else:
        by = "speech_level" if p56 >= detector else "noise_floor"
    out.update({"active_level_db": _r(act), "threshold_db": _r(thr), "threshold_set_by": by})
    speech = _mean_level(db[va]) if va.any() else None
    out["speech_level_db"] = _r(speech)
    silent = db[~va]
    if len(silent):
        out["background_level_db"] = _r(float(np.median(silent)))
        if speech is not None:
            out["snr_db"] = _r(speech - float(np.median(silent)))
    return out


# ---------------------------------------------------------------------------
# Leak between the channels
# ---------------------------------------------------------------------------

def _around(a: np.ndarray, j: int, fill: Any) -> np.ndarray:
    """Maximum of ``a`` over the frames ``f - j .. f + j``, for every ``f``."""
    pad = np.full(j, fill, dtype=a.dtype)
    win = np.lib.stride_tricks.sliding_window_view(np.concatenate([pad, a, pad]), 2 * j + 1)
    return win.max(axis=1)


def _single_talk(db_src: np.ndarray, va_src: np.ndarray, db_dst: np.ndarray,
                 frame: float, margin_db: float) -> np.ndarray:
    """Frames in which the source has been voiced within ``LEAK_MAX_LAG`` and
    the destination is at least ``LEAK_MASK_MARGIN_DB`` below the loudest of
    those voiced source frames."""
    n = min(len(db_src), len(db_dst), len(va_src))
    if n == 0:
        return np.zeros(0, dtype=bool)
    j = max(1, int(np.ceil(LEAK_MAX_LAG / frame - 1e-9)))
    va = va_src[:n].astype(bool)
    level = _around(np.where(va, db_src[:n], DB_FLOOR), j, DB_FLOOR)
    return _around(va, j, False) & (db_dst[:n] <= level - margin_db)


def _segment(x: np.ndarray, a: int, b: int, q: int, n_d: int) -> np.ndarray:
    """Samples ``a .. b`` of ``x`` averaged over ``q`` and pre-emphasised;
    zero outside the signal."""
    out = np.zeros(b - a)
    lo, hi = max(a, 0), min(b, n_d)
    if hi <= lo:
        return out
    first = max(lo - 1, 0)
    seg = np.asarray(x[first * q:hi * q], dtype=np.float64)
    if q > 1:
        seg = seg.reshape(-1, q).mean(axis=1)
    y = seg.copy()
    y[1:] -= LEAK_PREEMPHASIS * seg[:-1]
    out[lo - a:hi - a] = y[1:] if first < lo else y
    return out


class _Direction:
    """Sums for one direction of leak (source -> destination)."""

    def __init__(self, lag: int) -> None:
        self.r = np.zeros(2 * lag + 1)        # sum dst[t] src[t - k], k = -lag .. lag
        self.e_src = np.zeros(2 * lag + 1)    # sum src[t - k]^2 over the same t
        self.e_dst = 0.0
        self.n = 0

    def add(self, dst: np.ndarray, src: np.ndarray, f_src: np.ndarray,
            m: np.ndarray, lag: int) -> None:
        nfft, block = len(src), len(m)
        d = np.zeros(nfft)
        d[:block] = dst[lag:lag + block] * m
        c = np.fft.irfft(np.conj(np.fft.rfft(d)) * f_src, nfft)
        self.r += c[:2 * lag + 1][::-1]
        self.e_dst += float(np.dot(d, d))
        self.n += int(np.count_nonzero(m))
        # sum over the masked t of src[t - k]^2, from the cumulative power
        cum = np.concatenate([[0.0], np.cumsum(src * src)])
        runs = [(s, e) for s, e, v in _runs(m) if v]
        starts = np.array([s for s, _e in runs])[:, None]
        ends = np.array([e for _s, e in runs])[:, None]
        shift = lag - np.arange(-lag, lag + 1)[None, :]
        self.e_src += (cum[ends + shift] - cum[starts + shift]).sum(axis=0)

    def result(self, seconds_per_sample: float) -> Dict[str, Any]:
        seconds = self.n * seconds_per_sample
        out = {"level_db": None, "correlation": None, "peak_to_floor": None,
               "lag_seconds": None, "seconds": _r(seconds, 1)}
        ok = self.e_src > 0
        if seconds < LEAK_MIN_SECONDS or not ok.any():
            return out
        if not self.e_dst > 0:
            # digital silence wherever the other speaker talks: nothing leaks
            out.update({"level_db": DB_FLOOR, "correlation": 0.0})
            return out
        score = np.where(ok, self.r ** 2 / np.where(ok, self.e_src, 1.0), -1.0)
        k = int(np.argmax(score))
        gain = abs(self.r[k]) / self.e_src[k]
        lag = (len(self.r) - 1) // 2
        floor = float(np.median(np.sqrt(score[ok])))
        out.update({
            "peak_to_floor": _r(np.sqrt(score[k]) / floor, 1) if floor > 0 else None,
            "level_db": _r(max(20.0 * np.log10(max(gain, 1e-12)), DB_FLOOR), 1),
            "correlation": _r(min(1.0, abs(self.r[k]) / np.sqrt(self.e_dst * self.e_src[k])), 3),
            "lag_seconds": _r((k - lag) * seconds_per_sample, 4),
        })
        return out


def leak_estimates(user: np.ndarray, system: np.ndarray, sr: int, frame: float,
                   db_user: np.ndarray, db_system: np.ndarray,
                   va_user: np.ndarray, va_system: np.ndarray,
                   margin_db: float = LEAK_MASK_MARGIN_DB) -> Dict[str, Dict[str, Any]]:
    """The other speaker's voice in each channel (see the module docstring).

    ``level_db`` is the gain of the best delayed copy: -30 means the other
    speaker is heard 30 dB below their own channel. ``DB_FLOOR`` = the channel
    is digitally silent whenever the other speaker talks; ``None`` = too
    little single talk to tell.
    """
    q = max(1, int(round(sr / LEAK_RATE)))
    n_d = min(len(user), len(system)) // q
    lag = max(1, int(round(LEAK_MAX_LAG * sr / q)))
    n_frame = int(round(sr * frame))
    masks = {"system_to_user": _single_talk(db_system, va_system, db_user, frame, margin_db),
             "user_to_system": _single_talk(db_user, va_user, db_system, frame, margin_db)}
    acc = {k: _Direction(lag) for k in LEAK_DIRECTIONS}
    nfft = LEAK_FFT
    while nfft < 4 * lag:
        nfft *= 2
    while nfft // 2 >= max(n_d + 2 * lag, 4 * lag):
        nfft //= 2
    block = nfft - 2 * lag
    n_frames = min(len(m) for m in masks.values())

    def mask_of(name: str, a: int) -> np.ndarray:
        t = np.arange(a, a + block)
        f = (t * q) // n_frame
        ok = (t < n_d) & (f < n_frames)
        return masks[name][np.minimum(f, max(n_frames - 1, 0))] & ok if n_frames else ok & False

    starts = []
    for a in range(0, n_d, block):
        f0, f1 = (a * q) // n_frame, min(n_frames, ((a + block) * q) // n_frame + 1)
        if any(m[f0:f1].any() for m in masks.values()):
            starts.append(a)
    if len(starts) > LEAK_MAX_BLOCKS:
        pick = np.unique(np.round(np.linspace(0, len(starts) - 1, LEAK_MAX_BLOCKS)).astype(int))
        starts = [starts[i] for i in pick]
    for a in starts:
        seg = {"user": _segment(user, a - lag, a + block + lag, q, n_d),
               "system": _segment(system, a - lag, a + block + lag, q, n_d)}
        spec: Dict[str, np.ndarray] = {}
        for name, dst, src in (("system_to_user", "user", "system"),
                               ("user_to_system", "system", "user")):
            m = mask_of(name, a)
            if not m.any():
                continue
            if src not in spec:
                spec[src] = np.fft.rfft(seg[src])
            acc[name].add(seg[dst], seg[src], spec[src], m, lag)
    out = {k: acc[k].result(q / sr) for k in LEAK_DIRECTIONS}
    for name, va_dst in (("system_to_user", va_user), ("user_to_system", va_system)):
        m = masks[name][:n_frames]
        out[name]["detected_share"] = (
            _r(float((m & va_dst[:n_frames].astype(bool)).sum()) / int(m.sum()), 4)
            if m.any() else None)
    return out


# ---------------------------------------------------------------------------
# One recording
# ---------------------------------------------------------------------------

def recording_quality(user: np.ndarray, system: np.ndarray, sr: int, vad: VADConfig,
                      min_level_db: Optional[float],
                      activity_user: Tuple[np.ndarray, np.ndarray, float],
                      activity_system: Tuple[np.ndarray, np.ndarray, float]
                      ) -> Dict[str, Any]:
    """Estimates for one recording: ``user``, ``system`` (:func:`channel_quality`)
    and ``leak`` (:func:`leak_estimates`). ``activity_*`` are the
    ``(voice activity, frame levels, threshold)`` of the two channels."""
    va_u, db_u, thr_u = activity_user
    va_s, db_s, thr_s = activity_system
    q = {
        "user": channel_quality(user, va_u, db_u, thr_u, vad, min_level_db),
        "system": channel_quality(system, va_s, db_s, thr_s, vad, min_level_db),
        "leak": leak_estimates(user, system, sr, vad.frame, db_u, db_s, va_u, va_s),
    }
    for d in LEAK_DIRECTIONS:
        q["leak"][d]["above_threshold_db"] = leak_above_threshold(q, d)
    return q


# ---------------------------------------------------------------------------
# Thresholds of the warnings (the only place they are set)
# ---------------------------------------------------------------------------
# Each comes from measuring the same recordings clean and degraded and asking
# where the statistics of ``measure`` leave their tolerance. Degradations:
# stationary white and pink noise at a fixed SNR, a copy of the other channel
# delayed by 0.05 s at a fixed gain, a fixed gain with 16-bit clipping.

#: noise: warn when the speech is less than this far above the background.
#: With added stationary noise the statistics held at an SNR of 25 dB (white
#: and pink), moved for part of the systems at 20 dB (pink) and for all of them
#: at 15 dB. The estimate follows the SNR of the added noise to within about
#: 3 dB, so the threshold sits between 25 and 20: recordings at 25 dB stay
#: clear of it and recordings at 20 dB are all warned.
NOISE_MIN_SNR_DB = 22.0
#: clipping: warn when at least this share of the samples of a channel with
#: voice is at the peak value. With +12 dB of gain (statistics moved for part
#: of the systems) nearly all recordings are above this share; with +6 dB
#: (statistics held) most are below it.
CLIP_MAX_SHARE = 0.001
#: leak: warn when the other speaker is heard at this level or above. A copy
#: of the system's voice in the user's channel left the statistics alone at
#: -30 dB and moved them for part of the systems at -25 dB; the level is
#: recovered to within about 1 dB, so the threshold is halfway.
LEAK_MAX_LEVEL_DB = -27.5
#: ... or, whatever its level, when the leaked voice comes this close to the
#: voice threshold of the channel it leaks into (level of the other speaker +
#: leak level - threshold). A channel with little or no voice of its own has a
#: low threshold, and a leak of -40 dB then counts as its speech: the
#: statistics of recordings with a silent system channel moved at -40 dB.
LEAK_MAX_ABOVE_THRESHOLD_DB = -5.0
#: ... and, in both cases, when the estimate is more than the chance
#: correlation of two speakers (below this the level is not a leak's)
LEAK_MIN_CORRELATION = 0.05
#: an "other speaker" this much louder in this channel than in its own is not
#: a leak into this channel: the leak runs the other way
LEAK_REVERSED_DB = 6.0
#: quiet channel: a note (not a warning) when a channel has voice in less than
#: this share of its frames. With a leak in both directions, the statistics of
#: recordings whose system channel had no voice at all moved at -40 dB. A
#: channel with a little voice mostly held at -40 and -35 dB, and moved at
#: -30 dB more often than a channel with more voice; the less voice, the more
#: often. The value marks "a little" and was set by hand, it is not a measured
#: limit.
QUIET_MAX_VOICED_SHARE = 0.02

THRESHOLDS: Dict[str, float] = {
    "noise_min_snr_db": NOISE_MIN_SNR_DB,
    "clip_max_share": CLIP_MAX_SHARE,
    "leak_max_level_db": LEAK_MAX_LEVEL_DB,
    "leak_max_above_threshold_db": LEAK_MAX_ABOVE_THRESHOLD_DB,
    "leak_min_correlation": LEAK_MIN_CORRELATION,
    "quiet_max_voiced_share": QUIET_MAX_VOICED_SHARE,
}

#: flags that are notes (the recording is usable, but fragile) rather than
#: warnings (the numbers are probably shifted)
NOTE_FLAGS = ("quiet_channel",)


def _other(channel: str) -> str:
    return "system" if channel == "user" else "user"


def _source(direction: str) -> Tuple[str, str]:
    """``(source, destination)`` channel of a leak direction."""
    src, _, dst = direction.partition("_to_")
    return src, dst


def leak_above_threshold(q: Dict[str, Any], direction: str) -> Optional[float]:
    """How far the leaked voice is from being counted as speech, in dB: the
    speech level of the source channel plus the leak level, minus the voice
    threshold of the destination channel. None when one of them is unknown."""
    src, dst = _source(direction)
    level = q["leak"][direction]["level_db"]
    speech, thr = q[src]["speech_level_db"], q[dst]["threshold_db"]
    if level is None or speech is None or thr is None:
        return None
    return _r(speech + level - thr, 1)


def _leaks(q: Dict[str, Any], direction: str) -> Optional[str]:
    """Why the estimate of this direction is a suspected leak: ``"level"``
    (strong in absolute terms), ``"threshold"`` (weak, but close to the voice
    threshold of a quiet channel) or None."""
    e = q["leak"][direction]
    if (e["level_db"] is None or e["correlation"] is None
            or e["correlation"] < LEAK_MIN_CORRELATION or e["level_db"] > LEAK_REVERSED_DB):
        return None
    if e["level_db"] >= LEAK_MAX_LEVEL_DB:
        return "level"
    above = leak_above_threshold(q, direction)
    if above is not None and above >= LEAK_MAX_ABOVE_THRESHOLD_DB:
        return "threshold"
    return None


def flags_of(q: Dict[str, Any]) -> List[str]:
    """Which conditions of one recording are outside the checked range, as
    ``"<kind>:<channel>"`` (``leak:user`` = the system's voice leaks into the
    user's channel). Kinds: ``noise``, ``leak``, ``clipping``, ``low_level``
    and the note ``quiet_channel``."""
    out = []
    for ch in CHANNELS:
        c = q[ch]
        voiced = bool(c["voiced_share"])
        if c["snr_db"] is not None and c["snr_db"] < NOISE_MIN_SNR_DB:
            out.append(f"noise:{ch}")
        if _leaks(q, f"{_other(ch)}_to_{ch}"):
            out.append(f"leak:{ch}")
        if voiced and c["clipped_share"] >= CLIP_MAX_SHARE:
            out.append(f"clipping:{ch}")
        if voiced and c["threshold_set_by"] == "min_level_db":
            out.append(f"low_level:{ch}")
        if (c["digital_silence_share"] is not None and c["digital_silence_share"] < 1.0
                and c["voiced_share"] < QUIET_MAX_VOICED_SHARE):
            out.append(f"quiet_channel:{ch}")
    return out


# ---------------------------------------------------------------------------
# Several recordings
# ---------------------------------------------------------------------------

_POOLED_CHANNEL = ("speech_level_db", "background_level_db", "snr_db", "voiced_share",
                   "digital_silence_share", "clipped_share", "threshold_db")
_POOLED_LEAK = ("level_db", "correlation", "above_threshold_db")


def _spread(values: Sequence[Optional[float]], digits: int) -> Dict[str, Any]:
    v = np.array([x for x in values if x is not None], dtype=float)
    if not len(v):
        return {"n": 0, "median": None, "min": None, "max": None}
    return {"n": int(len(v)), "median": _r(float(np.median(v)), digits),
            "min": _r(float(v.min()), digits), "max": _r(float(v.max()), digits)}


def pool(qualities: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The estimates of several recordings: median, minimum and maximum of each
    quantity over the recordings, the number of recordings with each flag and
    the thresholds the flags were set with."""
    out: Dict[str, Any] = {"n_recordings": len(qualities)}
    for ch in CHANNELS:
        out[ch] = {k: _spread([q[ch][k] for q in qualities], 6 if k == "clipped_share" else 4
                              if k.endswith("share") else 2)
                   for k in _POOLED_CHANNEL}
        by = [q[ch]["threshold_set_by"] for q in qualities]
        out[ch]["threshold_set_by"] = {k: by.count(k) for k in sorted({b for b in by if b})}
    out["leak"] = {d: {k: _spread([q["leak"][d][k] for q in qualities], 3 if k == "correlation"
                                  else 1) for k in _POOLED_LEAK}
                   for d in LEAK_DIRECTIONS}
    counts: Dict[str, int] = {}
    for q in qualities:
        for f in q.get("flags", []):
            counts[f] = counts.get(f, 0) + 1
    out["flags"] = dict(sorted(counts.items()))
    out["thresholds"] = dict(THRESHOLDS)
    return out


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

def _range(values: Sequence[float], fmt: str) -> str:
    lo, hi = fmt.format(min(values)), fmt.format(max(values))
    return lo if lo == hi else f"{lo} to {hi}"


def _message(kind: str, ch: str, qs: Sequence[Dict[str, Any]],
             min_level_db: Optional[float]) -> str:
    """What is wrong on channel ``ch`` of the recordings ``qs`` and which way
    it moves the numbers."""
    other = _other(ch)
    if kind == "noise":
        snr = _range([q[ch]["snr_db"] for q in qs], "{:.0f}")
        return (f"the speech on the {ch} channel is only {snr} dB above the background "
                f"(checked down to 25 dB, warned below {NOISE_MIN_SNR_DB:.0f} dB). Noise "
                "first fills the silences (fewer pauses and turn ends, more overlap); "
                "stronger noise hides weak speech (higher no-response rate, longer latency)")
    if kind == "leak":
        d = f"{other}_to_{ch}"
        e = [q["leak"][d] for q in qs]
        quiet = ""
        if any(_leaks(q, d) == "threshold" for q in qs):
            quiet = (f" The {ch} channel is quiet, so that a leak this weak reaches its "
                     "voice threshold.")
        return (f"the {other}'s voice seems to leak into the {ch} channel, at about "
                f"{_range([x['level_db'] for x in e], '{:.0f}')} dB (correlation "
                f"{_range([x['correlation'] for x in e], '{:.2f}')}; an estimate, two people "
                f"making the same sound at once can look like a leak).{quiet} A leak is "
                f"counted as the {ch}'s speech: more overlap, more starts during user "
                "speech and more reactions in pauses, fewer turn ends")
    if kind == "clipping":
        share = _range([100 * q[ch]["clipped_share"] for q in qs], "{:.2f}")
        return (f"{share} % of the samples of the {ch} channel are at the peak value "
                f"(clipped; warned from {100 * CLIP_MAX_SHARE:.1f} %). Clipping gives "
                "slightly more overlap and slightly fewer pauses and turn ends")
    if kind == "low_level":
        level = _range([q[ch]["speech_level_db"] for q in qs], "{:.0f}")
        limit = "min_level_db" if min_level_db is None else f"min_level_db = {min_level_db:g} dBFS"
        return (f"the {ch} channel is recorded at a low level (speech at {level} dBFS): the "
                f"voice threshold was set by the fixed lower limit ({limit}), not by the "
                "speech. Quiet speech is then cut into pieces or missed (more pauses and "
                "turn ends, less overlap). If the speech is really this quiet, lower the "
                "limit, e.g. --set min_level_db=-84")
    if kind == "quiet_channel":
        share = _range([100 * q[ch]["voiced_share"] for q in qs], "{:.1f}")
        return (f"the {ch} channel has almost no voice ({share} % of the frames). Such a channel "
                f"is sensitive to a leak of the {other}'s voice, because its voice threshold "
                "follows whatever sound is there: with no voice at all a leak of -40 dB was "
                "counted as its speech, with little voice a leak of -30 dB and sometimes a "
                "weaker one. Make sure the channels are separated")
    raise ValueError(kind)


def messages(names: Sequence[str], qualities: Sequence[Dict[str, Any]],
             min_level_db: Optional[float]) -> Tuple[List[str], List[str]]:
    """``(warnings, notes)`` for the recordings: one line per kind of flag and
    channel, with the recordings it was raised on (the first three, by name;
    by position in the order measured when the names do not tell the
    recordings apart, as with ``<clip>/combined.wav``)."""
    warnings: List[str] = []
    notes: List[str] = []
    seen: Dict[str, List[int]] = {}
    for i, q in enumerate(qualities):
        for f in q.get("flags", []):
            seen.setdefault(f, []).append(i)
    n = len(qualities)
    order = [f"{k}:{ch}" for k in ("noise", "leak", "clipping", "low_level", "quiet_channel")
             for ch in CHANNELS]
    for f in order:
        if f not in seen:
            continue
        kind, ch = f.split(":")
        idx = seen[f]
        text = _message(kind, ch, [qualities[i] for i in idx], min_level_db)
        if n == 1:
            line = f"{names[0]}: {text}"
        else:
            if len(set(names)) == len(names):
                shown = ", ".join(names[i] for i in idx[:3])
            else:
                shown = "no. " + ", ".join(str(i + 1) for i in idx[:3])
            shown += ", ..." if len(idx) > 3 else ""
            line = f"{len(idx)} of {n} recordings ({shown}): {text}"
        (notes if kind in NOTE_FLAGS else warnings).append(line)
    return warnings, notes


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

def report_lines(pooled: Dict[str, Any]) -> List[str]:
    """The block of the text report: one row per estimate, one column per
    channel; ``median [min, max]`` over the recordings when there are several."""
    many = pooled["n_recordings"] > 1

    def cell(s: Dict[str, Any], fmt: str, scale: float = 1.0) -> str:
        if s["median"] is None:
            return "-"
        if not many or s["min"] == s["max"]:
            return fmt.format(scale * s["median"])
        return (f"{fmt.format(scale * s['median'])} [{fmt.format(scale * s['min'])}, "
                f"{fmt.format(scale * s['max'])}]")

    u, s, leak = pooled["user"], pooled["system"], pooled["leak"]
    rows = [
        ("speech level (dBFS)", cell(u["speech_level_db"], "{:.1f}"),
         cell(s["speech_level_db"], "{:.1f}")),
        ("speech above background (dB)", cell(u["snr_db"], "{:.1f}"), cell(s["snr_db"], "{:.1f}")),
        ("frames with voice (%)", cell(u["voiced_share"], "{:.1f}", 100),
         cell(s["voiced_share"], "{:.1f}", 100)),
        ("samples at the peak value (%)", cell(u["clipped_share"], "{:.3f}", 100),
         cell(s["clipped_share"], "{:.3f}", 100)),
        ("leak from the other channel (dB)", cell(leak["system_to_user"]["level_db"], "{:.0f}"),
         cell(leak["user_to_system"]["level_db"], "{:.0f}")),
        ("  correlation", cell(leak["system_to_user"]["correlation"], "{:.2f}"),
         cell(leak["user_to_system"]["correlation"], "{:.2f}")),
    ]
    w0 = max(len(r[0]) for r in rows)
    w1 = max(len("user"), *(len(r[1]) for r in rows))
    w2 = max(len("system"), *(len(r[2]) for r in rows))
    head = ("recording quality (estimated from the audio"
            + ("; median [min, max] over the recordings)" if many else ")"))
    return ([head, f"  {'':{w0}}  {'user':>{w1}}  {'system':>{w2}}"]
            + [f"  {a:{w0}}  {b:>{w1}}  {c:>{w2}}" for a, b, c in rows])
