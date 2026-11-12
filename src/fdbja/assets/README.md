# Attribution: the `always_talk` speech loop

The `always_talk` trivial baseline (`fdbja.adapters.baselines.AlwaysTalkAdapter`, default `source: auto`)
loops 60 s of synthetic Japanese speech. **No audio is stored in this repository**: the loop is synthesised on
first use by `fdbja.adapters.always_talk_loop` and cached as `always_talk_loop_ja.flac` under
`$FDBJA_CACHE_DIR` (default `~/.cache/fdbja`). To write it explicitly:
`uv run --with pyopenjtalk python examples/always_talk_loop/make_loop.py [out.flac]`.

- Text: nine sentences written for this toolkit (no third-party text), 80 ms gaps, 16 kHz mono, RMS 0.035.
- Synthesiser: Open JTalk via [`pyopenjtalk`](https://github.com/r9y9/pyopenjtalk) (Modified BSD).
- Voice: HTS voice **"Mei"** (`mei_normal.htsvoice`, shipped with pyopenjtalk), © 2009-2013 Nagoya Institute
  of Technology, Department of Computer Science, released by the MMDAgent Project Team
  (http://www.mmdagent.jp/) under the **Creative Commons Attribution 3.0** license
  (https://creativecommons.org/licenses/by/3.0/). The generated loop is a derivative of that voice; if you
  redistribute it, keep this attribution and the CC BY 3.0 terms.

This loop is not the one behind the `always_talk` row of the leaderboard (the source of that loop could not be
confirmed, so it is not redistributed). On 20 pause / 10 smooth_tt clips with `reazonspeech_whisperx` this loop
gave Pause TOR 1.000 / Smooth TT TOR 0.000, and the leaderboard loop 0.960 / 0.000 on the same clips. On the full
evaluation set the leaderboard row is 0.938 / 0.005 with audio boundaries (0.992 / 0.005 with transcript
boundaries; `docs/results.json`).
