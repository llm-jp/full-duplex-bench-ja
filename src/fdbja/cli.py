"""Command-line entry point for Full-Duplex-Bench-JA.

Config-driven evaluation. Pipeline:

    reconstruct  : LLM-jp-Zoom1 or your own two-channel dialogues (--layout
                   generic) -> detected events + 20s eval clips + manifest
    infer        : stream each clip's speaker-A audio to a model adapter,
                   record the response
    transcribe   : ASR the recorded responses into character-level jchat
                   (backend chosen by ``asr.backend``)
    score-offline: score precomputed model outputs (binary TOR; VAP-distance
                   when enabled) -> results.json, summary.json, summary.csv
    leaderboard  : rank a summary.json like the leaderboard page, optionally
                   next to the published rows (--reference)
    analyze-profile: per-system turn-taking profile + event-level report

Outside the benchmark (no config, no transcript, numpy + soundfile only):

    measure      : timing statistics of any two-channel recording (user on
                   one channel, system on the other); with --profile, scores
                   against a reference profile
    make-profile : build a reference profile from two-channel human-human
                   recordings (language / relationship / situation / medium)

``reconstruct`` needs the corpus and ``infer`` a model adapter (server or API
key); ``examples/dummy/`` runs every stage on a synthetic corpus instead.

Usage:
    fdbja --help
    fdbja transcribe --config configs/default.yaml [--model NAME] [--task T]
    fdbja score-offline --config configs/default.yaml [--limit N]
"""

from __future__ import annotations

import argparse
import sys

from fdbja.config import Config


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="fdbja",
                                     description="Full-Duplex-Bench-JA toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("reconstruct",
                       help="build eval clips + manifest from LLM-jp-Zoom1 "
                            "or your own two-channel dialogues")
    p.add_argument("--config", required=True)
    p.add_argument("--limit-sessions", type=int, default=0,
                   help="scan only the first N corpus sessions (debug)")
    p.add_argument("--sessions", default=None, metavar="FILE",
                   help="text file with one recording id per line; only these "
                        "sessions are scanned (e.g. a model's held-out split, "
                        "to build a domain: out evaluation set)")
    p.add_argument("--clips", default=None, metavar="CSV",
                   help="rebuild a published evaluation set from its clip list "
                        "(columns task, clip_id, recording_key, clip_start_s) "
                        "instead of sampling")
    p.add_argument("--layout", choices=["zoom1", "generic"], default=None,
                   help="corpus layout (default: corpus.layout, else zoom1). "
                        "generic = every *.json under corpus.text_dir paired "
                        "with the audio of the same file stem under "
                        "corpus.audio_dir (stereo, ch0 = A, ch1 = B)")

    p = sub.add_parser("infer", help="run a model adapter over the eval clips")
    p.add_argument("--config", required=True)
    p.add_argument("--limit", type=int, default=0,
                   help="generate only the first N clips per task (debug)")
    p.add_argument("--model", nargs="+", default=None,
                   help="only these model names from the config")

    p = sub.add_parser("transcribe",
                       help="ASR the model responses (infer output) into jchat "
                            "transcripts for TOR")
    p.add_argument("--config", default=None,
                   help="evaluation config (required unless --wav is given)")
    p.add_argument("--model", nargs="+", default=None,
                   help="only these model names from the config")
    p.add_argument("--task", nargs="+", default=None,
                   help="only these tasks (default: config tasks)")
    p.add_argument("--backend", default=None,
                   help="ASR backend name (default: asr.backend in the config)")
    p.add_argument("--device", default=None, help="e.g. cuda, cuda:1, cpu")
    p.add_argument("--limit", type=int, default=0,
                   help="transcribe only the first N clips per task (debug)")
    p.add_argument("--overwrite", action="store_true",
                   help="re-transcribe clips that already have a jchat file")
    p.add_argument("--wav", nargs="+", default=None,
                   help="ad-hoc mode: transcribe these audio files instead of "
                        "the config layout (writes <out>/<stem>.json)")
    p.add_argument("--out", default=None, help="output dir for --wav mode")
    p.add_argument("--channel", type=int, nargs="+", default=[1],
                   help="channel(s) of stereo input in --wav mode "
                        "(default 1; combined.wav has the model on ch1). "
                        "Give two with --speaker A B to transcribe a whole "
                        "two-channel dialogue into one jchat")
    p.add_argument("--speaker", nargs="+", default=None, choices=["A", "B"],
                   help="speaker label per --channel in --wav mode "
                        "(default B). An evaluation-set transcript needs "
                        "ch0 = A and ch1 = B: --channel 0 1 --speaker A B")

    p = sub.add_parser("score-offline",
                       help="score precomputed model outputs (TOR; VAP-distance when enabled)")
    p.add_argument("--config", required=True)
    p.add_argument("--limit", type=int, default=0,
                   help="score only the first N clips per task (debug)")
    p.add_argument("--model", nargs="+", default=None,
                   help="only these model names from the config")

    p = sub.add_parser("leaderboard",
                       help="rank the systems of a summary.json like the leaderboard page")
    p.add_argument("--results", default="results/summary.json",
                   help="summary.json or results.json written by score-offline")
    p.add_argument("--reference", default=None,
                   help="published results to merge in, e.g. docs/results.json "
                        "(differing scoring conditions are reported)")

    p = sub.add_parser("sample-human-eval",
                       help="build crowdsourcing manifests for human validation")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True, help="output dir for manifests/stimuli")
    p.add_argument("--n-a", type=int, default=60, help="Study A events per task")
    p.add_argument("--n-b", type=int, default=50, help="Study B events per task")
    p.add_argument("--per-worker", type=int, default=30,
                   help="max items shown to one worker (recruit enough workers "
                        "so each item gets the desired number of ratings)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-audio", action="store_true",
                   help="skip rendering Study-A stimulus audio")

    p = sub.add_parser("analyze-human-eval",
                       help="aggregate responses_*.json and validate metrics")
    p.add_argument("--responses", required=True,
                   help="dir containing responses_*.json from the web UI")
    p.add_argument("--bundle", required=True,
                   help="human-eval bundle dir (has study_a.csv/study_b.csv)")
    p.add_argument("--results", default=None,
                   help="score-offline results.json to join VAP-distance")
    p.add_argument("--out", required=True, help="output dir for the analysis")

    p = sub.add_parser(
        "analyze-pause",
        help="stratify existing Pause results by intra-turn gap duration")
    p.add_argument("--eval-dir", required=True,
                   help="evaluation directory containing pause/ and caches")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--models", nargs="+", default=None,
                   help="model directory suffixes (default: discover from ASR)")
    p.add_argument("--bootstrap", type=int, default=10000,
                   help="clip-level bootstrap replicates")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--vap-checkpoint", default=None,
                   help="VAP checkpoint file name whose score-offline cache "
                        "(vap_cache/<stem>/) to read (default: the only one present)")

    p = sub.add_parser(
        "analyze-profile",
        help="turn-taking profile per system (talks over / stays silent / slow) "
             "and an event-level report, from a score-offline results.json")
    p.add_argument("--results", required=True, help="results.json of score-offline")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--model", nargs="+", default=None, help="only these systems")
    p.add_argument("--takeover-hi", type=float, default=0.5,
                   help="'overlapping' when Pause talked-through share >= this")
    p.add_argument("--no-response-hi", type=float, default=0.5,
                   help="'silent' when Smooth TT no-response share >= this")
    p.add_argument("--slow-onset", type=float, default=1.0,
                   help="'slow' when the median Smooth TT onset (s) >= this")

    p = sub.add_parser(
        "measure",
        help="timing statistics of any two-channel recording from the audio alone "
             "(response latency, pauses, overlap); not comparable with the leaderboard")
    p.add_argument("audio", help="two-channel wav, a directory of them, or the "
                                 "user's mono wav together with --system")
    p.add_argument("--system", default=None, help="the system's mono wav (two mono inputs)")
    p.add_argument("--user-channel", type=int, default=0,
                   help="channel of the user (default 0; the other is the system)")
    p.add_argument("--system-channel", type=int, default=None,
                   help="channel of the system (needed with more than two channels)")
    p.add_argument("--profile", default=None,
                   help="'generic' (rule constants, not a human norm) or a profile "
                        "JSON written by make-profile; adds scores")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="detection setting, e.g. turn_min_seconds=4 or vad.margin_db=12 "
                        "(repeatable; default: the profile's settings)")
    p.add_argument("--condition", action="append", default=[], metavar="KEY=VALUE",
                   help="what was recorded (model, voice, network, ...), copied to "
                        "the output (repeatable)")
    p.add_argument("--condition-file", action="append", default=[], metavar="KEY=PATH",
                   help="record the SHA-256 prefix of a file (e.g. a prompt) as a "
                        "condition without its content (repeatable)")
    p.add_argument("--json", default=None, help="also write the full result (with events)")

    p = sub.add_parser(
        "make-profile",
        help="build a reference profile for `measure --profile` from two-channel "
             "human-human recordings")
    p.add_argument("recordings", nargs="+", help="two-channel wav files or directories")
    p.add_argument("--language", required=True, help="e.g. ja")
    p.add_argument("--relationship", required=True, help="e.g. strangers, friends")
    p.add_argument("--situation", required=True, help="e.g. chat, customer support")
    p.add_argument("--medium", required=True, help="e.g. video call, telephone, face to face")
    p.add_argument("--source", default=None, help="what the recordings are")
    p.add_argument("--name", default=None, help="profile name (default: from the four fields)")
    p.add_argument("--tolerance", type=float, nargs=2, default=[5.0, 95.0],
                   metavar=("LOW", "HIGH"),
                   help="percentiles of the human response latency that bound the "
                        "tolerance range (default 5 95)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="detection setting (repeatable)")
    p.add_argument("-o", "--out", required=True, help="output profile JSON")

    args = parser.parse_args(argv)

    if args.command == "measure":
        return _cmd_measure(args)

    if args.command == "make-profile":
        from fdbja import measurement as ms
        profile = ms.make_profile(
            args.recordings, language=args.language, relationship=args.relationship,
            situation=args.situation, medium=args.medium, source=args.source,
            name=args.name, config=ms.parse_settings(args.set),
            tolerance_percentiles=tuple(args.tolerance))
        path = ms.write_profile(profile, args.out)
        tol, meta = profile["norms"]["response_latency"], profile["meta"]
        print(f"[fdbja] wrote {path}: {ms.profile_id(profile)}, "
              f"{meta['n_recordings']} recordings, {meta['n_events']['responses']} responses, "
              f"latency tolerance {tol['low']:+.2f} to {tol['high']:+.2f} s", file=sys.stderr)
        for w in meta.get("warnings", []):
            print(f"[fdbja] warning: {w}", file=sys.stderr)
        return 0

    if args.command == "reconstruct":
        from fdbja.data.reconstruct import reconstruct, reconstruct_from_list
        if args.clips:
            reconstruct_from_list(Config.load(args.config), args.clips,
                                  layout=args.layout)
            return 0
        reconstruct(Config.load(args.config),
                    limit_sessions=args.limit_sessions,
                    sessions_file=args.sessions, layout=args.layout)
        return 0

    if args.command == "infer":
        from fdbja.runner import run_infer
        failed = run_infer(Config.load(args.config), limit=args.limit,
                           models=args.model)
        return 1 if failed else 0

    if args.command == "transcribe":
        return _cmd_transcribe(args)

    if args.command == "score-offline":
        from fdbja import report
        from fdbja.runner import score_offline
        config = Config.load(args.config)
        from fdbja.summary import write_summary
        results = score_offline(config, limit=args.limit, models=args.model)
        path = report.write_json(results, config.output_dir)
        extra = write_summary(results, config.output_dir)
        print(report.format_table(results))
        print(f"\n[fdbja] wrote {path}, " + ", ".join(extra), file=sys.stderr)
        return 0

    if args.command == "leaderboard":
        import json

        from fdbja import report
        with open(args.results, encoding="utf-8") as f:
            results = json.load(f)
        reference = None
        if args.reference:
            with open(args.reference, encoding="utf-8") as f:
                reference = json.load(f)
        print(report.format_leaderboard(results, reference))
        return 0

    if args.command == "sample-human-eval":
        from fdbja.human_eval import sample_human_eval
        sample_human_eval(Config.load(args.config), out_dir=args.out,
                          n_a=args.n_a, n_b=args.n_b, seed=args.seed,
                          per_worker=args.per_worker,
                          render_audio=not args.no_audio)
        return 0

    if args.command == "analyze-human-eval":
        from fdbja import analyze_human_eval
        analyze_human_eval.run(args.responses, args.bundle, args.out,
                               results_path=args.results)
        return 0

    if args.command == "analyze-profile":
        from fdbja.turn_profile import run as run_profile
        run_profile(args.results, args.out, models=args.model,
                    takeover_hi=args.takeover_hi, no_response_hi=args.no_response_hi,
                    slow_onset=args.slow_onset)
        return 0

    if args.command == "analyze-pause":
        from fdbja.pause_analysis import run
        run(args.eval_dir, args.out, models=args.models,
            n_boot=args.bootstrap, seed=args.seed,
            vap_checkpoint=args.vap_checkpoint)
        return 0
    return 1


def _cmd_measure(args) -> int:
    import json

    from fdbja import measurement as ms

    conditions = {}
    for item in args.condition:
        if "=" not in item:
            print(f"fdbja measure: --condition expects KEY=VALUE, got {item!r}", file=sys.stderr)
            return 2
        key, value = item.split("=", 1)
        conditions[key] = value
    for item in args.condition_file:
        if "=" not in item:
            print(f"fdbja measure: --condition-file expects KEY=PATH, got {item!r}",
                  file=sys.stderr)
            return 2
        key, path = item.split("=", 1)
        with open(path, "rb") as f:
            conditions[key] = ms.text_hash(f.read())
    result = ms.measure(args.audio, args.system, user_channel=args.user_channel,
                        system_channel=args.system_channel, profile=args.profile,
                        config=ms.parse_settings(args.set) if args.set else None,
                        conditions=conditions)
    print(ms.format_report(result))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=1)
            f.write("\n")
        print(f"[fdbja] wrote {args.json}", file=sys.stderr)
    return 0


def _cmd_transcribe(args) -> int:
    import os

    from fdbja.runner import run_transcribe, transcribe_channels

    if args.wav:
        from fdbja.asr import DEFAULT_BACKEND, load_asr
        config = Config.load(args.config) if args.config else None
        asr_cfg = (config.get("asr", default={}) if config else {}) or {}
        name = args.backend or asr_cfg.get("backend") or DEFAULT_BACKEND
        options = dict(asr_cfg.get("options", {}) or {})
        if args.device:
            options["device"] = args.device
        asr = load_asr(name, **options)
        speakers = args.speaker or ["B"] * len(args.channel)
        if len(speakers) != len(args.channel):
            print("fdbja transcribe: give one --speaker per --channel", file=sys.stderr)
            return 2
        gate = asr_cfg.get("silence_gate") or None
        out_dir = args.out or "."
        for wav in args.wav:
            stem = os.path.splitext(os.path.basename(wav))[0]
            out = os.path.join(out_dir, stem + ".json")
            items = transcribe_channels(asr, wav, out, args.channel, speakers, gate=gate)
            print(f"[fdbja] {wav} -> {out} ({len(items)} chars)", file=sys.stderr)
        return 0

    if not args.config:
        print("fdbja transcribe: --config is required (or use --wav)", file=sys.stderr)
        return 2
    run_transcribe(Config.load(args.config), limit=args.limit, models=args.model,
                   tasks=args.task, backend=args.backend, overwrite=args.overwrite,
                   device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
