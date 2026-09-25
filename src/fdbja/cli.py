"""Command-line entry point for Full-Duplex-Bench-JA.

Config-driven evaluation. Pipeline:

    reconstruct  : LLM-jp-Zoom1 -> detected events + 20s eval clips + manifest
    infer        : stream each clip's speaker-A audio to a model adapter,
                   record the response
    transcribe   : ASR the recorded responses into character-level jchat
                   (backend chosen by ``asr.backend``)
    score-offline: score precomputed model outputs (binary TOR + VAP-distance)

``score-offline`` is fully wired; ``infer`` / ``reconstruct`` depend on live
model servers / the corpus and are partially stubbed.

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
                       help="build eval clips + manifest from LLM-jp-Zoom1")
    p.add_argument("--config", required=True)
    p.add_argument("--limit-sessions", type=int, default=0,
                   help="scan only the first N corpus sessions (debug)")

    p = sub.add_parser("infer", help="run a model adapter over the eval clips")
    p.add_argument("--config", required=True)
    p.add_argument("--limit", type=int, default=0,
                   help="generate only the first N clips per task (debug)")

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
    p.add_argument("--channel", type=int, default=1,
                   help="response channel of stereo input in --wav mode "
                        "(default 1; combined.wav has the model on ch1)")

    p = sub.add_parser("score-offline",
                       help="score precomputed model outputs (TOR + VAP-distance)")
    p.add_argument("--config", required=True)
    p.add_argument("--limit", type=int, default=0,
                   help="score only the first N clips per task (debug)")

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

    args = parser.parse_args(argv)

    if args.command == "reconstruct":
        from fdbja.data.reconstruct import reconstruct
        reconstruct(Config.load(args.config),
                    limit_sessions=args.limit_sessions)
        return 0

    if args.command == "infer":
        from fdbja.runner import run_infer
        run_infer(Config.load(args.config), limit=args.limit)
        return 0

    if args.command == "transcribe":
        return _cmd_transcribe(args)

    if args.command == "score-offline":
        from fdbja import report
        from fdbja.runner import score_offline
        config = Config.load(args.config)
        results = score_offline(config, limit=args.limit)
        path = report.write_json(results, config.output_dir)
        print(report.format_table(results))
        print(f"\n[fdbja] wrote {path}", file=sys.stderr)
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

    if args.command == "analyze-pause":
        from fdbja.pause_analysis import run
        run(args.eval_dir, args.out, models=args.models,
            n_boot=args.bootstrap, seed=args.seed)
        return 0
    return 1


def _cmd_transcribe(args) -> int:
    import os

    from fdbja.runner import run_transcribe, transcribe_file

    if args.wav:
        from fdbja.asr import DEFAULT_BACKEND, load_asr
        config = Config.load(args.config) if args.config else None
        asr_cfg = (config.get("asr", default={}) if config else {}) or {}
        name = args.backend or asr_cfg.get("backend") or DEFAULT_BACKEND
        options = dict(asr_cfg.get("options", {}) or {})
        if args.device:
            options["device"] = args.device
        asr = load_asr(name, **options)
        out_dir = args.out or "."
        for wav in args.wav:
            stem = os.path.splitext(os.path.basename(wav))[0]
            out = os.path.join(out_dir, stem + ".json")
            items = transcribe_file(asr, wav, out, channel=args.channel)
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
