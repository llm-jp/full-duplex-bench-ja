#!/usr/bin/env python3
"""Decode & verify FDBJA1 submission tokens pasted by crowd workers.

Usage:
    python webui/decode_submissions.py <pasted.txt> <out_dir>

<pasted.txt> is any text file containing one or more ``FDBJA1.<b64>.<code>``
tokens (e.g. answer fields exported from a crowdsourcing platform).  Each token's completion
code is recomputed from the payload (FNV-1a over the base64 string, matching
webui/app.js); tokens whose code does not match are reported as TAMPERED and
skipped. The code is an unkeyed checksum: it catches accidental edits and
copy errors, not deliberate forgery (the algorithm is public).  Valid payloads are written to ``<out_dir>/responses_<worker>.json``
ready for ``fdbja analyze-human-eval``.
"""
import base64
import json
import os
import re
import sys


def fnv1a(s: str) -> int:
    h = 2166136261
    for ch in s:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def main(src: str, out_dir: str) -> None:
    text = open(src, encoding="utf-8").read()
    tokens = re.findall(r"FDBJA1\.([A-Za-z0-9+/=]+)\.([A-Z0-9]+)", text)
    os.makedirs(out_dir, exist_ok=True)
    ok = bad = 0
    for b64, code in tokens:
        if base36(fnv1a(b64)) != code:
            print(f"TAMPERED (code mismatch): ...{b64[-16:]} code={code}")
            bad += 1
            continue
        payload = json.loads(base64.b64decode(b64).decode("utf-8"))
        payload["completion_code"] = code
        wid = re.sub(r"[^A-Za-z0-9_-]", "_", payload.get("worker_id", "anon"))
        dst = os.path.join(out_dir, f"responses_{wid}.json")
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        rt = [r.get("response_time_ms", 0) for r in payload.get("responses", [])]
        flag = " [FAST?]" if rt and sorted(rt)[len(rt) // 2] < 5000 else ""
        print(f"OK {wid}: {payload.get('n_items')} items, "
              f"median {sorted(rt)[len(rt)//2]/1000:.1f}s/item{flag} -> {dst}")
        ok += 1
    print(f"\n{ok} valid, {bad} tampered/invalid")


def base36(n: int) -> str:
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    out = ""
    while n:
        n, r = divmod(n, 36)
        out = digits[r] + out
    return out or "0"


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
