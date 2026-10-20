"""Write the always_talk speech loop (Open JTalk, voice "Mei", CC BY 3.0) to a file.

    uv run --with pyopenjtalk python examples/always_talk_loop/make_loop.py [out.flac]

The adapter does this itself on first use (cached under ``$FDBJA_CACHE_DIR`` or ``~/.cache/fdbja``);
this script is for inspecting the audio or placing it somewhere explicit (``source_wav``).
"""
import sys

from fdbja.adapters.always_talk_loop import cache_path, ensure_loop

print(ensure_loop(sys.argv[1] if len(sys.argv) > 1 else cache_path()))
