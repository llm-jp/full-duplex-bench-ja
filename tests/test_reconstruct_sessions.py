import json
import os

from fdbja.data.reconstruct import collect_events, load_session_list


def _write(tmp_path, sid, key):
    d = tmp_path / sid
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{key}.json").write_text(json.dumps([]), encoding="utf-8")


def test_load_session_list_ignores_comments_and_blanks(tmp_path):
    p = tmp_path / "ids.txt"
    p.write_text("# held-out\n0001_W01_W02\n\n0002_W03_W04  # note\n", encoding="utf-8")
    assert load_session_list(str(p)) == {"0001_W01_W02", "0002_W03_W04"}


def test_collect_events_session_filter(tmp_path):
    text_dir = tmp_path / "text"
    _write(text_dir, "s1", "0001_W01_W02")
    _write(text_dir, "s2", "0002_W03_W04")
    _write(text_dir, "s3", "0003_W05_W06")
    logs = []
    # match by session dir name or by file stem
    collect_events(str(text_dir), ["pause"], log=logs.append,
                   sessions={"s1", "0003_W05_W06"})
    assert any("2/3 sessions kept" in m for m in logs)
    # no filter scans everything
    logs.clear()
    collect_events(str(text_dir), ["pause"], log=logs.append)
    assert any("scanning 3 sessions" in m for m in logs)
    assert os.path.isdir(text_dir)


def test_reconstruct_from_clip_list_reproduces_clips(tmp_path):
    """``reconstruct --clips`` rebuilds the same clips from the published list
    (clip names carry the start rounded to 0.1 s)."""
    import csv
    import importlib.util
    import os

    import numpy as np
    import soundfile as sf
    import yaml

    from fdbja.config import Config
    from fdbja.data.reconstruct import reconstruct, reconstruct_from_list

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dummy = os.path.join(root, "examples", "dummy")
    spec = importlib.util.spec_from_file_location(
        "_dummy_corpus", os.path.join(dummy, "make_dummy_corpus.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.make_dummy_corpus(str(tmp_path / "zoom1"), n_sessions=4, seed=0)
    with open(os.path.join(dummy, "config.yaml"), encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw["corpus"] = {"zoom1_root": str(tmp_path / "zoom1"), "eval_dir": str(tmp_path / "a")}
    raw["detection"]["clips_per_task"] = 3
    quiet = lambda *a, **k: None  # noqa: E731
    reconstruct(Config(raw), log=quiet)

    rows = []
    for task in raw["tasks"]:
        with open(tmp_path / "a" / task / "manifest.jsonl", encoding="utf-8") as f:
            for line in f:
                m = json.loads(line)
                rows.append({"task": task, "clip_id": m["clip"], "recording_key": m["key"],
                             "clip_start_s": m["clip"].split("_")[-2]})
    lst = tmp_path / "clips.csv"
    with open(lst, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    raw["corpus"]["eval_dir"] = str(tmp_path / "b")
    done = reconstruct_from_list(Config(raw), str(lst), log=quiet)
    assert sum(done.values()) == len(rows)
    for r in rows:
        a = tmp_path / "a" / r["task"] / r["clip_id"]
        b = tmp_path / "b" / r["task"] / r["clip_id"]
        assert json.loads(open(f"{a}.json", encoding="utf-8").read()) == \
            json.loads(open(f"{b}.json", encoding="utf-8").read())
        assert np.array_equal(sf.read(f"{a}.wav")[0], sf.read(f"{b}.wav")[0])
