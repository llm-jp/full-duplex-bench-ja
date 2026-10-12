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
