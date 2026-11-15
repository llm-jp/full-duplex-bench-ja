"""What has to hold for the package to be uploadable to PyPI."""

import re
import unittest
from pathlib import Path

import fdbja

PYPROJECT = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")


def _table(name: str) -> str:
    """The text of one table of pyproject.toml (up to the next table header)."""
    m = re.search(rf"^\[{re.escape(name)}\]\n(.*?)(?=^\[)", PYPROJECT, flags=re.M | re.S)
    assert m, name
    return m.group(1)


class TestPackaging(unittest.TestCase):
    def test_one_version(self):
        m = re.search(r'^version = "([^"]+)"', _table("project"), flags=re.M)
        self.assertEqual(m.group(1), fdbja.__version__)

    def test_no_direct_url_in_the_published_metadata(self):
        # PyPI rejects `name @ url` under [project]; such dependencies live in
        # [dependency-groups], which are not part of the metadata
        for name in ("project", "project.optional-dependencies"):
            self.assertNotIn(" @ ", _table(name), name)
            self.assertNotIn("git+", _table(name), name)
        self.assertIn("git+", _table("dependency-groups"))


if __name__ == "__main__":
    unittest.main()
