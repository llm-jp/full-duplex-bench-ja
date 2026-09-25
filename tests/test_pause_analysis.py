import unittest

import numpy as np

from fdbja.pause_analysis import aggregate, gap_bin


class PauseAnalysisTest(unittest.TestCase):
    def test_gap_bin_boundaries(self):
        self.assertEqual(gap_bin(0.4), "0.4-0.6")
        self.assertEqual(gap_bin(0.5999), "0.4-0.6")
        self.assertEqual(gap_bin(0.6), "0.6-0.8")
        self.assertEqual(gap_bin(0.8), "0.8-1.0")
        self.assertEqual(gap_bin(1.0), "0.8-1.0")
        self.assertIsNone(gap_bin(0.39))

    def test_aggregate_reports_each_bin_and_macro(self):
        rows = []
        for index, label in enumerate(("0.4-0.6", "0.6-0.8", "0.8-1.0")):
            rows.extend([
                {"model": "m", "clip": f"c{index}a", "gap_bin": label,
                 "TO": 0, "mean_dkl": 0.1 + index},
                {"model": "m", "clip": f"c{index}b", "gap_bin": label,
                 "TO": 1, "mean_dkl": 0.3 + index},
            ])
        summary = aggregate(rows, n_boot=100, seed=1)
        self.assertEqual([r["gap_bin"] for r in summary], [
            "0.4-0.6", "0.6-0.8", "0.8-1.0", "macro"])
        self.assertTrue(all(np.isclose(r["TOR"], 0.5) for r in summary))
