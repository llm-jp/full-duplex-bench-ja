"""Discrimination measures for how well a continuous score separates
desired from undesired model behavior."""

from __future__ import annotations

import numpy as np


def auc(binary, cont) -> float | None:
    """AUC of ``cont`` separating positives (binary==1) from negatives.

    Mann-Whitney U with average ranks for ties. Returns None if a class is
    empty. 0.5 = chance.
    """
    b = np.asarray(binary)
    c = np.asarray(cont, dtype=float)
    pos, neg = c[b == 1], c[b == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    u = ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def point_biserial(binary, cont) -> float | None:
    """Pearson r between a binary and a continuous variable."""
    b = np.asarray(binary, dtype=float)
    c = np.asarray(cont, dtype=float)
    if len(b) < 3 or len(set(b.tolist())) < 2:
        return None
    return float(np.corrcoef(b, c)[0, 1])
