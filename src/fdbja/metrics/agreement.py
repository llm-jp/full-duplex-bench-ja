"""Inter-rater agreement and rank-correlation helpers (numpy only).

Crowdsourcing gives a *variable* number of raters per item, so the helpers
here are chosen to tolerate that:

  - ``pct_agreement`` : mean pairwise agreement, defined for any rater count.
  - ``fleiss_kappa``  : classic Fleiss; requires a fixed rater count, so the
                        caller subsets items to the modal count first.
  - ``icc1``          : one-way random-effects ICC(1); handles unequal group
                        sizes, the right model when raters are nested in items.
  - ``spearman`` / ``bootstrap_ci`` / ``partial_spearman`` for metric validity.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Categorical agreement (Study A)
# ---------------------------------------------------------------------------

def pct_agreement(ratings: Sequence[Sequence[int]]) -> Optional[float]:
    """Mean over items of the pairwise agreement among that item's raters.

    ``ratings`` is a list of per-item label lists (lengths may differ).
    Items with < 2 ratings are skipped. Returns None if none qualify.
    """
    vals = []
    for r in ratings:
        n = len(r)
        if n < 2:
            continue
        agree = 0
        for i in range(n):
            for j in range(i + 1, n):
                agree += int(r[i] == r[j])
        vals.append(agree / (n * (n - 1) / 2))
    return float(np.mean(vals)) if vals else None


def fleiss_kappa(table: np.ndarray) -> Optional[float]:
    """Fleiss' kappa from an items x categories count matrix.

    Every item must have the same total count (rater count). Returns None if
    that does not hold or there are < 2 items.
    """
    table = np.asarray(table, dtype=float)
    if table.shape[0] < 2:
        return None
    n_per = table.sum(axis=1)
    if not np.allclose(n_per, n_per[0]) or n_per[0] < 2:
        return None
    n = n_per[0]
    N = table.shape[0]
    p_j = table.sum(axis=0) / (N * n)
    P_i = (np.square(table).sum(axis=1) - n) / (n * (n - 1))
    P_bar = P_i.mean()
    P_e = float(np.square(p_j).sum())
    if P_e >= 1.0:
        return None
    return float((P_bar - P_e) / (1 - P_e))


def to_count_table(ratings: Sequence[Sequence[int]], n_cat: int
                   ) -> Tuple[np.ndarray, List[int]]:
    """Build the items x categories count matrix for items sharing the modal
    rater count. Returns (table, kept_item_indices)."""
    counts = [len(r) for r in ratings if r]
    if not counts:
        return np.zeros((0, n_cat)), []
    modal = int(np.bincount(counts).argmax())
    table, kept = [], []
    for k, r in enumerate(ratings):
        if len(r) != modal:
            continue
        row = np.zeros(n_cat)
        for v in r:
            if 0 <= v < n_cat:
                row[v] += 1
        table.append(row)
        kept.append(k)
    return np.asarray(table), kept


# ---------------------------------------------------------------------------
# Continuous agreement (Study B Likert)
# ---------------------------------------------------------------------------

def icc1(groups: Sequence[Sequence[float]]) -> Optional[float]:
    """One-way random-effects ICC(1): reliability of a single rating.

    ``groups`` is a list of per-item rating lists (unequal lengths allowed).
    Uses the unbalanced one-way ANOVA. Returns None if < 2 items have ratings.
    """
    groups = [np.asarray(g, dtype=float) for g in groups if len(g)]
    k = len(groups)
    if k < 2:
        return None
    n_i = np.array([len(g) for g in groups], dtype=float)
    N = n_i.sum()
    grand = np.concatenate(groups).mean()
    means = np.array([g.mean() for g in groups])
    ms_between = (n_i * (means - grand) ** 2).sum() / (k - 1)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    dfw = N - k
    if dfw <= 0:
        return None
    ms_within = ssw / dfw
    # n0: mean group size correction for unbalanced design
    n0 = (N - (n_i ** 2).sum() / N) / (k - 1)
    if n0 == 0 or (ms_between + (n0 - 1) * ms_within) == 0:
        return None
    return float((ms_between - ms_within) /
                 (ms_between + (n0 - 1) * ms_within))


# ---------------------------------------------------------------------------
# Rank correlation (metric validity)
# ---------------------------------------------------------------------------

def _rankdata(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    order = a.argsort()
    ranks = np.empty(len(a))
    ranks[order] = np.arange(1, len(a) + 1)
    sv = a[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return None
    return _pearson(_rankdata(x), _rankdata(y))


def bootstrap_ci(x: Sequence[float], y: Sequence[float], stat=spearman,
                 n_boot: int = 2000, seed: int = 0, ci: float = 0.95
                 ) -> Optional[Tuple[float, float]]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 4:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        s = stat(x[idx], y[idx])
        if s is not None:
            vals.append(s)
    if not vals:
        return None
    lo = float(np.percentile(vals, (1 - ci) / 2 * 100))
    hi = float(np.percentile(vals, (1 + ci) / 2 * 100))
    return lo, hi


def partial_spearman(x: Sequence[float], y: Sequence[float],
                     covars: Sequence[Sequence[float]]) -> Optional[float]:
    """Spearman partial correlation of x, y controlling for ``covars``
    (a list of covariate vectors). Computed on ranks via residualization."""
    x = _rankdata(np.asarray(x, dtype=float))
    y = _rankdata(np.asarray(y, dtype=float))
    cs = [c for c in covars if c is not None]
    if len(x) < 4:
        return None
    if not cs:
        return _pearson(x, y)
    Z = np.column_stack([_rankdata(np.asarray(c, dtype=float)) for c in cs])
    Z = np.column_stack([np.ones(len(x)), Z])

    def resid(v):
        beta, *_ = np.linalg.lstsq(Z, v, rcond=None)
        return v - Z @ beta

    return _pearson(resid(x), resid(y))
