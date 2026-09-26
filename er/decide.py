"""
Turn pair probabilities into the final match lists.

1. One-owner constraint: every S2/S3 record keeps only its best-scoring S1.
2. Per-S1 subset choice, either
   a) global threshold t (tuned for macro F0.5 on validation), or
   b) expected-F0.5 maximisation (plug-in approximation, see docs).
"""
import numpy as np
import pandas as pd


def exclusive(d: pd.DataFrame) -> pd.DataFrame:
    """Keep, for each candidate i2, only the row with max p (ties -> first)."""
    idx = d.groupby('i2').p.idxmax()
    return d.loc[idx]


def by_threshold(d: pd.DataFrame, t: float):
    return d[d.p >= t]


def by_expected_f(d: pd.DataFrame, beta=0.5, floor=0.05):
    """For each S1 choose k maximising E[F_beta] ~ (1+b2) sum_{j<=k} p_j / (b2*sum_all p + k),
    compared with the empty set whose expected score is P(no true match) = prod(1-p_j)."""
    b2 = beta * beta
    d = d[d.p >= floor].sort_values(['i1', 'p'], ascending=[True, False])
    keep = []
    for i1, g in d.groupby('i1', sort=False):
        p = g.p.values
        csum = np.cumsum(p)
        k = np.arange(1, len(p) + 1)
        ef = (1 + b2) * csum / (b2 * p.sum() + k)
        best = ef.argmax()
        if ef[best] > np.prod(1 - p):
            keep.append(g.index.values[:best + 1])
    return d.loc[np.concatenate(keep)] if keep else d.iloc[:0]


def to_sets(d: pd.DataFrame, ids1, ids2):
    out = {}
    for i1, g in d.groupby('i1'):
        out[ids1[i1]] = set(ids2[g.i2.values])
    return out
