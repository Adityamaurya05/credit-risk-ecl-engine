"""Weight-of-Evidence binning with monotonicity, as used in bank scorecards.

WoE_i = ln(%good_i / %bad_i). Positive WoE = safer than average.
IV    = sum((%good_i - %bad_i) * WoE_i). Rule of thumb: <0.02 useless,
0.02-0.1 weak, 0.1-0.3 medium, >0.3 strong.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config

EPS = 0.5  # Laplace smoothing so empty cells do not give infinite WoE


@dataclass
class FeatureBinning:
    name: str
    edges: list            # interior cut points; bins are (-inf, e0], (e0, e1], ..., (ek, inf)
    table: pd.DataFrame = field(repr=False)

    @property
    def iv(self) -> float:
        return float(self.table["iv"].sum())

    def bin_index(self, x) -> np.ndarray:
        return np.searchsorted(np.asarray(self.edges), np.asarray(x, float), side="left")

    def transform(self, x) -> np.ndarray:
        return self.table["woe"].to_numpy()[self.bin_index(x)]


def _stats(idx, y, n_bins):
    bad = np.bincount(idx, weights=y, minlength=n_bins)
    total = np.bincount(idx, minlength=n_bins)
    return total, bad, total - bad


def _woe_table(idx, y, edges):
    n_bins = len(edges) + 1
    total, bad, good = _stats(idx, y, n_bins)
    pct_good = (good + EPS) / (good.sum() + EPS * n_bins)
    pct_bad = (bad + EPS) / (bad.sum() + EPS * n_bins)
    woe = np.log(pct_good / pct_bad)
    lows = [-np.inf] + list(edges)
    highs = list(edges) + [np.inf]
    return pd.DataFrame({
        "bin": [f"({lo:,.4g}, {hi:,.4g}]" for lo, hi in zip(lows, highs)],
        "count": total, "bads": bad, "bad_rate": bad / np.maximum(total, 1),
        "share": total / total.sum(), "woe": woe,
        "iv": (pct_good - pct_bad) * woe,
    })


def fit_feature(x, y, name, n_fine=config.N_FINE_BINS, min_share=config.MIN_BIN_SHARE) -> FeatureBinning:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    edges = list(np.unique(np.quantile(x, np.linspace(0, 1, n_fine + 1)[1:-1])))

    def merge_once(edges):
        """Remove the single edge that best fixes size / monotonicity issues."""
        if not edges:
            return None
        idx = np.searchsorted(edges, x, side="left")
        total, bad, _ = _stats(idx, y, len(edges) + 1)
        rate = bad / np.maximum(total, 1)
        share = total / total.sum()
        # 1) bins that are too small: merge with the more similar neighbour
        small = np.where(share < min_share)[0]
        if len(small):
            i = int(small[0])
            if i == 0:
                return edges[1:]
            if i == len(edges):
                return edges[:-1]
            left_gap, right_gap = abs(rate[i] - rate[i - 1]), abs(rate[i] - rate[i + 1])
            drop = i - 1 if left_gap <= right_gap else i
            return edges[:drop] + edges[drop + 1:]
        # 2) enforce a monotonic bad-rate trend in the direction of the overall correlation
        diffs = np.diff(rate)
        direction = np.sign(np.corrcoef(x, y)[0, 1]) or 1.0
        violations = np.where(diffs * direction < 0)[0]
        if len(violations):
            j = int(violations[np.argmin(np.abs(diffs[violations]))])
            return edges[:j] + edges[j + 1:]
        return None

    while (new := merge_once(edges)) is not None:
        edges = new

    idx = np.searchsorted(edges, x, side="left")
    return FeatureBinning(name, edges, _woe_table(idx, y, edges))


class WoETransformer:
    def __init__(self, min_iv=config.MIN_IV):
        self.min_iv = min_iv
        self.binnings: dict[str, FeatureBinning] = {}
        self.selected: list[str] = []

    def fit(self, X: pd.DataFrame, y):
        self.binnings = {c: fit_feature(X[c], y, c) for c in X.columns}
        self.selected = [c for c, b in self.binnings.items() if b.iv >= self.min_iv and len(b.edges) > 0]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({c: self.binnings[c].transform(X[c]) for c in self.selected}, index=X.index)

    def iv_table(self) -> pd.DataFrame:
        rows = [{"feature": c, "iv": b.iv, "n_bins": len(b.edges) + 1, "selected": c in self.selected}
                for c, b in self.binnings.items()]
        df = pd.DataFrame(rows).sort_values("iv", ascending=False)
        df["strength"] = pd.cut(df["iv"], [-np.inf, 0.02, 0.1, 0.3, 0.5, np.inf],
                                labels=["useless", "weak", "medium", "strong", "suspicious"])
        return df.reset_index(drop=True)
