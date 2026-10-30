"""Champion (WoE logistic scorecard) and challenger (gradient boosting) PD models."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from . import config
from .woe import WoETransformer


class ScorecardModel:
    """Logistic regression on WoE features, scaled to integer scorecard points."""

    def __init__(self, base_score=config.BASE_SCORE, base_odds=config.BASE_ODDS, pdo=config.PDO):
        self.woe = WoETransformer()
        self.lr = LogisticRegression(C=1.0, max_iter=1000)
        self.factor = pdo / np.log(2)
        self.offset = base_score - self.factor * np.log(base_odds)

    def fit(self, X: pd.DataFrame, y):
        self.woe.fit(X, y)
        self._drop_correlated(X)
        self.lr.fit(self.woe.transform(X), y)
        # A WoE feature with a positive coefficient contradicts its own binning
        # (higher WoE = safer). Drop such features and refit, as a validator would ask.
        while (bad := [f for f, c in zip(self.woe.selected, self.lr.coef_[0]) if c > 0]):
            self.woe.selected = [f for f in self.woe.selected if f not in bad]
            self.lr.fit(self.woe.transform(X), y)
        return self

    def _drop_correlated(self, X, max_corr=config.MAX_WOE_CORR):
        """Of any pair of WoE features correlated above max_corr, keep the one with higher IV."""
        ranked = sorted(self.woe.selected, key=lambda f: self.woe.binnings[f].iv, reverse=True)
        W = self.woe.transform(X)
        kept = []
        for f in ranked:
            if all(abs(np.corrcoef(W[f], W[k])[0, 1]) <= max_corr for k in kept):
                kept.append(f)
        self.woe.selected = [f for f in self.woe.selected if f in kept]

    @property
    def features(self):
        return list(self.woe.selected)

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        return self.lr.predict_proba(self.woe.transform(X))[:, 1]

    def points_table(self) -> pd.DataFrame:
        """Points per attribute: score = sum of points, higher = lower risk."""
        n = len(self.features)
        b0 = self.lr.intercept_[0]
        rows = []
        for f, b in zip(self.features, self.lr.coef_[0]):
            for _, r in self.woe.binnings[f].table.iterrows():
                pts = -(b * r["woe"] + b0 / n) * self.factor + self.offset / n
                rows.append({"feature": f, "bin": r["bin"], "woe": r["woe"],
                             "bad_rate": r["bad_rate"], "points": int(round(pts))})
        return pd.DataFrame(rows)

    def score(self, X: pd.DataFrame) -> np.ndarray:
        pts = self.points_table()
        total = np.zeros(len(X))
        for f in self.features:
            bins = self.woe.binnings[f].bin_index(X[f])
            total += pts.loc[pts["feature"] == f, "points"].to_numpy()[bins]
        return total

    def coefficients(self) -> pd.DataFrame:
        return pd.DataFrame({"feature": self.features, "coef": self.lr.coef_[0]})


class ChallengerModel:
    """Gradient boosting on raw features: more accurate, much harder to explain."""

    def __init__(self):
        self.model = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
            l2_regularization=1.0, early_stopping=True, random_state=config.RANDOM_STATE)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_pd(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]
