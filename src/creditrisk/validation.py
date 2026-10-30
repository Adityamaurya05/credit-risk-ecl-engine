"""Model validation metrics a bank model-risk (SR 11-7 style) review would ask for."""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve


def ks_statistic(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def discrimination(y, p) -> dict:
    auc = roc_auc_score(y, p)
    return {"auc": float(auc), "gini": float(2 * auc - 1),
            "ks": ks_statistic(y, p), "brier": float(brier_score_loss(y, p))}


def calibration_table(y, p, n_bins=10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["decile"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False) + 1
    t = df.groupby("decile").agg(n=("y", "size"), predicted_pd=("p", "mean"), observed_dr=("y", "mean"))
    return t.reset_index()


def hosmer_lemeshow(y, p, n_bins=10) -> dict:
    t = calibration_table(y, p, n_bins)
    exp_bad = t["predicted_pd"] * t["n"]
    obs_bad = t["observed_dr"] * t["n"]
    exp_good = t["n"] - exp_bad
    obs_good = t["n"] - obs_bad
    chi2 = float((((obs_bad - exp_bad) ** 2) / exp_bad + ((obs_good - exp_good) ** 2) / exp_good).sum())
    return {"hl_chi2": chi2, "hl_pvalue": float(1 - stats.chi2.cdf(chi2, n_bins - 2))}


def psi(expected, actual, n_bins=10) -> float:
    """Population Stability Index. <0.1 stable, 0.1-0.25 monitor, >0.25 investigate."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)[1:-1]))
    e = np.bincount(np.searchsorted(edges, expected), minlength=len(edges) + 1) / len(expected)
    a = np.bincount(np.searchsorted(edges, actual), minlength=len(edges) + 1) / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def score_band_table(y, score, n_bands=10) -> pd.DataFrame:
    """Bad rate by score band. A good scorecard rank-orders: bad rate falls as score rises."""
    df = pd.DataFrame({"y": np.asarray(y), "score": np.asarray(score)})
    df["band"] = pd.qcut(df["score"].rank(method="first"), n_bands, labels=False) + 1
    t = df.groupby("band").agg(min_score=("score", "min"), max_score=("score", "max"),
                               n=("y", "size"), bad_rate=("y", "mean")).reset_index()
    t["cum_bad_capture"] = (t["bad_rate"] * t["n"]).cumsum() / (t["bad_rate"] * t["n"]).sum()
    return t


def is_rank_ordered(band_table: pd.DataFrame, tolerance=0.01) -> bool:
    """True if bad rate never rises by more than `tolerance` from one band to the next."""
    return bool(np.all(np.diff(band_table["bad_rate"].to_numpy()) <= tolerance))


def vif(X: pd.DataFrame) -> pd.DataFrame:
    """Variance inflation factor from the inverse correlation matrix."""
    corr = np.corrcoef(X.to_numpy(float), rowvar=False)
    vals = np.diag(np.linalg.pinv(corr))
    return pd.DataFrame({"feature": X.columns, "vif": vals}).sort_values("vif", ascending=False)


def fairness_table(y, p, group: pd.Series, threshold: float) -> pd.DataFrame:
    """Approval rate and bad rate by a protected group. Not used in the model."""
    df = pd.DataFrame({"y": np.asarray(y), "approve": np.asarray(p) < threshold, "g": np.asarray(group)})
    t = df.groupby("g").agg(n=("y", "size"), approval_rate=("approve", "mean"), bad_rate=("y", "mean"))
    t["approval_ratio_vs_max"] = t["approval_rate"] / t["approval_rate"].max()
    return t.reset_index().rename(columns={"g": "group"})
