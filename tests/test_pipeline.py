import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from creditrisk import data, ecl, validation  # noqa: E402
from creditrisk.features import build_features  # noqa: E402
from creditrisk.models import ScorecardModel  # noqa: E402
from creditrisk.woe import fit_feature  # noqa: E402


@pytest.fixture(scope="module")
def sample():
    raw = data.make_sample(n=3000, seed=1)
    return raw, build_features(raw), raw["default"].to_numpy()


def test_clean_maps_undocumented_codes():
    df = data.make_sample(n=200)
    df.loc[0, "EDUCATION"], df.loc[1, "MARRIAGE"], df.loc[2, "PAY_0"] = 6, 0, -2
    out = data.clean(df)
    assert set(out["EDUCATION"]) <= {1, 2, 3, 4}
    assert set(out["MARRIAGE"]) <= {1, 2, 3}
    assert out["PAY_0"].min() >= 0


def test_standardise_handles_ucimlrepo_names():
    df = data.make_sample(n=50).drop(columns="ID")
    renamed = df.rename(columns={v: k for k, v in data.UCIMLREPO_MAP.items()})
    assert list(data.standardise_columns(renamed).columns) == data.RAW_COLS


def test_woe_bins_are_monotonic_and_large_enough(sample):
    _, X, y = sample
    b = fit_feature(X["utilisation_current"], y, "util")
    rates = b.table["bad_rate"].to_numpy()
    assert np.all(np.diff(rates) >= 0) or np.all(np.diff(rates) <= 0)
    assert b.table["share"].min() >= 0.05 - 1e-9
    assert b.iv > 0.02


def test_scorecard_scaling_and_direction(sample):
    _, X, y = sample
    model = ScorecardModel().fit(X, y)
    assert (model.lr.coef_[0] <= 0).all()
    p, s = model.predict_pd(X), model.score(X)
    # higher score must mean lower PD
    assert np.corrcoef(p, s)[0, 1] < -0.95
    # PDO check: +20 points ~ doubling of good:bad odds
    log_odds_good = np.log((1 - p) / p)
    slope = np.polyfit(s, log_odds_good, 1)[0]
    assert slope == pytest.approx(np.log(2) / 20, rel=0.05)


def test_validation_metrics_ranges(sample):
    _, X, y = sample
    p = ScorecardModel().fit(X, y).predict_pd(X)
    d = validation.discrimination(y, p)
    assert 0.5 < d["auc"] <= 1 and d["gini"] == pytest.approx(2 * d["auc"] - 1)
    assert validation.psi(p, p) == pytest.approx(0, abs=1e-9)
    assert validation.psi(p, np.clip(p * 2, 0, 1)) > 0.1


def test_staging_rules():
    stage = ecl.assign_stage(np.array([0, 0, 1, 3, 5]), np.array([0.05, 0.4, 0.05, 0.05, 0.9]))
    assert list(stage) == [1, 2, 2, 3, 3]


def test_ecl_math():
    port = pd.DataFrame({"pd_12m": [0.02, 0.02, 0.5], "months_delay_current": [0, 1, 4],
                         "limit_bal": [100.0, 100.0, 100.0], "drawn": [40.0, 40.0, 100.0]})
    scen = {"Base": {"weight": 1.0, "logit_shift": 0.0}}
    out = ecl.compute_ecl(port, scen, lgd=0.5, ccf=0.5, years=3)
    assert list(out["ead"]) == [70.0, 70.0, 100.0]
    assert out["ecl"].iloc[0] == pytest.approx(0.02 * 0.5 * 70)                  # stage 1: 12m PD
    assert out["ecl"].iloc[1] == pytest.approx((1 - 0.98 ** 3) * 0.5 * 70)       # stage 2: lifetime
    assert out["ecl"].iloc[2] == pytest.approx(1.0 * 0.5 * 100)                  # stage 3: PD = 1


def test_downside_scenario_increases_ecl():
    port = pd.DataFrame({"pd_12m": [0.05, 0.1], "months_delay_current": [0, 0],
                         "limit_bal": [100.0, 100.0], "drawn": [50.0, 50.0]})
    base = ecl.compute_ecl(port, {"B": {"weight": 1, "logit_shift": 0}})["ecl"].sum()
    down = ecl.compute_ecl(port, {"D": {"weight": 1, "logit_shift": 0.5}})["ecl"].sum()
    assert down > base


def test_basel_capital_positive_and_bounded():
    cap = ecl.basel_irb_capital(np.array([0.01, 0.05, 0.2]), 0.65, np.array([100, 100, 100]))
    assert (cap["k"] > 0).all() and (cap["k"] < 0.65).all()
