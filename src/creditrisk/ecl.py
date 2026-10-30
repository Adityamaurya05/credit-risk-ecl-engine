"""IFRS 9 / Ind AS 109 expected credit loss: ECL = PD x LGD x EAD per stage,
probability-weighted across forward-looking scenarios. Plus Basel IRB capital
for comparison (regulatory vs accounting view of the same portfolio)."""

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import config


def logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def inv_logit(z):
    return 1 / (1 + np.exp(-z))


def assign_stage(months_delay_current, pd_12m,
                 s3=config.STAGE3_MIN_MONTHS_DELAY, s2=config.STAGE2_MIN_MONTHS_DELAY,
                 pd_trigger=config.STAGE2_PD_THRESHOLD) -> np.ndarray:
    delay = np.asarray(months_delay_current)
    pd_12m = np.asarray(pd_12m)
    stage = np.ones(len(delay), dtype=int)
    stage[(delay >= s2) | (pd_12m >= pd_trigger)] = 2
    stage[delay >= s3] = 3
    return stage


def exposure_at_default(limit, drawn, ccf=config.CCF) -> np.ndarray:
    drawn = np.clip(np.asarray(drawn, float), 0, None)
    undrawn = np.clip(np.asarray(limit, float) - drawn, 0, None)
    return drawn + ccf * undrawn


def lifetime_pd(pd_12m, years=config.LIFETIME_YEARS) -> np.ndarray:
    """Constant-hazard extrapolation of the 12-month PD over the behavioural life."""
    return 1 - (1 - np.asarray(pd_12m)) ** years


def ecl_single_scenario(pd_12m, stage, ead, lgd=config.LGD, years=config.LIFETIME_YEARS) -> np.ndarray:
    pd_used = np.where(stage == 1, pd_12m, np.where(stage == 2, lifetime_pd(pd_12m, years), 1.0))
    return pd_used * lgd * ead


def compute_ecl(portfolio: pd.DataFrame, scenarios=None, lgd=config.LGD, ccf=config.CCF,
                years=config.LIFETIME_YEARS) -> pd.DataFrame:
    """portfolio needs columns: pd_12m, months_delay_current, limit_bal, drawn."""
    scenarios = scenarios or config.SCENARIOS
    out = portfolio.copy()
    out["stage"] = assign_stage(out["months_delay_current"], out["pd_12m"])
    out["ead"] = exposure_at_default(out["limit_bal"], out["drawn"], ccf)
    total_w = sum(s["weight"] for s in scenarios.values())
    out["ecl"] = 0.0
    for name, s in scenarios.items():
        pd_s = inv_logit(logit(out["pd_12m"].to_numpy()) + s["logit_shift"])
        out[f"ecl_{name.lower()}"] = ecl_single_scenario(pd_s, out["stage"].to_numpy(), out["ead"].to_numpy(), lgd, years)
        out["ecl"] += s["weight"] / total_w * out[f"ecl_{name.lower()}"]
    return out


def stage_summary(ecl_df: pd.DataFrame) -> pd.DataFrame:
    t = ecl_df.groupby("stage").agg(accounts=("ead", "size"), ead=("ead", "sum"), ecl=("ecl", "sum"))
    t.loc["Total"] = t.sum()
    t["coverage_ratio"] = t["ecl"] / t["ead"]
    t["share_of_ecl"] = t["ecl"] / t.loc["Total", "ecl"]
    return t


def basel_irb_capital(pd_12m, lgd, ead, rho=config.BASEL_QRRE_RHO, q=config.BASEL_CONFIDENCE) -> pd.DataFrame:
    """Basel IRB retail formula (no maturity adjustment). Capital covers
    unexpected loss at 99.9%; ECL/provisions cover expected loss."""
    p = np.clip(np.asarray(pd_12m), 0.0003, 0.9999)  # Basel II PD floor of 3bp (Basel III final raises it for QRRE revolvers)
    cond_pd = norm.cdf((norm.ppf(p) + np.sqrt(rho) * norm.ppf(q)) / np.sqrt(1 - rho))
    k = lgd * (cond_pd - p)
    return pd.DataFrame({"k": k, "rwa": k * 12.5 * np.asarray(ead), "capital": k * np.asarray(ead)})


def sensitivity_grid(portfolio: pd.DataFrame, lgds=(0.45, 0.55, 0.65, 0.75, 0.85),
                     shifts=(-0.5, -0.25, 0.0, 0.25, 0.5, 1.0)) -> pd.DataFrame:
    """Total ECL under combinations of LGD and a uniform PD stress (log-odds shift)."""
    rows = []
    for lgd in lgds:
        for s in shifts:
            scen = {"Only": {"weight": 1.0, "logit_shift": s}}
            rows.append({"lgd": lgd, "pd_logit_shift": s,
                         "total_ecl": compute_ecl(portfolio, scen, lgd=lgd)["ecl"].sum()})
    return pd.DataFrame(rows).pivot(index="lgd", columns="pd_logit_shift", values="total_ecl")
