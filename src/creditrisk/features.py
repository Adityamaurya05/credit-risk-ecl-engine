"""Behavioural features a card risk team would actually build."""

import numpy as np
import pandas as pd

from .data import BILL_COLS, PAY_COLS, PAYAMT_COLS


def _safe_div(a, b):
    return np.where(b > 0, a / np.where(b > 0, b, 1), 0.0)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    limit = df["LIMIT_BAL"].to_numpy(float)
    bills = df[BILL_COLS].to_numpy(float).clip(min=0)
    pays = df[PAYAMT_COLS].to_numpy(float)
    delays = df[PAY_COLS].to_numpy(float)

    out["limit_bal"] = limit
    out["age"] = df["AGE"].to_numpy(float)
    out["education"] = df["EDUCATION"].to_numpy(float)

    # Delinquency history (PAY_0 is the most recent month)
    out["months_delay_current"] = delays[:, 0]
    out["max_delay_6m"] = delays.max(axis=1)
    out["n_months_delinquent_6m"] = (delays > 0).sum(axis=1)

    # Utilisation
    out["utilisation_current"] = np.clip(_safe_div(bills[:, 0], limit), 0, 2)
    out["utilisation_avg_6m"] = np.clip(_safe_div(bills.mean(axis=1), limit), 0, 2)
    out["utilisation_trend"] = np.clip(_safe_div(bills[:, 0] - bills[:, 5], limit), -2, 2)

    # Payment behaviour: payment in month t settles the bill of month t+1
    paid_vs_prev_bill = np.clip(_safe_div(pays[:, :5], bills[:, 1:]), 0, 1)
    out["payment_ratio_recent"] = paid_vs_prev_bill[:, 0]
    out["payment_ratio_avg"] = paid_vs_prev_bill.mean(axis=1)
    out["n_zero_payments_6m"] = (pays == 0).sum(axis=1)
    return out
