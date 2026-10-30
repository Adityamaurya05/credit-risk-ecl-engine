"""Download, load and clean the UCI credit card default dataset."""

import io
import zipfile

import numpy as np
import pandas as pd
import requests

from . import config

PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYAMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
RAW_COLS = (["ID", "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
            + PAY_COLS + BILL_COLS + PAYAMT_COLS + [config.TARGET])

# The ucimlrepo package names the columns X1..X23 / Y; map them back.
_X_NAMES = (["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
            + PAY_COLS + BILL_COLS + PAYAMT_COLS)
UCIMLREPO_MAP = {f"X{i + 1}": name for i, name in enumerate(_X_NAMES)} | {"Y": config.TARGET}


def download(dest_dir=config.DATA_DIR) -> "Path":
    """Download the zip from UCI and extract the .xls into data/."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    resp = requests.get(config.UCI_URL, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        xls = [n for n in zf.namelist() if n.lower().endswith((".xls", ".xlsx", ".csv"))]
        if not xls:
            raise RuntimeError(f"No data file in archive: {zf.namelist()}")
        zf.extract(xls[0], dest_dir)
        return dest_dir / xls[0]


def find_local_file(data_dir=config.DATA_DIR):
    for pattern in ("*.xls", "*.xlsx", "*.csv"):
        hits = sorted(p for p in data_dir.rglob(pattern) if "sample" not in p.name)
        if hits:
            return hits[0]
    return None


def _read_any(path) -> pd.DataFrame:
    path = str(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    # The original .xls has a first row of X1..X23 labels above the real header.
    df = pd.read_excel(path, header=None)
    header_row = df.index[df.astype(str).apply(lambda r: r.str.upper().eq("LIMIT_BAL").any(), axis=1)]
    row = int(header_row[0]) if len(header_row) else 0
    df.columns = df.iloc[row].astype(str).str.strip()
    return df.iloc[row + 1:].reset_index(drop=True)


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=lambda c: str(c).strip())
    df = df.rename(columns=UCIMLREPO_MAP)
    rename = {}
    for c in df.columns:
        lc = c.lower()
        if lc.startswith("default"):
            rename[c] = config.TARGET
        elif c.upper() == "PAY_1":  # some copies rename PAY_0 -> PAY_1
            rename[c] = "PAY_0"
    df = df.rename(columns=rename)
    if "ID" not in df.columns:
        df.insert(0, "ID", np.arange(1, len(df) + 1))
    missing = set(RAW_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")
    return df[RAW_COLS].apply(pd.to_numeric, errors="coerce")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the documented data-quality issues in the raw file.

    * EDUCATION codes 0, 5, 6 and MARRIAGE code 0 are undocumented -> 'other'.
    * PAY_x values -2 (no consumption) and 0 (revolving, paid minimum) are not
      delinquencies, so delay is floored at 0 months.
    """
    df = df.dropna().copy()
    df["EDUCATION"] = df["EDUCATION"].where(df["EDUCATION"].isin([1, 2, 3]), 4)
    df["MARRIAGE"] = df["MARRIAGE"].where(df["MARRIAGE"].isin([1, 2]), 3)
    for c in PAY_COLS:
        df[c] = df[c].clip(lower=0)
    df = df.drop_duplicates(subset="ID")
    return df.astype({config.TARGET: int}).reset_index(drop=True)


def load(path=None, allow_download=True) -> pd.DataFrame:
    path = path or find_local_file()
    if path is None:
        if not allow_download:
            raise FileNotFoundError("No dataset in data/. Run with download enabled or use --sample.")
        path = download()
    return clean(standardise_columns(_read_any(path)))


def make_sample(n=6000, seed=0) -> pd.DataFrame:
    """Synthetic data with the same schema, for tests and offline demos ONLY.
    Never report results from this data."""
    rng = np.random.default_rng(seed)
    limit = rng.choice([20, 50, 80, 120, 200, 300, 500], n) * 1000
    risk = rng.normal(0, 1, n)
    delay = np.clip(np.round(rng.poisson(np.exp(risk - 1.2), (6, n))), 0, 8).T
    util = np.clip(rng.beta(2, 3, n) + 0.15 * risk, 0, 1.2)
    bills = np.column_stack([limit * np.clip(util + rng.normal(0, .05, n), 0, None) for _ in range(6)])
    pay_ratio = np.clip(rng.beta(2, 5, n) - 0.05 * risk, 0, 1)
    pays = bills * pay_ratio[:, None] * rng.uniform(.8, 1.2, (n, 6))
    logit = -1.6 + 0.9 * delay[:, 0] + 0.8 * util - 2.0 * pay_ratio + 0.3 * risk
    df = pd.DataFrame({
        "ID": np.arange(1, n + 1), "LIMIT_BAL": limit,
        "SEX": rng.integers(1, 3, n), "EDUCATION": rng.integers(1, 5, n),
        "MARRIAGE": rng.integers(1, 4, n), "AGE": rng.integers(21, 70, n),
    })
    for i, c in enumerate(PAY_COLS):
        df[c] = delay[:, i]
    for i, c in enumerate(BILL_COLS):
        df[c] = bills[:, i].round()
    for i, c in enumerate(PAYAMT_COLS):
        df[c] = pays[:, i].round()
    df[config.TARGET] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return clean(df)
