"""Charts and the markdown model validation report."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from . import config  # noqa: E402

NAVY, TEAL, GREY, RED = "#1f3a5f", "#2a9d8f", "#8d99ae", "#c0392b"


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_roc(y, preds: dict, path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for (name, p), color in zip(preds.items(), [NAVY, TEAL]):
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=color, label=name)
    ax.plot([0, 1], [0, 1], ls="--", color=GREY, lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (test set)")
    ax.legend(frameon=False)
    _save(fig, path)


def plot_calibration(cal, path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(cal["predicted_pd"], cal["observed_dr"], "o-", color=NAVY, label="Scorecard")
    lim = max(cal["predicted_pd"].max(), cal["observed_dr"].max()) * 1.05
    ax.plot([0, lim], [0, lim], ls="--", color=GREY, lw=1, label="Perfect calibration")
    ax.set(xlabel="Predicted PD (decile mean)", ylabel="Observed default rate", title="Calibration (test set)")
    ax.legend(frameon=False)
    _save(fig, path)


def plot_score_bands(bands, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(bands["band"].astype(str), bands["bad_rate"], color=NAVY)
    ax.set(xlabel="Score band (1 = lowest score)", ylabel="Observed bad rate", title="Rank ordering by score band")
    _save(fig, path)


def plot_ecl_by_stage(summary, path):
    s = summary.drop(index="Total")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].bar([f"Stage {i}" for i in s.index], s["ecl"] / 1e6, color=[TEAL, "#e9c46a", RED])
    axes[0].set(title="ECL by stage (NT$ mn)")
    axes[1].bar([f"Stage {i}" for i in s.index], s["coverage_ratio"] * 100, color=[TEAL, "#e9c46a", RED])
    axes[1].set(title="Coverage ratio (ECL / EAD, %)")
    _save(fig, path)


def _fmt(v):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
        return str(v)
    if float(v).is_integer():
        return f"{int(v):,}"
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.4f}"


def _md_table(df):
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(v) for v in row) + " |")
    return "\n".join(lines)


def validation_report(m: dict, tables: dict, sample_mode: bool) -> str:
    warn = ("\n> **WARNING: generated from SYNTHETIC sample data. Do not quote these numbers.**\n"
            if sample_mode else "")
    champ, chal = m["champion_test"], m["challenger_test"]
    hl_ok = m["hosmer_lemeshow"]["hl_pvalue"] > 0.05
    return f"""# Model Validation Report: Credit Card PD Scorecard
{warn}
## 1. Purpose and scope
Probability-of-default (PD) model for a revolving credit card portfolio, used as the
PD input to IFRS 9 / Ind AS 109 expected credit loss (ECL) provisioning.

* Data: UCI Default of Credit Card Clients, {m['n_accounts']:,} accounts, observed default rate {m['default_rate']:.2%}.
* Split: stratified {1 - config.TEST_SIZE:.0%} train / {config.TEST_SIZE:.0%} test (random_state={config.RANDOM_STATE}).
* Protected attributes excluded from the model: {', '.join(config.PROTECTED)}.

## 2. Performance (hold-out test set)

| Metric | Champion: WoE scorecard | Challenger: gradient boosting |
|---|---|---|
| AUC | {champ['auc']:.3f} | {chal['auc']:.3f} |
| Gini | {champ['gini']:.3f} | {chal['gini']:.3f} |
| KS | {champ['ks']:.3f} | {chal['ks']:.3f} |
| Brier score | {champ['brier']:.4f} | {chal['brier']:.4f} |

Gini gap (challenger - champion): **{chal['gini'] - champ['gini']:+.3f}**. The champion is kept because every
score can be explained bin by bin to a customer, an auditor or a regulator; the challenger is used as a
benchmark that tells us how much predictive power the simpler model leaves on the table.

## 3. Stability and calibration
* PSI of score distribution, train vs test: **{m['psi_train_test']:.4f}** (<0.10 = stable).
* Hosmer-Lemeshow: chi2 = {m['hosmer_lemeshow']['hl_chi2']:.2f}, p = {m['hosmer_lemeshow']['hl_pvalue']:.3f}
  ({'no evidence of miscalibration at 5%' if hl_ok else 'calibration rejected at 5%; recalibration recommended'}).
* Rank ordering by score band is monotonic: **{m['rank_ordered']}**.

{_md_table(tables['calibration'])}

## 4. Variables (Information Value)
{_md_table(tables['iv'])}

Max VIF across selected WoE features: {m['max_vif']:.2f} (>5 would signal multicollinearity).

## 5. Fairness check (not a model input)
Approval rate by SEX at a PD cut-off of {m['fairness_threshold']:.0%}; ratio vs best-treated group
(the "four-fifths rule" flags ratios below 0.80):

{_md_table(tables['fairness'])}

## 6. ECL results (full portfolio, probability-weighted)
Total EAD NT${m['ecl']['total_ead']/1e6:,.1f} mn, total ECL NT${m['ecl']['total_ecl']/1e6:,.1f} mn,
coverage {m['ecl']['coverage']:.2%}. Downside-only ECL is {m['ecl']['downside_uplift']:+.1%} vs weighted.
Basel IRB capital (QRRE, illustrative): NT${m['basel']['capital']/1e6:,.1f} mn, RWA density {m['basel']['rwa_density']:.1%}.

{_md_table(tables['stage_summary'])}

## 7. Limitations (stated up front, as a validator would)
1. **Target definition.** The label is default in the month after a 6-month window, not a 12-month
   default. It is used as a 12-month PD proxy.
2. **No time dimension.** One snapshot (Apr-Sep 2005), so no out-of-time test and no through-the-cycle view.
3. **LGD, CCF, lifetime and scenario weights are assumptions** (see `config.py`), not estimated from data.
   Section 6 should be read with the LGD x PD sensitivity grid in `outputs/ecl_sensitivity.csv`.
4. **No origination PD**, so SICR uses an absolute PD trigger plus the 30 DPD backstop instead of a
   relative PD change since origination.
5. Data is from Taiwan in 2005 and does not represent any current Indian or US portfolio.
"""


def resume_bullets(m: dict) -> str:
    champ, chal = m["champion_test"], m["challenger_test"]
    return f"""# Resume bullets (generated from YOUR run; numbers are real outputs)

**Credit Risk Scorecard & IFRS 9 ECL Engine** | Python, scikit-learn, Streamlit | [GitHub link]
- Built a WoE/logistic PD scorecard on {m['n_accounts']:,} credit card accounts (Gini {champ['gini']:.2f}, KS {champ['ks']:.2f} on hold-out), benchmarked against a gradient-boosting challenger (Gini {chal['gini']:.2f}) and kept the explainable model for governance
- Engineered {m['n_features_built']} behavioural features (utilisation, delinquency, payment ratio), selected {m['n_features_selected']} by Information Value with monotonic binning, and validated with PSI ({m['psi_train_test']:.3f}), Hosmer-Lemeshow, VIF and a four-fifths-rule fairness check
- Implemented IFRS 9 / Ind AS 109 staging and probability-weighted ECL across 3 macro scenarios (NT${m['ecl']['total_ecl']/1e6:,.0f} mn, {m['ecl']['coverage']:.1%} coverage) with Basel IRB capital comparison and LGD/PD sensitivity grid in an interactive Streamlit dashboard
"""
