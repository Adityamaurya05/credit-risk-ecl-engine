"""End-to-end pipeline: data -> features -> scorecard -> validation -> ECL -> report.

    python run_pipeline.py            # real UCI data (downloads on first run)
    python run_pipeline.py --sample   # synthetic data, offline smoke test only
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent / "src"))

from creditrisk import config, data, ecl, report, validation  # noqa: E402
from creditrisk.features import build_features  # noqa: E402
from creditrisk.models import ChallengerModel, ScorecardModel  # noqa: E402


def main(sample=False, data_path=None):
    out = config.OUTPUT_DIR
    out.mkdir(exist_ok=True)
    print("1/6 Loading data ...")
    raw = data.make_sample() if sample else data.load(data_path)
    X = build_features(raw)
    y = raw[config.TARGET].to_numpy()
    print(f"    {len(raw):,} accounts, default rate {y.mean():.2%}")

    X_tr, X_te, y_tr, y_te, raw_tr, raw_te = train_test_split(
        X, y, raw, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE)

    print("2/6 Fitting champion scorecard and challenger ...")
    champ = ScorecardModel().fit(X_tr, y_tr)
    chal = ChallengerModel().fit(X_tr, y_tr)
    p_te, p_te_chal = champ.predict_pd(X_te), chal.predict_pd(X_te)

    print("3/6 Validating ...")
    s_tr, s_te = champ.score(X_tr), champ.score(X_te)
    cal = validation.calibration_table(y_te, p_te)
    bands = validation.score_band_table(y_te, s_te)
    threshold = float(np.quantile(p_te, 0.80))  # approve the 80% lowest-PD applicants
    fair = validation.fairness_table(y_te, p_te, raw_te["SEX"].map({1: "male", 2: "female"}), threshold)
    vif = validation.vif(champ.woe.transform(X_tr))

    print("4/6 Computing IFRS 9 ECL and Basel capital ...")
    portfolio = pd.DataFrame({
        "ID": raw["ID"], "pd_12m": champ.predict_pd(X),
        "months_delay_current": X["months_delay_current"], "limit_bal": X["limit_bal"],
        "drawn": raw["BILL_AMT1"].clip(lower=0), "score": champ.score(X),
    })
    ecl_df = ecl.compute_ecl(portfolio)
    summary = ecl.stage_summary(ecl_df)
    performing = ecl_df[ecl_df["stage"] < 3]
    basel = ecl.basel_irb_capital(performing["pd_12m"], config.LGD, performing["ead"])
    sens = ecl.sensitivity_grid(portfolio)

    metrics = {
        "sample_mode": sample,
        "n_accounts": int(len(raw)), "default_rate": float(y.mean()),
        "n_features_built": int(X.shape[1]), "n_features_selected": len(champ.features),
        "features_selected": champ.features,
        "champion_test": validation.discrimination(y_te, p_te),
        "champion_train": validation.discrimination(y_tr, champ.predict_pd(X_tr)),
        "challenger_test": validation.discrimination(y_te, p_te_chal),
        "psi_train_test": validation.psi(s_tr, s_te),
        "hosmer_lemeshow": validation.hosmer_lemeshow(y_te, p_te),
        "rank_ordered": validation.is_rank_ordered(bands),
        "max_vif": float(vif["vif"].max()), "fairness_threshold": threshold,
        "ecl": {
            "total_ead": float(summary.loc["Total", "ead"]),
            "total_ecl": float(summary.loc["Total", "ecl"]),
            "coverage": float(summary.loc["Total", "coverage_ratio"]),
            "downside_uplift": float(ecl_df["ecl_downside"].sum() / ecl_df["ecl"].sum() - 1),
            "stage_mix": {int(k): int(v) for k, v in ecl_df["stage"].value_counts().sort_index().items()},
        },
        "basel": {"capital": float(basel["capital"].sum()),
                  "rwa_density": float(basel["rwa"].sum() / performing["ead"].sum())},
    }

    print("5/6 Writing tables, charts and model ...")
    tables = {"calibration": cal, "iv": champ.woe.iv_table(), "fairness": fair,
              "stage_summary": summary.reset_index()}
    champ.woe.iv_table().to_csv(out / "iv_table.csv", index=False)
    champ.points_table().to_csv(out / "scorecard_points.csv", index=False)
    cal.to_csv(out / "calibration_table.csv", index=False)
    bands.to_csv(out / "score_bands.csv", index=False)
    summary.to_csv(out / "ecl_stage_summary.csv")
    sens.to_csv(out / "ecl_sensitivity.csv")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    figs = out / "figures"
    report.plot_roc(y_te, {"WoE scorecard": p_te, "Gradient boosting": p_te_chal}, figs / "roc.png")
    report.plot_calibration(cal, figs / "calibration.png")
    report.plot_score_bands(bands, figs / "score_bands.png")
    report.plot_ecl_by_stage(summary, figs / "ecl_by_stage.png")
    with open(out / "model.pkl", "wb") as f:
        pickle.dump({"champion": champ, "challenger": chal, "portfolio": portfolio,
                     "X_test": X_te, "y_test": y_te, "sample_mode": sample}, f)

    print("6/6 Writing validation report and resume bullets ...")
    (out / "validation_report.md").write_text(report.validation_report(metrics, tables, sample))
    (out / "resume_bullets.md").write_text(report.resume_bullets(metrics))

    c = metrics["champion_test"]
    print(f"\nDone. Test AUC {c['auc']:.3f} | Gini {c['gini']:.3f} | KS {c['ks']:.3f} | "
          f"ECL NT${metrics['ecl']['total_ecl'] / 1e6:,.1f} mn")
    if sample:
        print("NOTE: synthetic sample data - do not quote these numbers.")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="use synthetic data (offline test)")
    ap.add_argument("--data", type=Path, help="path to a local copy of the UCI file")
    args = ap.parse_args()
    main(sample=args.sample, data_path=args.data)
