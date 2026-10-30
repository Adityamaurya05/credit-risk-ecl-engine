"""Interactive dashboard:  streamlit run app.py  (run run_pipeline.py first)."""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))
from creditrisk import config, ecl, validation  # noqa: E402
import run_pipeline  # noqa: E402

st.set_page_config(page_title="Credit Risk & ECL Engine", layout="wide")


@st.cache_resource
def load_artifacts():
    path = config.OUTPUT_DIR / "model.pkl"
    if not path.exists():  # first launch on a fresh deploy (e.g. Streamlit Cloud)
        with st.spinner("First run: downloading data and training the model (about a minute) ..."):
            try:
                run_pipeline.main()
            except Exception as exc:  # no internet or UCI moved the file
                st.error(f"Pipeline failed: {exc}. Run `python run_pipeline.py` locally.")
                return None, None
    with open(path, "rb") as f:
        art = pickle.load(f)
    metrics = json.loads((config.OUTPUT_DIR / "metrics.json").read_text())
    return art, metrics


art, m = load_artifacts()
if art is None:
    st.stop()

champ, chal, portfolio = art["champion"], art["challenger"], art["portfolio"]
X_te, y_te = art["X_test"], art["y_test"]

st.title("Credit Card PD Scorecard & IFRS 9 ECL Engine")
if art["sample_mode"]:
    st.warning("Running on SYNTHETIC sample data. Run `python run_pipeline.py` for real results.")

tab_over, tab_model, tab_card, tab_ecl = st.tabs(["Overview", "Model validation", "Scorecard", "ECL & stress testing"])

with tab_over:
    c = m["champion_test"]
    cols = st.columns(6)
    cols[0].metric("Accounts", f"{m['n_accounts']:,}")
    cols[1].metric("Default rate", f"{m['default_rate']:.1%}")
    cols[2].metric("Test AUC", f"{c['auc']:.3f}")
    cols[3].metric("Gini", f"{c['gini']:.3f}")
    cols[4].metric("Total ECL (NT$ mn)", f"{m['ecl']['total_ecl'] / 1e6:,.1f}")
    cols[5].metric("Coverage", f"{m['ecl']['coverage']:.2%}")
    left, right = st.columns(2)
    stage_mix = pd.Series(m["ecl"]["stage_mix"]).rename(index=lambda s: f"Stage {s}")
    left.plotly_chart(px.pie(values=stage_mix.values, names=stage_mix.index, hole=0.5,
                             title="Accounts by IFRS 9 stage"), width="stretch")
    right.plotly_chart(px.histogram(portfolio, x="score", nbins=40, title="Score distribution (portfolio)"),
                       width="stretch")
    st.markdown("**Pipeline:** raw data -> 12 behavioural features -> monotonic WoE binning & IV selection -> "
                "logistic scorecard -> validation -> staging -> scenario-weighted ECL -> Basel capital.")

with tab_model:
    p_te, p_chal = champ.predict_pd(X_te), chal.predict_pd(X_te)
    left, right = st.columns(2)
    fig = go.Figure()
    for name, p in [("WoE scorecard", p_te), ("Gradient boosting", p_chal)]:
        fpr, tpr, _ = roc_curve(y_te, p)
        fig.add_scatter(x=fpr, y=tpr, name=f"{name} (AUC {validation.discrimination(y_te, p)['auc']:.3f})")
    fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash"), name="Random", showlegend=False)
    fig.update_layout(title="ROC curve, test set", xaxis_title="FPR", yaxis_title="TPR")
    left.plotly_chart(fig, width="stretch")
    cal = validation.calibration_table(y_te, p_te)
    fig = px.line(cal, x="predicted_pd", y="observed_dr", markers=True, title="Calibration by decile")
    fig.add_scatter(x=[0, cal["predicted_pd"].max()], y=[0, cal["predicted_pd"].max()],
                    line=dict(dash="dash"), name="Perfect")
    right.plotly_chart(fig, width="stretch")
    left, right = st.columns(2)
    left.subheader("Information Value")
    left.dataframe(champ.woe.iv_table(), hide_index=True, width="stretch")
    bands = validation.score_band_table(y_te, champ.score(X_te))
    right.plotly_chart(px.bar(bands, x="band", y="bad_rate", title="Bad rate by score band (1 = riskiest)"),
                       width="stretch")
    st.caption(f"PSI train vs test: {m['psi_train_test']:.4f} | Hosmer-Lemeshow p-value: "
               f"{m['hosmer_lemeshow']['hl_pvalue']:.3f} | Max VIF: {m['max_vif']:.2f}")

with tab_card:
    pts = champ.points_table()
    left, right = st.columns([3, 2])
    feat = left.selectbox("Feature", champ.features)
    sub = pts[pts["feature"] == feat]
    left.plotly_chart(px.bar(sub, x="bin", y="points", hover_data=["bad_rate", "woe"],
                             title=f"Points by bin: {feat}"), width="stretch")
    right.subheader("Score a new applicant")
    applicant = {}
    for f in champ.features:
        lo, hi = float(X_te[f].min()), float(X_te[f].quantile(0.99))
        applicant[f] = right.slider(f, lo, max(hi, lo + 1e-6), float(np.clip(X_te[f].median(), lo, hi)))
    row = pd.DataFrame([applicant])
    score, pd_hat = champ.score(row)[0], champ.predict_pd(row)[0]
    right.metric("Score", f"{score:.0f}")
    right.metric("Predicted PD", f"{pd_hat:.1%}")
    right.caption(f"Scaling: {config.BASE_SCORE} points = {config.BASE_ODDS}:1 good:bad odds, "
                  f"+{config.PDO} points doubles the odds.")

with tab_ecl:
    st.markdown("Change the assumptions and the provision recalculates. Defaults come from `config.py`.")
    a, b, c = st.columns(3)
    lgd = a.slider("LGD", 0.30, 0.95, config.LGD, 0.05)
    ccf = b.slider("CCF on undrawn limit", 0.0, 1.0, config.CCF, 0.05)
    years = c.slider("Lifetime for Stage 2 (years)", 1, 6, config.LIFETIME_YEARS)
    scen = {}
    for col, (name, s) in zip(st.columns(3), config.SCENARIOS.items()):
        col.markdown(f"**{name}**")
        w = col.slider(f"{name} weight", 0.0, 1.0, s["weight"], 0.05, key=f"w{name}")
        sh = col.slider(f"{name} PD log-odds shift", -1.0, 2.0, s["logit_shift"], 0.05, key=f"s{name}")
        scen[name] = {"weight": w, "logit_shift": sh}
    if sum(s["weight"] for s in scen.values()) == 0:
        st.error("Scenario weights cannot all be zero.")
        st.stop()
    res = ecl.compute_ecl(portfolio, scen, lgd=lgd, ccf=ccf, years=years)
    summ = ecl.stage_summary(res)
    k = st.columns(3)
    k[0].metric("Total ECL (NT$ mn)", f"{summ.loc['Total', 'ecl'] / 1e6:,.1f}",
                f"{summ.loc['Total', 'ecl'] / m['ecl']['total_ecl'] - 1:+.1%} vs base config")
    k[1].metric("Coverage ratio", f"{summ.loc['Total', 'coverage_ratio']:.2%}")
    perf = res[res["stage"] < 3]
    cap = ecl.basel_irb_capital(perf["pd_12m"], lgd, perf["ead"])
    k[2].metric("Basel IRB capital (NT$ mn)", f"{cap['capital'].sum() / 1e6:,.1f}")
    left, right = st.columns(2)
    left.dataframe(summ.style.format({"ead": "{:,.0f}", "ecl": "{:,.0f}", "accounts": "{:,.0f}",
                                      "coverage_ratio": "{:.2%}", "share_of_ecl": "{:.1%}"}),
                   width="stretch")
    by_scen = pd.Series({n: res[f"ecl_{n.lower()}"].sum() / 1e6 for n in scen})
    right.plotly_chart(px.bar(x=by_scen.index, y=by_scen.values, labels={"x": "", "y": "NT$ mn"},
                              title="ECL by scenario"), width="stretch")
    grid = ecl.sensitivity_grid(portfolio) / 1e6
    st.plotly_chart(px.imshow(grid, text_auto=".0f", aspect="auto", color_continuous_scale="Reds",
                              labels=dict(x="PD log-odds shift", y="LGD", color="ECL NT$ mn"),
                              title="Sensitivity: total ECL (NT$ mn) by LGD and PD stress"),
                    width="stretch")
