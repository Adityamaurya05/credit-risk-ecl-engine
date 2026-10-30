# Credit Card PD Scorecard & IFRS 9 Expected Credit Loss Engine

An end-to-end credit risk workflow of the kind a bank's retail risk team runs. It starts from raw
account data, builds an explainable probability-of-default (PD) scorecard and validates it the way a
model-risk team would. It then turns the PDs into an **IFRS 9 / Ind AS 109 loan-loss provision**
across macroeconomic scenarios and compares that provision with **Basel IRB regulatory capital**.

> **Business question:** *How much should a card issuer set aside for expected losses, which
> customers drive that number, and how sensitive is it to the economy and to our own assumptions?*

![ROC](outputs/figures/roc.png) ![ECL](outputs/figures/ecl_by_stage.png)

## What it does

| Step | What | Why a bank cares |
|---|---|---|
| 1. Data | 30,000 card accounts (UCI), cleaning of undocumented codes | Data quality is the first finding in any model review |
| 2. Features | 12 behavioural features: utilisation, delinquency depth/frequency, payment ratio | These are the standard drivers in card risk |
| 3. WoE binning | Quantile bins merged until bad rate is **monotonic** and every bin is >= 5% of accounts | Regulators and auditors expect risk to move in one direction per variable |
| 4. Selection | Information Value filter + correlation pruning (keep higher-IV feature) | Removes weak and redundant variables (checked with VIF) |
| 5. Champion | Logistic regression on WoE, scaled to **scorecard points** (600 = 50:1 odds, PDO 20) | Every decision can be explained to a customer |
| 6. Challenger | Gradient boosting on raw features | Shows how much accuracy the simple model gives up |
| 7. Validation | AUC, Gini, KS, Brier, calibration by decile, Hosmer-Lemeshow, PSI, rank ordering, VIF, four-fifths fairness test | Mirrors an SR 11-7 style independent validation |
| 8. Staging | Stage 3 = 90+ DPD; Stage 2 = 30+ DPD backstop or high PD; else Stage 1 | IFRS 9 / Ind AS 109 three-stage model |
| 9. ECL | PD x LGD x EAD, 12-month (Stage 1) vs lifetime (Stage 2), weighted over Upside/Base/Downside | The number that lands in the P&L as a provision |
| 10. Capital | Basel IRB retail (QRRE) formula at 99.9% | Expected loss (provisions) vs unexpected loss (capital) |
| 11. Dashboard | Streamlit app: applicant scoring, live ECL with sliders for LGD/CCF/scenarios, sensitivity heatmap | Lets a non-technical stakeholder test the assumptions |

## Results

After you run the pipeline, the full numbers are in **[outputs/validation_report.md](outputs/validation_report.md)**
and `outputs/metrics.json`. The report is generated from code, so nothing in it is typed in by hand.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py          # downloads the UCI data on first run (about 5 MB)
streamlit run app.py
python -m pytest -q             # 9 unit tests: binning, scaling, staging, ECL math
```

If the automatic download fails, download *Default of Credit Card Clients* from the
[UCI ML Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) and put the
`.xls` file in `data/`.

`python run_pipeline.py --sample` runs on **synthetic** data for an offline smoke test. Never quote those numbers.

## Project structure

```
src/creditrisk/
  config.py      every assumption in one place (LGD, CCF, staging rules, scenarios)
  data.py        download, schema checks, cleaning
  features.py    behavioural features
  woe.py         monotonic WoE binning and Information Value
  models.py      scorecard (champion) and gradient boosting (challenger)
  validation.py  AUC/Gini/KS, calibration, HL test, PSI, VIF, fairness
  ecl.py         staging, EAD, lifetime PD, scenario-weighted ECL, Basel capital
  report.py      charts, validation report, resume bullets
run_pipeline.py  orchestrates everything and writes outputs/
app.py           Streamlit dashboard
```

## Key assumptions and limitations

All assumptions are in `src/creditrisk/config.py`, and the dashboard lets you change them.

* **Target is next-month default, used as a 12-month PD proxy.** A real IFRS 9 model would use a 12-month performance window.
* **One snapshot (Apr-Sep 2005), so no out-of-time validation is possible.** The test set is a random hold-out.
* **LGD (65%), CCF (50%), behavioural life (3 years) and scenario weights are judgement calls.** The dataset has no recovery data. The sensitivity grid shows how much they matter.
* **No origination PD.** SICR therefore uses an absolute PD trigger plus the 30 DPD backstop, not a relative change since origination.
* Taiwan, 2005. The data is not representative of any current portfolio.

## Data source

*Default of Credit Card Clients*, UCI Machine Learning Repository (dataset id 350), donated by I-Cheng Yeh.
Original study: Yeh, I-C. & Lien, C-H. (2009), *Expert Systems with Applications*.
Copy the exact citation from the UCI dataset page.
