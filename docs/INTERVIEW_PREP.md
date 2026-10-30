# Interview Prep: Credit Risk Scorecard & IFRS 9 ECL

Read this until you can answer every question **without looking**. Interviewers at banks will
dig into one or two decisions and keep asking "why?"

---

## The 60-second pitch (memorise this shape, use your own numbers)

> "I built an end-to-end credit risk workflow on 30,000 credit card accounts. First I built a
> probability-of-default scorecard using Weight-of-Evidence binning and logistic regression, because
> banks need models they can explain to regulators. It reached a Gini of **[X]** on hold-out data. I
> benchmarked it against gradient boosting, which only added **[Y]** Gini, so the explainable model was
> the right trade-off. Then I validated it like a model-risk team would: calibration, PSI for stability,
> and a fairness check. Finally I turned the PDs into an IFRS 9 provision: three stages, 12-month vs
> lifetime loss, and weighted across three economic scenarios. The total came to **[Z]**, and the
> dashboard shows that the number is most sensitive to the LGD assumption, which is exactly what I'd want
> management to challenge."

---

## Core concepts, explained simply

**PD, LGD, EAD.** Expected loss = PD x LGD x EAD.
* PD = probability the customer defaults.
* LGD = share of the exposure you lose if they default (after recoveries).
* EAD = how much they owe at the moment of default.

**Why EAD > current balance for cards:** customers in trouble draw down their unused limit before
defaulting. CCF (credit conversion factor) = share of the unused limit assumed to be drawn.
`EAD = drawn + CCF x (limit - drawn)`.

**Weight of Evidence (WoE).** For each bin of a variable: `ln(% of goods in bin / % of bads in bin)`.
Positive = safer than average. It turns any variable into a number on the same "log-odds" scale, so
logistic regression on WoE gives coefficients that are directly comparable.

**Information Value (IV).** How much a variable separates goods from bads. <0.02 useless,
0.02-0.1 weak, 0.1-0.3 medium, 0.3-0.5 strong, >0.5 "too good to be true" (check for leakage).
*Expect the question:* "Your delinquency variable has IV > 0.5. Is that leakage?" **Answer:** No. It's
past behaviour observed before the outcome month, and current delinquency is genuinely the strongest
predictor of future default in cards. But I checked that it isn't measured after the default date.

**Why monotonic bins?** If risk goes down, then up, then down as utilisation rises, the model is
fitting noise. Business logic says higher utilisation should mean more risk. Monotonic bins are
easier to defend and more stable over time.

**Scorecard points.** `Score = Offset + Factor x ln(good:bad odds)`, with `Factor = PDO / ln(2)`.
With base 600 at 50:1 and PDO 20, a 620 customer has 100:1 odds. Points per bin are additive, so a
customer can be told exactly which attribute cost them points (adverse action reasons).

**Gini, AUC, KS.** AUC = chance that a random bad gets a higher PD than a random good. Gini = 2xAUC-1.
KS = maximum gap between the cumulative distributions of goods and bads. For retail behavioural
scorecards, Gini of roughly 0.5-0.7 is often called good. *I'm not certain of an "official" benchmark;
it varies by product and bank, so say "depends on the portfolio".*

**Calibration vs discrimination.** Discrimination = ranks people correctly (AUC). Calibration = the
PD numbers are right on average (a 20% PD bucket really defaults ~20%). **IFRS 9 needs calibration,**
because the PD value itself goes into the provision. A model can rank well but be badly calibrated.

**PSI (Population Stability Index).** Compares the score distribution between two populations
(build vs today). <0.1 stable, 0.1-0.25 monitor, >0.25 investigate or redevelop.

**Hosmer-Lemeshow.** Chi-square test of predicted vs observed defaults by decile. A high p-value means
no evidence of miscalibration. Weakness: with very large samples it rejects almost everything.

---

## IFRS 9 / Ind AS 109 in plain English

* **Old rule (IAS 39):** provide only after a loss has happened ("incurred loss"). This was criticised
  after 2008 for being "too little, too late".
* **IFRS 9 (effective 2018):** provide for **expected** losses from day one, using forward-looking
  information. India's equivalent is **Ind AS 109**. *Verify which Indian institutions currently apply it;
  RBI's move of banks to an ECL framework has been discussed and phased, and it may have changed since my
  knowledge was last updated.*
* **Stage 1:** performing. Provision = **12-month** ECL.
* **Stage 2:** significant increase in credit risk (SICR) since origination. Provision = **lifetime** ECL.
  The 30 days-past-due presumption is a backstop.
* **Stage 3:** credit-impaired (typically 90+ DPD). Lifetime ECL with PD = 100%. Interest is recognised on the net amount.
* **Forward-looking scenarios:** ECL must be a probability-weighted average across scenarios, not just the base case.
  Because losses are non-linear (convex), the weighted ECL is higher than the ECL of the base case alone.

**ECL vs Basel capital.** Provisions cover *expected* loss, the average you expect to lose.
Capital covers *unexpected* loss: what you could lose in a 1-in-1000-year bad year (99.9% under the Basel
IRB formula). That's why the project computes both.

---

## Likely questions and strong answers

**Q: Why logistic regression and not XGBoost?**
A: "I tested both. The boosting challenger only added [Y] Gini. Banks have to explain each decline
to the customer and to regulators, and model-risk teams validate a simple, monotonic model faster.
The challenger is still useful as a benchmark: if the gap grows during monitoring, that's a signal to redevelop."

**Q: Why exclude SEX and MARRIAGE?**
A: "Fair-lending principles. Protected attributes shouldn't drive credit decisions. I still kept them
outside the model to *test* for disparate impact with the four-fifths rule. Excluding a variable doesn't
guarantee fairness, because other features can act as proxies, so you have to measure."

**Q: The dataset's default rate is about 22%. Is that realistic?**
A: "No. That's much higher than a typical card portfolio, and the sample is probably not representative.
In production I'd recalibrate the PDs to the bank's long-run default rate. The ranking stays the same;
only the level shifts."

**Q: How did you choose LGD = 65%?**
A: "It's an assumption, because the dataset has no recovery data. Unsecured card LGDs are high.
I made it a parameter and built a sensitivity grid. ECL scales linearly with LGD, so it's one of the
assumptions management should challenge most." *(Don't claim 65% is an industry figure.)*

**Q: What would you do with more time or data?**
A: (1) Out-of-time validation on a later cohort. (2) A proper 12-month default definition.
(3) Estimate LGD and CCF from recovery and drawdown data. (4) Link scenario shifts to real macro
variables (unemployment, GDP) with a satellite regression. (5) Relative SICR using origination PD.

**Q: What is SICR and why is it hard?**
A: A significant increase in credit risk since origination. It's hard because it's judgement-heavy.
Banks combine a quantitative trigger (lifetime PD has, say, doubled), qualitative triggers (watchlist,
forbearance) and the 30 DPD backstop. Moving accounts between Stage 1 and Stage 2 creates "cliff effects"
in the provision.

**Q: Walk me through one customer's score.**
A: Open the Scorecard tab, pick a customer and show the points for each attribute and the total.
Practise this live; it impresses interviewers.

**Q: What's the difference between PD for IFRS 9 and PD for Basel?**
A: IFRS 9 uses **point-in-time**, forward-looking PDs (reflect today's economy). Basel IRB typically
uses **through-the-cycle** PDs (long-run average) so that capital doesn't swing too much with the economy.

---

## Things to be honest about if asked
* The scenario shifts are stylised log-odds shifts, not driven by a macro model.
* Lifetime PD uses a constant-hazard assumption, which is simple.
* Synthetic data was only used for unit tests. All reported numbers come from the real UCI data.
