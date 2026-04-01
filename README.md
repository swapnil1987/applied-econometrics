# Applied Econometrics — PhD Intensive Course

**Duration:** 24 hours (8 sessions × 3 hours)  
**Format:** Lectures with derivations, worked examples, and discussion of empirical applications  
**Prerequisites:** Basic probability and statistics, matrix algebra, introductory econometrics  
**Software:** Python

---

## Course Description

This intensive course covers the core toolkit of modern applied econometrics, with emphasis on causal inference. We begin with the conceptual foundations — potential outcomes, DAGs, and the role of randomization — then build up through regression, instrumental variables, difference-in-differences, and regression discontinuity designs. Throughout, we stress both the mathematical underpinnings and the practical intuition needed to apply these methods in research.

Panel data methods will be covered by Dr Aline Stundžienė.

---

## Course Schedule

### Session 1 — Foundations of Causal Inference (3 hours)
*What does it mean to identify a causal effect, and why is it hard?*

- Potential outcomes framework (Rubin causal model)
- Selection bias and the fundamental problem of causal inference
- Directed Acyclic Graphs (DAGs): d-separation, backdoor criterion
- Randomized experiments and design-based inference
- Fisher's exact p-values, Neyman's repeated sampling

**Lecture slides:** `lectures/01_causal_foundations.pdf` *(placeholder)*

**Readings:**
- Angrist & Pischke, *Mostly Harmless Econometrics* (MHE), Ch. 1–2
- Cunningham, *Causal Inference: The Mixtape*, Ch. 4 (Potential Outcomes)
- Imbens (2020), "Potential Outcome and Directed Acyclic Graph Approaches to Causality," *Journal of Machine Learning Research*
- Holland (1986), "Statistics and Causal Inference," *JASA*

---

### Session 2 — Selection on Observables (3 hours)
*When can we control our way to a causal effect?*

- Conditional independence / unconfoundedness
- Propensity score: theory (Rosenbaum & Rubin, 1983)
- Propensity score estimation, matching, inverse probability weighting (IPW)
- Doubly robust estimation and augmented IPW
- Interference, spillovers, and SUTVA violations

**Lecture slides:** `lectures/02_selection_observables.pdf` *(placeholder)*

**Readings:**
- MHE, Ch. 3.3 (Propensity Score)
- Rosenbaum & Rubin (1983), "The Central Role of the Propensity Score in Observational Studies," *Biometrika*
- Imbens (2004), "Nonparametric Estimation of Average Treatment Effects Under Exogeneity," *REStat*
- Aronow & Samii (2017), "Estimating Average Causal Effects Under General Interference," *Annals of Applied Statistics*

---

### Session 3 — Linear Regression: What Does It Estimate? (3 hours)
*OLS as an approximation, not a structural model.*

- OLS as the best linear predictor
- Regression anatomy (Frisch–Waugh–Lovell theorem)
- What regression estimates when the CEF is not linear
- OLS with heterogeneous treatment effects: the Angrist (1998) result
- Omitted variable bias formula and sensitivity analysis
- Semiparametric regression and visualization

**Lecture slides:** `lectures/03_regression.pdf` *(placeholder)*

**Readings:**
- MHE, Ch. 3.1–3.2
- Angrist (1998), "Estimating the Labor Market Impact of Voluntary Military Service Using Social Security Data," *Econometrica*
- Aronow & Miller, *Foundations of Agnostic Statistics*, Ch. 5–6
- Goldsmith-Pinkham, Hull & Kolesár (2024), "Contamination Bias in Linear Regressions" *(working paper)*

---

### Session 4 — Inference and the Bootstrap (3 hours)
*Getting the standard errors right.*

- Heteroskedasticity-robust (EHW) standard errors: derivation and finite-sample properties
- When and how to cluster standard errors
- The bootstrap: nonparametric, parametric, wild bootstrap
- When does the bootstrap work? When does it fail?
- Small-sample corrections (CR2, CR3, Bell–McCaffrey)
- Randomization inference

**Lecture slides:** `lectures/04_inference_bootstrap.pdf` *(placeholder)*

**Readings:**
- MHE, Ch. 8 (Standard Errors)
- Cameron, Gelbach & Miller (2008), "Bootstrap-Based Improvements for Inference with Clustered Errors," *REStat*
- Cameron & Miller (2015), "A Practitioner's Guide to Cluster-Robust Inference," *JHR*
- Kolesár, Lecture Notes on Bootstrap and EHW (course materials)

---

### Session 5 — Instrumental Variables I (3 hours)
*Using exogenous variation to identify causal effects.*

- IV motivation and the exclusion restriction
- 2SLS: mechanics and derivation
- IV under constant effects vs. heterogeneous effects
- Local Average Treatment Effect (LATE) — Imbens & Angrist (1994)
- Weak instruments: detection (first-stage F, effective F) and consequences
- Anderson–Rubin confidence sets

**Lecture slides:** `lectures/05_iv_part1.pdf` *(placeholder)*

**Readings:**
- MHE, Ch. 4
- Imbens & Angrist (1994), "Identification and Estimation of Local Average Treatment Effects," *Econometrica*
- Andrews, Stock & Sun (2019), "Weak Instruments in IV Regression: Theory and Practice," *Annual Review of Economics*
- Lee, McCrary, Moreira & Porter (2022), "Valid t-ratio Inference for IV," *AER*

---

### Session 6 — Instrumental Variables II: Modern Extensions (3 hours)
*Shift-share, judge designs, and many instruments.*

- Many instruments: bias of 2SLS, JIVE, LIML
- Judge/examiner designs as IVs
- Shift-share (Bartik) instruments: identification and inference
- Borusyak, Hull & Jaravel (2022) exposure-based approach
- Goldsmith-Pinkham, Sorkin & Swift (2020) share-based approach
- Practical guidance: which IV strategy for which setting?

**Lecture slides:** `lectures/06_iv_part2.pdf` *(placeholder)*

**Readings:**
- Goldsmith-Pinkham, Sorkin & Swift (2020), "Bartik Instruments: What, When, Why, and How," *AER*
- Borusyak, Hull & Jaravel (2022), "Quasi-Experimental Shift-Share Research Designs," *REStat*
- Frandsen, Lefgren & Leslie (2023), "Judging Judge Fixed Effects," *AER*
- Kolesár, Lecture Notes on Many IV and Shift-Share IV (course materials)

---

### Session 7 — Difference-in-Differences and Synthetic Control (3 hours)
*Exploiting policy variation over time and across units.*

- Classic 2×2 DiD: identification, parallel trends
- DiD with staggered treatment: problems with TWFE
- Modern DiD estimators: Callaway & Sant'Anna, Sun & Abraham, de Chaisemartin & D'Haultfoeuille
- Event study designs and pre-trends testing
- Synthetic control method (Abadie, Diamond & Hainmueller)
- Synthetic DiD (Arkhangelsky et al., 2021)
- Changes-in-changes (Athey & Imbens, 2006)

**Lecture slides:** `lectures/07_did_synth.pdf` *(placeholder)*

**Readings:**
- MHE, Ch. 5
- Roth, Sant'Anna, Bilinski & Poe (2023), "What's Trending in Difference-in-Differences?," *JoE*
- Callaway & Sant'Anna (2021), "Difference-in-Differences with Multiple Time Periods," *JoE*
- Abadie (2021), "Using Synthetic Controls: Feasibility, Data Requirements, and Methodological Aspects," *JEL*
- Arkhangelsky, Athey, Hirshberg, Imbens & Wager (2021), "Synthetic Difference-in-Differences," *AER*

---

### Session 8 — Regression Discontinuity and Machine Learning for Causal Inference (3 hours)
*Sharp cutoffs and data-driven methods.*

**Part A: Regression Discontinuity (2 hours)**
- Sharp RD identification and graphical analysis
- Local polynomial estimation and bandwidth selection
- Bias-aware inference (Armstrong & Kolesár, 2020)
- Fuzzy RD as local IV
- RD extensions: kink designs, geographic RD
- Validation and manipulation testing (McCrary, Cattaneo–Jansson–Ma)

**Part B: Machine Learning for Causal Inference (1 hour)**
- Prediction vs. causal inference: when ML helps
- Double/debiased machine learning (Chernozhukov et al., 2018)
- Heterogeneous treatment effects: causal forests (Wager & Athey)
- Brief overview of partial identification

**Lecture slides:** `lectures/08_rd_ml.pdf` *(placeholder)*

**Readings:**
- Cattaneo, Idrobo & Titiunik (2020), *A Practical Introduction to Regression Discontinuity Designs*
- Lee & Lemieux (2010), "Regression Discontinuity Designs in Economics," *JEL*
- Armstrong & Kolesár (2020), "Simple and Honest Confidence Intervals in Nonparametric Regression," *QE*
- Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey & Robins (2018), "Double/Debiased Machine Learning for Treatment and Structural Parameters," *Econometrics Journal*
- Athey & Imbens (2019), "Machine Learning Methods That Economists Should Know About," *Annual Review of Economics*

---

## Main Textbooks

1. Angrist, J.D. & Pischke, J.-S. (2009). *Mostly Harmless Econometrics: An Empiricist's Companion*. Princeton University Press.
2. Cunningham, S. (2021). *Causal Inference: The Mixtape*. Yale University Press. [Free online: https://mixtape.scunning.com/]
3. Aronow, P.M. & Miller, B.T. (2019). *Foundations of Agnostic Statistics*. Cambridge University Press.
4. Cattaneo, M.D., Idrobo, N. & Titiunik, R. (2020). *A Practical Introduction to Regression Discontinuity Designs*. Cambridge Elements.

## Supplementary References

- Hansen, B.E. (2022). *Econometrics*. Princeton University Press. [Free online: https://www.ssc.wisc.edu/~bhansen/econometrics/]
- Imbens, G.W. & Rubin, D.B. (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences*. Cambridge University Press.
- Healy, K. (2019). *Data Visualization: A Practical Introduction*. Princeton University Press. [Free online: https://socviz.co/]

---

## Assessment

| Component | Weight |
|---|---|
| Problem sets (4 sets) | 60% |
| Final take-home exam | 40% |

Problem sets will involve both analytical derivations and empirical exercises using real data.

---

## Source Material

This course draws on and synthesizes material from:

- **PG:** Paul Goldsmith-Pinkham, *Applied Empirical Methods* (Yale MGMT 737)
- **MK:** Michal Kolesár, *Econ 539b* (Princeton)
