# Application Fraud Detection & Fairness Audit

*"Deteksi Application Fraud pada Digital Onboarding dan Audit Keadilan (Fairness) Menggunakan HistGradientBoosting"* — PPKD Jakarta Selatan Data Analyst final project (Project-Based Learning), analyzing the NeurIPS 2022 Bank Account Fraud (BAF) dataset.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/status-completed-2ea44f?style=flat-square)

<p align="center">
  <img src="screenshots/streamlit-app.png" alt="Fraud detection Streamlit app" width="850">
</p>

<p align="center">
  <a href="#live-demo"><strong>Live Demo →</strong></a> · <a href="eda-notebook-export.pdf"><strong>Full EDA/Modeling Walkthrough (PDF) →</strong></a>
</p>

## Contents
- [Business Problem](#business-problem)
- [Data](#data)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [Deployment](#deployment)
- [Recommendation](#recommendation)
- [Live Demo](#live-demo)
- [Limitations](#limitations)

## Business Problem
Digital account-opening (onboarding) fraud is a major and growing risk for banks and e-KYC systems. A detection model needs to catch fraud attempts without being either (a) too permissive, missing fraud, or (b) unfairly biased against legitimate applicants from any particular age group.

## Data
- **Source:** Bank Account Fraud (BAF) dataset suite, published by Feedzai and presented at NeurIPS 2022 — built with CTGAN synthetic-data generation plus differential privacy, and explicitly includes fairness/bias-relevant fields (age group, employment status)
- **Scale:** 1,000,000 rows × 32 columns 
- **Class imbalance:** only **1.10%** of applications are fraudulent (11,029 cases) — a realistic, highly imbalanced fraud-detection scenario

## Methodology
- Full EDA: distribution checks, skewness/kurtosis, Shapiro-Wilk and Kolmogorov-Smirnov normality testing, IQR/Z-score outlier detection
- Sentinel value `-1` (used by the dataset creators to mean "no prior history") was **deliberately retained rather than imputed**, since "no history" is itself a fraud-risk signal
- Outliers were **not removed** — extreme transaction velocity is itself a fraud signal (same principle as the PaySim fraud literature) — instead, a log-transform was applied to normalize `velocity_6h`
- **Strict temporal train/test split** (train: months 0–5, test: months 6–7) to simulate real production conditions and avoid data leakage, with one-hot encoding fit only on the training set
- Final model: **HistGradientBoostingClassifier** (`class_weight='balanced'`, `max_iter=200`), trained in 42.5 seconds

### Feature engineering
Beyond the raw 32 fields, 5 additional binary "has-history" flags were engineered from the sentinel-value columns (e.g. `prev_address_months_count_ada_riwayat`) — turning "no data" into an explicit, model-usable signal rather than letting -1 be treated as a literal numeric value.

## Key Findings

| Metric | Value |
|---|---|
| Recall @ 5% FPR | **53.5%** |
| AUC | **0.89** |
| Fairness gap (60+ vs. 20-year-olds FPR) | **8.7x** |
| Training time | 42.5 seconds |

- **Final model: 53.5% Recall at 5% False Positive Rate, 0.89 AUC** — roughly double a linear baseline.
- **Fairness audit (Predictive Equality):** false-positive rate (legitimate applicants wrongly flagged as fraud) was compared across age — applicants aged 60+ were wrongly flagged at **8.7x the rate** of 20-year-old applicants, well above the model's intended 5% FPR ceiling for that group. This is a critical finding for production deployment: an elderly applicant is far more likely to be wrongly rejected than a young one, for no fraud-related reason.
- **Counter-intuitive insight:** legitimate applicants filled out the registration form *faster* than fraudulent ones (higher `velocity_6h`) — the opposite of the naive assumption that fast form-filling signals bot/fraud activity.
- **Temporal drift detected:** the fraud rate is not stable across the 8-month window — it dips around month 2, then climbs noticeably toward month 7, suggesting an evolving fraud pattern that a static, one-time-trained model would miss.

### Deeper EDA highlights
- All 6 core numeric features (income, age, credit risk score, velocity_6h, bank tenure, session length) **failed Shapiro-Wilk and Kolmogorov-Smirnov normality tests** — none are normally distributed, which ruled out modeling approaches that assume normality.
- `session_length_in_minutes` is extremely right-skewed (skewness 3.30, kurtosis ~15 — a "heavy-tailed" distribution), while most other features are close to symmetric.
- Linear correlations between individual features and fraud are all weak in isolation (Pearson r: credit_risk_score 0.071, customer_age 0.063, income 0.045, velocity_6h −0.017) — confirming that fraud in this dataset is driven by **feature interactions**, not any single strong linear predictor, which is exactly why a tree-based ensemble (HistGradientBoosting) outperforms simpler linear models.
- Housing-status category "BA" had a fraud rate of 3.75% — roughly 3–6x higher than every other housing category (all ≤0.86%).

*Sourcing note: the full evaluation code (ROC/AUC, recall-at-5%-FPR thresholding, confusion matrix, and the fairness-audit calculation above) has been reviewed line-by-line and confirmed to compute exactly what's described here. The specific printed output values match the project's originally reported results.*

## Deployment
6 trained artifacts were exported for production use: the HistGradientBoosting model, the fitted `StandardScaler`, the chosen decision threshold, the final feature list, the scaled-column list, and a median-value template for handling missing input fields — then wired into the Streamlit app shown above.

## Recommendation
Deploy with active fairness monitoring, not just accuracy monitoring — the 8.7x FPR gap needs mitigation before production use. Because fraud patterns drift over time, the model should be retrained/monitored on a rolling basis rather than treated as a one-time deployment.

## Live Demo
Deployed as an interactive **Streamlit** app — enter applicant details (age, credit bureau risk score, income decile, form-fill velocity, housing/employment status codes, device OS, registration channel) and get a live fraud-probability score and approve/reject decision.

![Streamlit].(https://pbl-datascience-baf2022-8lmr9vkarc7mfckfaea8hy.streamlit.app/). — see the deployment instructions I sent alongside this file for how to get this link and drop it in.

## Limitations
The BAF dataset is synthetically generated (CTGAN + differential privacy), so absolute performance numbers may shift on real transaction data. The fairness audit here checks one protected attribute (age) in isolation; a production fairness review should also test intersectional effects (e.g. age combined with employment status) before deployment.

---

<sub>**Muhammad Yahya Ayyasy** — [LinkedIn](https://linkedin.com/in/muhammadayyass) · [muhammadayyas22@gmail.com](mailto:muhammadayyas22@gmail.com)</sub>
