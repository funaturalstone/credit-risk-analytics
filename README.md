# Credit Risk Analytics — Lending Club Loan Portfolio

End-to-end credit scoring pipeline framed as a profit-optimization problem. Built using XGBoost and Logistic Regression on Lending Club loan data, this project demonstrates how machine learning translates directly into business decisions by tuning classification thresholds to maximize Expected Value per loan rather than pure accuracy.

## Overview

The project balances the trade-off between approving bad loans (loss) and rejecting good ones (opportunity cost). Instead of optimizing accuracy or AUC alone, the model maximizes **Expected Value (EV) per loan** based on actual business parameters derived from historical payment data.

## Key Results

| Metric | Value |
|---|---|
| Dataset | Lending Club — 80,000 loans (19.8% default rate) |
| Train / Val / Test Split | 60% / 20% / 20% |
| Best Model | XGBoost |
| Test AUC | 0.71 |
| Optimal Threshold | 0.20 |
| Expected Value per Loan | $2,540 |
| Improvement vs FICO Baseline | +222% (3.2x) |

## Tech Stack

- **Language:** Python 3.10+
- **Core Libraries:** Pandas, NumPy, Scikit-learn, XGBoost
- **Explainability:** SHAP
- **Visualization:** Matplotlib, Power BI

## Methodology

1. **Data Preparation** — Filter loan status (Fully Paid vs Charged Off), handle missing values with median imputation, create missing-flag features
2. **Business Parameters** — Derive actual profit/loss values from historical payment data (PROFIT_TN, LOSS_FN, COST_FP, REWARD_TP)
3. **Train / Val / Test Split** — Stratified 60/20/20 split to preserve default rate across sets
4. **Preprocessing** — StandardScaler for numerics, OneHotEncoder for categoricals, Ordinal encoding for term and employment length
5. **Model Training** — Logistic Regression (baseline) and XGBoost (with scale_pos_weight for class imbalance)
6. **Threshold Tuning** — Sweep 200 candidate thresholds on validation set, pick the one that maximizes EV
7. **Baseline Comparison** — Approve All, Reject All, FICO-based rule
8. **Final Evaluation** — One-time evaluation on held-out test set
9. **Interpretation** — SHAP analysis for feature importance and individual prediction explanations
10. **Dashboard** — Power BI dashboard summarizing model performance, EV curves, and feature importance for stakeholder review

## Project Structure

```
credit-risk-analytics/
├── notebooks/
│   └── credit_risk_pipeline.ipynb    # Full pipeline with outputs
├── src/
│   └── credit_risk_pipeline.py       # Production-ready Python script
├── docs/                              # Documentation
├── requirements.txt                   # Python dependencies
└── README.md
```

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download the Lending Club dataset (80k-row version) and place at the path specified in the notebook.

3. Run the notebook or script:
   ```bash
   jupyter notebook notebooks/credit_risk_pipeline.ipynb
   # or
   python src/credit_risk_pipeline.py
   ```

## Key Insights

- **XGBoost outperformed simple FICO-based rules by 222%** in expected profitability, demonstrating clear business value of ML-driven credit decisions
- **Optimal threshold (0.20) is significantly lower than default (0.50)** because false negatives (funding bad loans) cost ~3.8x more than the profit from true negatives
- **Top default drivers** identified via SHAP include interest rate, debt-to-income ratio, and FICO score range

## Author

**Tanakit Tuadpitak**
Bachelor of Science in Digital Transformation and Innovation
Thammasat University, Thailand
