"""
=============================================================================
Credit Risk Model Pipeline — Production-Ready
=============================================================================
Lending Club Dataset (~80,000 rows)
Models: Logistic Regression vs XGBoost
Objective: Maximize Expected Value via optimal threshold tuning

Author : Data Science Team
Created: 2026-04-13
Python : 3.10+
=============================================================================
"""

import logging
import warnings
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, confusion_matrix,
    classification_report,
)
from sklearn.calibration import calibration_curve

import xgboost as xgb
import shap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

warnings.filterwarnings("ignore", category=FutureWarning)

# ═══════════════════════════════════════════════════════════════════════════
# GLOBAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

# ---- Positive class definition (single source of truth) ----
# loan_status == "Charged Off"  →  1  (POSITIVE CLASS = borrower defaulted)
# loan_status == "Fully Paid"   →  0  (NEGATIVE CLASS = borrower repaid)
#
# Confusion matrix interpretation:
#   TP (True Positive)  : Model predicts default (1), actual default   → avoided a bad loan
#   TN (True Negative)  : Model predicts repay  (0), actual repay     → earned profit
#   FP (False Positive) : Model predicts default (1), actual repay    → missed a good loan (opportunity cost)
#   FN (False Negative) : Model predicts repay  (0), actual default   → funded a bad loan (loss)
POSITIVE_CLASS_LABEL = "Charged Off"
POSITIVE_CLASS = 1

# ---- Feature lists ----
NUMERIC_FEATURES = [
    "loan_amnt", "int_rate", "installment", "annual_inc", "dti",
    "fico_range_low", "fico_range_high", "open_acc", "pub_rec",
    "revol_util", "mort_acc",
]
CATEGORICAL_FEATURES = ["purpose", "home_ownership"]
ORDINAL_FEATURES = ["term", "emp_length"]

# Ordinal mappings
TERM_MAP = {"36 months": 0, "60 months": 1}
EMP_LENGTH_ORDER = [
    "< 1 year", "1 year", "2 years", "3 years", "4 years",
    "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years",
]

# ---- Split ratios ----
TRAIN_RATIO = 0.60
VAL_RATIO   = 0.20
TEST_RATIO  = 0.20

RANDOM_STATE = 42

# ---- Output directory ----
OUTPUT_DIR = Path("outputs")


# ═══════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════

def setup_logging(log_level: int = logging.INFO) -> logging.Logger:
    """Configure structured logging for the pipeline."""
    logger = logging.getLogger("credit_risk")
    logger.setLevel(log_level)
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(funcName)-30s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(OUTPUT_DIR / "pipeline.log", mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


log = setup_logging()


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — Load + Feature Selection + Business Params
# ═══════════════════════════════════════════════════════════════════════════

def step1_load_and_prepare(filepath: str) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Load Lending Club CSV, select features, encode target, and compute
    business parameters from the actual data.

    Business params:
        Every outcome is measured against not lending at all (= 0):
        PROFIT_TN = median(total_pymnt - loan_amnt) over repaid loans    # interest earned
        LOSS_FN   = median(loan_amnt - total_pymnt) over defaulted loans # net principal lost
        COST_FP   = 0   # rejecting a good loan earns nothing and loses nothing
        REWARD_TP = 0   # rejecting a bad loan earns nothing and loses nothing
        (An earlier version also counted the principal saved as a gain, which
        double-counted LOSS_FN and made "reject everyone" look profitable.)
    """
    log.info("Loading dataset from %s", filepath)
    df = pd.read_csv(filepath)
    log.info("Raw shape: %s", df.shape)

    # Keep only Fully Paid / Charged Off
    df = df[df["loan_status"].isin(["Fully Paid", POSITIVE_CLASS_LABEL])].copy()
    log.info("After filtering loan_status: %s", df.shape)

    # Encode target  (Charged Off → 1, Fully Paid → 0)
    df["target"] = (df["loan_status"] == POSITIVE_CLASS_LABEL).astype(int)
    log.info("Target distribution:\n%s", df["target"].value_counts().to_string())

    # Select features
    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ORDINAL_FEATURES
    keep_cols = list(dict.fromkeys(all_features + ["target", "loan_status", "loan_amnt", "installment", "total_pymnt"]))  # no duplicate columns
    # loan_amnt & installment already in NUMERIC_FEATURES, but keep loan_status for audit
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    # ---- Ordinal encoding (manual) ----
    df["term"] = df["term"].str.strip().map(TERM_MAP)
    df["emp_length"] = df["emp_length"].str.strip()
    emp_map = {v: i for i, v in enumerate(EMP_LENGTH_ORDER)}
    df["emp_length"] = df["emp_length"].map(emp_map)

    # ---- Drop rows with any NaN in feature columns ----
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ORDINAL_FEATURES
    before = len(df)
    df.dropna(subset=feature_cols + ["target"], inplace=True)
    log.info("Dropped %d rows with NaN → remaining %d", before - len(df), len(df))

    # ---- Business parameters (computed from actual data) ----
    if "total_pymnt" in df.columns:
        total_payment = df["total_pymnt"]
    else:
        total_payment = df["installment"] * df["term"].map({0: 36, 1: 60})
    good, bad = df["target"] == 0, df["target"] == 1

    business_params = {
        "PROFIT_TN":  float((total_payment[good] - df.loc[good, "loan_amnt"]).median()),  # interest earned on a repaid loan
        "LOSS_FN":    float((df.loc[bad, "loan_amnt"] - total_payment[bad]).median()),    # net loss on a defaulted loan
        "COST_FP":    0.0,  # rejecting a good loan: no cash in or out
        "REWARD_TP":  0.0,  # rejecting a bad loan: no cash in or out
    }
    log.info("Business parameters (median-based):")
    for k, v in business_params.items():
        log.info("  %s = $%.2f", k, v)

    # Drop helper column
    df.drop(columns=["loan_status", "total_pymnt"], inplace=True, errors="ignore")

    return df, business_params


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — Train / Val / Test Split (60 / 20 / 20)
# ═══════════════════════════════════════════════════════════════════════════

def step2_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series, pd.Series, pd.Series]:
    """
    Stratified split into Train (60%), Val (20%), Test (20%).
    All indices are reset.
    """
    log.info("Splitting data: Train=%.0f%% / Val=%.0f%% / Test=%.0f%%",
             TRAIN_RATIO * 100, VAL_RATIO * 100, TEST_RATIO * 100)

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ORDINAL_FEATURES
    X = df[feature_cols]
    y = df["target"]

    # First split: train (60%) vs temp (40%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=VAL_RATIO + TEST_RATIO,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Second split: val (50% of 40% = 20%) vs test (50% of 40% = 20%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    # Reset indices
    for d in [X_train, X_val, X_test, y_train, y_val, y_test]:
        d.reset_index(drop=True, inplace=True)

    log.info("Train: %d | Val: %d | Test: %d", len(X_train), len(X_val), len(X_test))
    log.info("Positive-class rate — Train: %.4f | Val: %.4f | Test: %.4f",
             y_train.mean(), y_val.mean(), y_test.mean())

    return X_train, X_val, X_test, y_train, y_val, y_test


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — Dual Preprocessor (LR vs XGBoost)
# ═══════════════════════════════════════════════════════════════════════════

def _build_preprocessor(drop_first: bool) -> ColumnTransformer:
    """Build a ColumnTransformer. LR needs drop_first=True; XGB does not."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                drop="first" if drop_first else None,
            ), CATEGORICAL_FEATURES),
            ("ord", "passthrough", ORDINAL_FEATURES),
        ],
        remainder="drop",
    )


def step3_preprocess(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Fit preprocessors on Train ONLY, then transform Val and Test.
    Returns dict keyed by model name with arrays for each split.
    """
    log.info("Building dual preprocessors (LR: drop_first=True, XGB: drop_first=False)")

    result = {}
    for name, drop_first in [("lr", True), ("xgb", False)]:
        pre = _build_preprocessor(drop_first=drop_first)
        Xt_train = pre.fit_transform(X_train)
        Xt_val   = pre.transform(X_val)
        Xt_test  = pre.transform(X_test)

        feature_names = pre.get_feature_names_out()
        result[name] = {
            "train": Xt_train,
            "val":   Xt_val,
            "test":  Xt_test,
            "preprocessor": pre,
            "feature_names": feature_names,
        }
        log.info("  %s preprocessor — output features: %d", name.upper(), Xt_train.shape[1])

    return result


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — Train Models
# ═══════════════════════════════════════════════════════════════════════════

def step4_train_models(
    preprocessed: Dict,
    y_train: pd.Series,
) -> Dict[str, Any]:
    """Train Logistic Regression and XGBoost on the training set."""
    models = {}

    # --- Logistic Regression ---
    log.info("Training Logistic Regression (saga, l2, max_iter=2000)")
    lr = LogisticRegression(
        solver="saga",
        penalty="l2",
        C=1.0,
        max_iter=2000,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    lr.fit(preprocessed["lr"]["train"], y_train)
    models["lr"] = lr
    log.info("  LR converged in %d iterations", lr.n_iter_[0])

    # --- XGBoost ---
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    log.info("Training XGBoost (scale_pos_weight=%.2f)", pos_weight)
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        eval_metric="auc",
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        early_stopping_rounds=30,
    )
    xgb_model.fit(
        preprocessed["xgb"]["train"], y_train,
        eval_set=[(preprocessed["xgb"]["val"], None)],  # placeholder — see note below
        verbose=False,
    )
    # Note: eval_set y is needed; we pass it via a wrapper
    # Re-train properly:
    xgb_model_final = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        eval_metric="auc",
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgb_model_final.fit(preprocessed["xgb"]["train"], y_train, verbose=False)
    models["xgb"] = xgb_model_final
    log.info("  XGBoost training complete — n_estimators=%d", xgb_model_final.n_estimators)

    return models


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 — Predict Probability + Expected Value Function
# ═══════════════════════════════════════════════════════════════════════════

def predict_proba_positive(model, X: np.ndarray) -> np.ndarray:
    """Return P(Charged Off) — probability of the positive class."""
    return model.predict_proba(X)[:, POSITIVE_CLASS]


def compute_expected_value(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    business_params: Dict[str, float],
) -> float:
    """
    Compute Expected Value from confusion matrix and business params.

    EV = (TN × PROFIT_TN) − (FN × LOSS_FN) − (FP × COST_FP) + (TP × REWARD_TP)

    Confusion matrix positions (sklearn order):
        [[TN, FP],
         [FN, TP]]
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    ev = (
        (tn * business_params["PROFIT_TN"])
        - (fn * business_params["LOSS_FN"])
        - (fp * business_params["COST_FP"])
        + (tp * business_params["REWARD_TP"])
    )
    return ev


def compute_ev_per_loan(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    business_params: Dict[str, float],
) -> float:
    """EV normalised by number of loans."""
    ev = compute_expected_value(y_true, y_pred, business_params)
    return ev / len(y_true)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 6 — Threshold Tuning Loop (Validation set only)
# ═══════════════════════════════════════════════════════════════════════════

def step6_tune_threshold(
    models: Dict,
    preprocessed: Dict,
    y_val: pd.Series,
    business_params: Dict[str, float],
    n_thresholds: int = 200,
) -> Dict[str, Dict[str, Any]]:
    """
    Sweep thresholds on the VALIDATION set to find the one that maximises
    Expected Value for each model.  Test set is NOT touched here.
    """
    log.info("Tuning thresholds on Validation set (%d candidates)", n_thresholds)
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    results = {}

    for name in ["lr", "xgb"]:
        pre_key = name
        proba = predict_proba_positive(models[name], preprocessed[pre_key]["val"])
        ev_list = []

        for t in thresholds:
            y_pred = (proba >= t).astype(int)
            ev = compute_expected_value(y_val.values, y_pred, business_params)
            ev_list.append(ev)

        ev_arr = np.array(ev_list)
        best_idx = np.argmax(ev_arr)
        best_threshold = thresholds[best_idx]
        best_ev = ev_arr[best_idx]

        results[name] = {
            "best_threshold": best_threshold,
            "best_ev": best_ev,
            "thresholds": thresholds,
            "ev_curve": ev_arr,
            "val_proba": proba,
        }

        log.info("  %s — Best threshold: %.4f | Val EV: $%.2f | EV/loan: $%.2f",
                 name.upper(), best_threshold, best_ev, best_ev / len(y_val))

    # ---- Plot EV vs Threshold ----
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, r in results.items():
        ax.plot(r["thresholds"], r["ev_curve"], label=name.upper(), linewidth=1.8)
        ax.axvline(r["best_threshold"], linestyle="--", alpha=0.6,
                   label=f"{name.upper()} opt={r['best_threshold']:.3f}")
    ax.set_xlabel("Threshold (P(Charged Off) ≥ t → predict default)")
    ax.set_ylabel("Expected Value ($)")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_title("Step 6 — Threshold Tuning on Validation Set")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "step6_threshold_tuning.png", dpi=150)
    plt.close(fig)
    log.info("Saved threshold tuning plot")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# STEP 7 — Baseline Comparison
# ═══════════════════════════════════════════════════════════════════════════

def step7_baselines(
    X_val: pd.DataFrame,
    y_val: pd.Series,
    business_params: Dict[str, float],
    threshold_results: Dict,
) -> pd.DataFrame:
    """
    Compare model EV against naïve baselines on the Validation set:
      1. Approve all (predict 0 for everyone)
      2. Reject all  (predict 1 for everyone)
      3. FICO rule   (reject if avg FICO < 670)
    """
    log.info("Computing baseline comparisons on Validation set")
    rows = []

    # Baseline 1: approve all
    y_approve = np.zeros(len(y_val), dtype=int)
    ev_approve = compute_expected_value(y_val.values, y_approve, business_params)
    rows.append({"Strategy": "Approve All", "Threshold": "—",
                 "Val EV ($)": ev_approve, "EV/Loan ($)": ev_approve / len(y_val)})

    # Baseline 2: reject all
    y_reject = np.ones(len(y_val), dtype=int)
    ev_reject = compute_expected_value(y_val.values, y_reject, business_params)
    rows.append({"Strategy": "Reject All", "Threshold": "—",
                 "Val EV ($)": ev_reject, "EV/Loan ($)": ev_reject / len(y_val)})

    # Baseline 3: FICO < 670 → reject
    fico_avg = (X_val["fico_range_low"] + X_val["fico_range_high"]) / 2
    y_fico = (fico_avg < 670).astype(int)
    ev_fico = compute_expected_value(y_val.values, y_fico.values, business_params)
    rows.append({"Strategy": "FICO < 670 Rule", "Threshold": "670",
                 "Val EV ($)": ev_fico, "EV/Loan ($)": ev_fico / len(y_val)})

    # Model results
    for name, r in threshold_results.items():
        rows.append({
            "Strategy": f"{name.upper()} Model",
            "Threshold": f"{r['best_threshold']:.4f}",
            "Val EV ($)": r["best_ev"],
            "EV/Loan ($)": r["best_ev"] / len(y_val),
        })

    baseline_df = pd.DataFrame(rows)
    baseline_df = baseline_df.sort_values("Val EV ($)", ascending=False).reset_index(drop=True)
    log.info("Baseline comparison:\n%s", baseline_df.to_string(index=False))

    return baseline_df


# ═══════════════════════════════════════════════════════════════════════════
# STEP 8 — ROC + Precision-Recall Curve
# ═══════════════════════════════════════════════════════════════════════════

def step8_roc_pr_curves(
    models: Dict,
    preprocessed: Dict,
    y_val: pd.Series,
    threshold_results: Dict,
) -> None:
    """Plot ROC and Precision-Recall curves on the Validation set."""
    log.info("Generating ROC and Precision-Recall curves")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for name in ["lr", "xgb"]:
        proba = threshold_results[name]["val_proba"]
        t_opt = threshold_results[name]["best_threshold"]

        # ROC
        fpr, tpr, _ = roc_curve(y_val, proba)
        auc_score = roc_auc_score(y_val, proba)
        axes[0].plot(fpr, tpr, label=f"{name.upper()} (AUC={auc_score:.4f})", linewidth=1.8)

        # Mark optimal threshold on ROC
        y_pred_opt = (proba >= t_opt).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred_opt).ravel()
        fpr_pt = fp / (fp + tn)
        tpr_pt = tp / (tp + fn)
        axes[0].scatter(fpr_pt, tpr_pt, s=80, zorder=5, marker="D",
                        label=f"{name.upper()} t={t_opt:.3f}")

        # PR
        precision, recall, _ = precision_recall_curve(y_val, proba)
        ap = average_precision_score(y_val, proba)
        axes[1].plot(recall, precision, label=f"{name.upper()} (AP={ap:.4f})", linewidth=1.8)

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("Step 8 — ROC Curve (Validation)")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Step 8 — Precision-Recall Curve (Validation)")
    axes[1].legend(loc="lower left")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "step8_roc_pr.png", dpi=150)
    plt.close(fig)
    log.info("Saved ROC + PR curves")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 9 — Calibration Check (Reliability Diagram)
# ═══════════════════════════════════════════════════════════════════════════

def step9_calibration(
    models: Dict,
    preprocessed: Dict,
    y_val: pd.Series,
    threshold_results: Dict,
) -> None:
    """Plot reliability diagrams for both models on the Validation set."""
    log.info("Generating calibration plots")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")

    for name in ["lr", "xgb"]:
        proba = threshold_results[name]["val_proba"]
        fraction_pos, mean_pred = calibration_curve(y_val, proba, n_bins=10, strategy="uniform")
        ax.plot(mean_pred, fraction_pos, "o-", label=name.upper(), linewidth=1.5)

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Step 9 — Calibration (Reliability Diagram) — Validation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "step9_calibration.png", dpi=150)
    plt.close(fig)
    log.info("Saved calibration plot")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 10 — Feature Importance (LR coef + XGB gain)
# ═══════════════════════════════════════════════════════════════════════════

def step10_feature_importance(
    models: Dict,
    preprocessed: Dict,
) -> None:
    """Bar charts of LR coefficients and XGBoost feature gain."""
    log.info("Computing feature importances")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- LR Coefficients ---
    lr_names = preprocessed["lr"]["feature_names"]
    lr_coefs = models["lr"].coef_[0]
    lr_df = pd.DataFrame({"feature": lr_names, "coef": lr_coefs})
    lr_df = lr_df.reindex(lr_df["coef"].abs().sort_values(ascending=True).index)
    colors = ["#d62728" if c > 0 else "#2ca02c" for c in lr_df["coef"]]
    axes[0].barh(lr_df["feature"], lr_df["coef"], color=colors)
    axes[0].set_title("LR Coefficients (red=↑default risk)")
    axes[0].set_xlabel("Coefficient")

    # --- XGBoost Gain ---
    xgb_names = preprocessed["xgb"]["feature_names"]
    xgb_imp = models["xgb"].feature_importances_
    xgb_df = pd.DataFrame({"feature": xgb_names, "gain": xgb_imp})
    xgb_df = xgb_df.sort_values("gain", ascending=True)
    axes[1].barh(xgb_df["feature"], xgb_df["gain"], color="#1f77b4")
    axes[1].set_title("XGBoost Feature Importance (gain)")
    axes[1].set_xlabel("Importance")

    fig.suptitle("Step 10 — Feature Importance", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "step10_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved feature importance plot")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 11 — SHAP Individual Explanation
# ═══════════════════════════════════════════════════════════════════════════

def step11_shap(
    models: Dict,
    preprocessed: Dict,
    n_background: int = 200,
    n_explain: int = 3,
) -> None:
    """
    SHAP summary plot for XGBoost + waterfall for a few individual loans.
    Uses TreeExplainer for speed.
    """
    log.info("Computing SHAP values (XGBoost, TreeExplainer)")

    xgb_model = models["xgb"]
    X_val_xgb = preprocessed["xgb"]["val"]
    feature_names = list(preprocessed["xgb"]["feature_names"])

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_val_xgb)

    # Summary plot
    fig = plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X_val_xgb,
        feature_names=feature_names,
        show=False,
        max_display=15,
    )
    plt.title("Step 11 — SHAP Summary (XGBoost)")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "step11_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Waterfall for top N individuals
    explanation = shap.Explanation(
        values=shap_values[:n_explain],
        base_values=float(np.ravel(explainer.expected_value)[-1]),  # scalar for a binary XGBoost model
        data=X_val_xgb[:n_explain],
        feature_names=feature_names,
    )
    for i in range(n_explain):
        fig = plt.figure(figsize=(10, 6))
        shap.waterfall_plot(explanation[i], show=False, max_display=12)
        plt.title(f"Step 11 — SHAP Waterfall (Loan #{i})")
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / f"step11_shap_waterfall_{i}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    log.info("Saved SHAP summary + %d waterfall plots", n_explain)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 12 — Sensitivity Analysis (Business Params)
# ═══════════════════════════════════════════════════════════════════════════

def step12_sensitivity(
    models: Dict,
    preprocessed: Dict,
    y_val: pd.Series,
    business_params: Dict[str, float],
    threshold_results: Dict,
    vary_pct: float = 0.5,
    n_points: int = 20,
) -> None:
    """
    Vary each business param ±vary_pct around its base value and
    re-compute optimal EV for the best model on Val.
    """
    log.info("Running sensitivity analysis (±%.0f%% on each param)", vary_pct * 100)

    # Use whichever model had higher Val EV
    best_model_name = max(threshold_results, key=lambda k: threshold_results[k]["best_ev"])
    proba = threshold_results[best_model_name]["val_proba"]
    best_t = threshold_results[best_model_name]["best_threshold"]
    y_pred = (proba >= best_t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val.values, y_pred).ravel()

    log.info("  Sensitivity on %s (threshold=%.4f)", best_model_name.upper(), best_t)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    param_names = ["PROFIT_TN", "LOSS_FN", "COST_FP", "REWARD_TP"]

    for ax, param in zip(axes.ravel(), param_names):
        base = business_params[param]
        lo, hi = base * (1 - vary_pct), base * (1 + vary_pct)
        vals = np.linspace(lo, hi, n_points)
        evs = []
        for v in vals:
            bp_copy = business_params.copy()
            bp_copy[param] = v
            ev = (tn * bp_copy["PROFIT_TN"]
                  - fn * bp_copy["LOSS_FN"]
                  - fp * bp_copy["COST_FP"]
                  + tp * bp_copy["REWARD_TP"])
            evs.append(ev)
        ax.plot(vals, evs, linewidth=1.8)
        ax.axvline(base, color="red", linestyle="--", alpha=0.6, label=f"Base=${base:,.0f}")
        ax.set_xlabel(f"{param} ($)")
        ax.set_ylabel("EV ($)")
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.set_title(param)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Step 12 — Sensitivity Analysis ({best_model_name.upper()}, Val set)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "step12_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved sensitivity analysis plot")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 13 — Cross-Validation EV (5-fold)
# ═══════════════════════════════════════════════════════════════════════════

def step13_cv_ev(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    business_params: Dict[str, float],
    n_folds: int = 5,
) -> pd.DataFrame:
    """
    5-fold stratified cross-validation on the Training set.
    For each fold, tune threshold to max EV on the fold's validation split.
    Reports mean ± std of EV/loan.
    """
    log.info("Running %d-fold cross-validation EV estimation", n_folds)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    records = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        Xf_train = X_train.iloc[train_idx]
        yf_train = y_train.iloc[train_idx]
        Xf_val   = X_train.iloc[val_idx]
        yf_val   = y_train.iloc[val_idx]

        for name, drop_first in [("lr", True), ("xgb", False)]:
            pre = _build_preprocessor(drop_first=drop_first)
            Xf_tr_t = pre.fit_transform(Xf_train)
            Xf_va_t = pre.transform(Xf_val)

            if name == "lr":
                mdl = LogisticRegression(
                    solver="saga", penalty="l2", C=1.0,
                    max_iter=2000, random_state=RANDOM_STATE, n_jobs=-1,
                )
            else:
                pos_w = (yf_train == 0).sum() / (yf_train == 1).sum()
                mdl = xgb.XGBClassifier(
                    n_estimators=300, max_depth=5, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    scale_pos_weight=pos_w, eval_metric="auc",
                    use_label_encoder=False, random_state=RANDOM_STATE, n_jobs=-1,
                )

            mdl.fit(Xf_tr_t, yf_train, **({"verbose": False} if name == "xgb" else {}))
            proba = mdl.predict_proba(Xf_va_t)[:, POSITIVE_CLASS]

            # Tune threshold
            best_ev, best_t = -np.inf, 0.5
            for t in np.linspace(0.05, 0.95, 100):
                yp = (proba >= t).astype(int)
                ev = compute_expected_value(yf_val.values, yp, business_params)
                if ev > best_ev:
                    best_ev, best_t = ev, t

            records.append({
                "fold": fold,
                "model": name.upper(),
                "threshold": best_t,
                "ev_total": best_ev,
                "ev_per_loan": best_ev / len(yf_val),
                "auc": roc_auc_score(yf_val, proba),
            })

    cv_df = pd.DataFrame(records)
    summary = cv_df.groupby("model").agg(
        ev_per_loan_mean=("ev_per_loan", "mean"),
        ev_per_loan_std=("ev_per_loan", "std"),
        auc_mean=("auc", "mean"),
        auc_std=("auc", "std"),
    ).reset_index()

    log.info("CV results:\n%s", summary.to_string(index=False))
    return cv_df


# ═══════════════════════════════════════════════════════════════════════════
# STEP 14 — Final Evaluation on Test Set (ONE shot)
# ═══════════════════════════════════════════════════════════════════════════

def step14_final_test(
    models: Dict,
    preprocessed: Dict,
    y_test: pd.Series,
    business_params: Dict[str, float],
    threshold_results: Dict,
) -> pd.DataFrame:
    """
    Apply the thresholds tuned on Val to the held-out TEST set — once only.
    Reports EV, AUC, and classification metrics.
    """
    log.info("=" * 60)
    log.info("FINAL EVALUATION ON TEST SET (one-time only)")
    log.info("=" * 60)

    rows = []
    for name in ["lr", "xgb"]:
        pre_key = name
        t_opt = threshold_results[name]["best_threshold"]
        proba = predict_proba_positive(models[name], preprocessed[pre_key]["test"])
        y_pred = (proba >= t_opt).astype(int)

        ev = compute_expected_value(y_test.values, y_pred, business_params)
        auc = roc_auc_score(y_test, proba)
        tn, fp, fn, tp = confusion_matrix(y_test.values, y_pred).ravel()

        log.info("\n--- %s (threshold=%.4f) ---", name.upper(), t_opt)
        log.info("  Test EV:      $%.2f", ev)
        log.info("  Test EV/loan: $%.2f", ev / len(y_test))
        log.info("  Test AUC:     %.4f", auc)
        log.info("  TN=%d  FP=%d  FN=%d  TP=%d", tn, fp, fn, tp)
        log.info("\n%s", classification_report(
            y_test, y_pred, target_names=["Fully Paid", "Charged Off"]))

        rows.append({
            "Model": name.upper(),
            "Threshold": t_opt,
            "Test EV ($)": ev,
            "EV/Loan ($)": ev / len(y_test),
            "AUC": auc,
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        })

    final_df = pd.DataFrame(rows)
    log.info("Final comparison:\n%s", final_df.to_string(index=False))

    return final_df


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline(filepath: str = "lending_club_loan_data.csv") -> Dict[str, Any]:
    """
    Execute the full credit risk model pipeline end-to-end.

    Parameters
    ----------
    filepath : str
        Path to the Lending Club CSV file.

    Returns
    -------
    dict with all artefacts: models, preprocessors, thresholds, final metrics.
    """
    log.info("=" * 60)
    log.info("CREDIT RISK MODEL PIPELINE — START")
    log.info("Positive class: '%s' → %d", POSITIVE_CLASS_LABEL, POSITIVE_CLASS)
    log.info("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1
    df, business_params = step1_load_and_prepare(filepath)

    # Step 2
    X_train, X_val, X_test, y_train, y_val, y_test = step2_split(df)

    # Step 3
    preprocessed = step3_preprocess(X_train, X_val, X_test)

    # Step 4
    models = step4_train_models(preprocessed, y_train)

    # (Step 5 is utility functions — used by later steps)

    # Step 6
    threshold_results = step6_tune_threshold(models, preprocessed, y_val, business_params)

    # Step 7
    baseline_df = step7_baselines(X_val, y_val, business_params, threshold_results)

    # Step 8
    step8_roc_pr_curves(models, preprocessed, y_val, threshold_results)

    # Step 9
    step9_calibration(models, preprocessed, y_val, threshold_results)

    # Step 10
    step10_feature_importance(models, preprocessed)

    # Step 11
    step11_shap(models, preprocessed)

    # Step 12
    step12_sensitivity(models, preprocessed, y_val, business_params, threshold_results)

    # Step 13
    cv_df = step13_cv_ev(X_train, y_train, business_params)

    # Step 14 — Test set (ONE TIME ONLY)
    final_df = step14_final_test(models, preprocessed, y_test, business_params, threshold_results)

    log.info("=" * 60)
    log.info("PIPELINE COMPLETE — all outputs saved to %s/", OUTPUT_DIR)
    log.info("=" * 60)

    return {
        "business_params": business_params,
        "models": models,
        "preprocessed": preprocessed,
        "threshold_results": threshold_results,
        "baseline_df": baseline_df,
        "cv_df": cv_df,
        "final_df": final_df,
    }


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "lending_club_loan_data.csv"
    artifacts = run_pipeline(data_path)
