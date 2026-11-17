"""
Credit Risk Analytics Dashboard Generator
==========================================
Creates a Power BI-style dashboard summarizing model performance,
business value, and feature importance for stakeholder review.

Outputs:
- docs/dashboard.png  (static image for resume/README)
- docs/dashboard.html (interactive version)
Reads the results that the notebook saves in docs/data/.
"""

import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ═══════════════════════════════════════════════════════════════
#  RESULTS — read from docs/data/, written by the last cell of
#  notebooks/credit_risk_pipeline.ipynb (run the notebook first)
# ═══════════════════════════════════════════════════════════════
import numpy as np

DATA = "docs/data"
model_comparison = pd.read_csv(f"{DATA}/model_comparison.csv")
business_params = pd.read_csv(f"{DATA}/business_parameters.csv")
feature_importance = pd.read_csv(f"{DATA}/feature_importance.csv")
curve = pd.read_csv(f"{DATA}/ev_threshold_curve.csv")
test_df = pd.read_csv(f"{DATA}/test_results_summary.csv")

thresholds = curve["Threshold"].to_numpy()
ev_curve_xgb = curve["EV_XGBoost"].to_numpy()
ev_curve_lr = curve["EV_LogisticRegression"].to_numpy()
best_xgb, best_lr = int(ev_curve_xgb.argmax()), int(ev_curve_lr.argmax())

xgb_row = test_df[test_df["Model"] == "XGB"].iloc[0]
test_results = {
    "Model": "XGBoost",
    "AUC": float(xgb_row["AUC"]),
    "Threshold": float(xgb_row["Threshold"]),
    "EV_per_Loan": float(xgb_row["EV/Loan ($)"]),
    "Total_EV": float(xgb_row["Test EV ($)"]),
    "Approval_Rate": float(xgb_row["Approval_Rate"]),
    "TN": int(xgb_row["TN"]), "FP": int(xgb_row["FP"]),
    "FN": int(xgb_row["FN"]), "TP": int(xgb_row["TP"]),
}
approve_all_ev = float(model_comparison.loc[model_comparison["Strategy"] == "Approve All", "EV_per_Loan"].iloc[0])
model_val_ev = float(model_comparison["EV_per_Loan"].max())

# Dataset summary
dataset_summary = {
    "Total_Loans": 80000,
    "Default_Rate": 0.1981,
    "Train_Size": 47962,
    "Val_Size": 15988,
    "Test_Size": 15988,
    "Features_Used": 17,
}

# ═══════════════════════════════════════════════════════════════
#  COLOR PALETTE (Power BI-inspired)
# ═══════════════════════════════════════════════════════════════

COLORS = {
    "primary": "#118DFF",      # Power BI Blue
    "secondary": "#12239E",    # Dark Blue
    "success": "#118C45",      # Green
    "warning": "#E66C37",      # Orange
    "danger": "#D64550",       # Red
    "neutral": "#605E5C",      # Gray
    "background": "#F3F2F1",   # Light Gray
    "card": "#FFFFFF",         # White
    "text": "#252423",         # Almost Black
}

# ═══════════════════════════════════════════════════════════════
#  DASHBOARD CONSTRUCTION
# ═══════════════════════════════════════════════════════════════

fig = make_subplots(
    rows=4, cols=4,
    specs=[
        # Row 1: KPI Cards
        [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
        # Row 2: Model Comparison (span 2 cols) + Confusion Matrix (span 2 cols)
        [{"type": "bar", "colspan": 2}, None, {"type": "heatmap", "colspan": 2}, None],
        # Row 3: EV vs Threshold curve (span 4 cols)
        [{"type": "scatter", "colspan": 4}, None, None, None],
        # Row 4: Feature Importance (span 2) + Business Params (span 2)
        [{"type": "bar", "colspan": 2}, None, {"type": "table", "colspan": 2}, None],
    ],
    subplot_titles=(
        "", "", "", "",
        "<b>Model Performance vs Baselines (EV per Loan, USD)</b>",
        "<b>Test Set Confusion Matrix (Business Outcome)</b>",
        "<b>Expected Value vs Classification Threshold</b>",
        "<b>Top Feature Importance (SHAP)</b>",
        "<b>Business Parameters (Derived from Payment Data)</b>",
    ),
    row_heights=[0.14, 0.32, 0.27, 0.27],
    horizontal_spacing=0.10,
    vertical_spacing=0.13,
)

# ───────────────────────────────────────────────────────────────
#  Row 1: KPI Cards
# ───────────────────────────────────────────────────────────────

kpi_configs = [
    {
        "title": "Dataset Size",
        "value": 80000,
        "suffix": " loans",
        "color": COLORS["primary"],
        "subtitle": f"Default Rate: {dataset_summary['Default_Rate']:.1%}",
    },
    {
        "title": "Test AUC",
        "value": test_results["AUC"],
        "suffix": "",
        "color": COLORS["secondary"],
        "subtitle": "XGBoost on held-out set",
    },
    {
        "title": "EV per Loan",
        "value": test_results["EV_per_Loan"],
        "prefix": "$",
        "color": COLORS["success"],
        "subtitle": f"Total: ${test_results['Total_EV']/1e6:.1f}M",
    },
    {
        "title": "Approval Rate",
        "value": test_results["Approval_Rate"] * 100,
        "suffix": "%",
        "color": COLORS["warning"],
        "subtitle": f"Approve-all baseline: ${approve_all_ev:,.0f}/loan",
    },
]

for i, kpi in enumerate(kpi_configs):
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=kpi["value"],
            number={
                "prefix": kpi.get("prefix", ""),
                "suffix": kpi.get("suffix", ""),
                "font": {"size": 38, "color": kpi["color"], "family": "Segoe UI"},
                "valueformat": ",.4f" if kpi["title"] == "Test AUC" else ",.0f",
            },
            title={
                "text": f"<b>{kpi['title']}</b><br><span style='font-size:11px;color:#605E5C'>{kpi['subtitle']}</span>",
                "font": {"size": 14, "color": COLORS["text"], "family": "Segoe UI"},
            },
            domain={"row": 0, "column": i},
        ),
        row=1, col=i+1,
    )

# ───────────────────────────────────────────────────────────────
#  Row 2 (Left): Model Comparison Bar Chart
# ───────────────────────────────────────────────────────────────

bar_colors = [COLORS["success"], COLORS["primary"], COLORS["neutral"],
              COLORS["warning"], COLORS["danger"]]

fig.add_trace(
    go.Bar(
        x=model_comparison["Strategy"],
        y=model_comparison["EV_per_Loan"],
        marker_color=bar_colors,
        text=[f"${v:,.0f}" for v in model_comparison["EV_per_Loan"]],
        textposition="outside",
        textfont={"size": 11, "family": "Segoe UI", "color": COLORS["text"]},
        showlegend=False,
        hovertemplate="<b>%{x}</b><br>EV/Loan: $%{y:,.2f}<extra></extra>",
    ),
    row=2, col=1,
)

# ───────────────────────────────────────────────────────────────
#  Row 2 (Right): Confusion Matrix Heatmap
#  Layout (standard convention):
#    Pred Repay (0) | Pred Default (1)
#    -----------------------------------
#    Actual Repay   |   TN     |   FP
#    Actual Default |   FN     |   TP
#
#  Plotly heatmap: cm[0] renders at BOTTOM, cm[1] at TOP
#  So we put "Actual Default" row first (will be at top visually)
# ───────────────────────────────────────────────────────────────

cm = [
    [test_results["FN"], test_results["TP"]],   # Actual Default (top row visually)
    [test_results["TN"], test_results["FP"]],   # Actual Repay (bottom row visually)
]

cm_text = [
    [f"<b>FN: {test_results['FN']:,}</b><br>Bad loan funded<br>(Critical error)",
     f"<b>TP: {test_results['TP']:,}</b><br>Correctly rejected<br>(Avoided default)"],
    [f"<b>TN: {test_results['TN']:,}</b><br>Earned profit<br>(Correct approve)",
     f"<b>FP: {test_results['FP']:,}</b><br>Missed opportunity<br>(False reject)"],
]

fig.add_trace(
    go.Heatmap(
        z=cm,
        x=["Predicted: Repay", "Predicted: Default"],
        y=["Actual: Default", "Actual: Repay"],
        text=cm_text,
        texttemplate="%{text}",
        textfont={"size": 11, "family": "Segoe UI", "color": "white"},
        colorscale=[[0, "#A0C4E0"], [1, "#118DFF"]],
        showscale=False,
        hovertemplate="<b>%{x}</b><br><b>%{y}</b><br>Count: %{z:,}<extra></extra>",
    ),
    row=2, col=3,
)

# ───────────────────────────────────────────────────────────────
#  Row 3: EV vs Threshold Line Chart
# ───────────────────────────────────────────────────────────────

fig.add_trace(
    go.Scatter(
        x=thresholds, y=ev_curve_xgb,
        mode="lines",
        line={"color": COLORS["success"], "width": 3},
        name="XGBoost",
        hovertemplate="Threshold: %{x:.2f}<br>EV/Loan: $%{y:,.0f}<extra>XGBoost</extra>",
    ),
    row=3, col=1,
)
fig.add_trace(
    go.Scatter(
        x=thresholds, y=ev_curve_lr,
        mode="lines",
        line={"color": COLORS["primary"], "width": 3, "dash": "dash"},
        name="Logistic Regression",
        hovertemplate="Threshold: %{x:.2f}<br>EV/Loan: $%{y:,.0f}<extra>LR</extra>",
    ),
    row=3, col=1,
)

# Optimal threshold markers
fig.add_trace(
    go.Scatter(
        x=[thresholds[best_xgb]], y=[ev_curve_xgb[best_xgb]],
        mode="markers+text",
        marker={"size": 14, "color": COLORS["success"],
                "line": {"width": 2, "color": "white"}},
        text=[f"XGB Optimal: {thresholds[best_xgb]:.2f}"],
        textposition="top right",
        textfont={"size": 11, "color": COLORS["success"], "family": "Segoe UI"},
        showlegend=False,
        hovertemplate=f"XGBoost Optimal<br>Threshold: {thresholds[best_xgb]:.2f}<br>EV: ${ev_curve_xgb[best_xgb]:,.0f}<extra></extra>",
    ),
    row=3, col=1,
)
fig.add_trace(
    go.Scatter(
        x=[thresholds[best_lr]], y=[ev_curve_lr[best_lr]],
        mode="markers+text",
        marker={"size": 14, "color": COLORS["primary"],
                "line": {"width": 2, "color": "white"}},
        text=[f"LR Optimal: {thresholds[best_lr]:.2f}"],
        textposition="bottom left",
        textfont={"size": 11, "color": COLORS["primary"], "family": "Segoe UI"},
        showlegend=False,
        hovertemplate=f"LR Optimal<br>Threshold: {thresholds[best_lr]:.2f}<br>EV: ${ev_curve_lr[best_lr]:,.0f}<extra></extra>",
    ),
    row=3, col=1,
)

# ───────────────────────────────────────────────────────────────
#  Row 4 (Left): Feature Importance
# ───────────────────────────────────────────────────────────────

sorted_features = feature_importance.sort_values("SHAP_Importance")

fig.add_trace(
    go.Bar(
        y=sorted_features["Feature"],
        x=sorted_features["SHAP_Importance"],
        orientation="h",
        marker_color=COLORS["primary"],
        text=[f"{v:.3f}" for v in sorted_features["SHAP_Importance"]],
        textposition="outside",
        textfont={"size": 10, "family": "Segoe UI", "color": COLORS["text"]},
        showlegend=False,
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:.3f}<extra></extra>",
    ),
    row=4, col=1,
)

# ───────────────────────────────────────────────────────────────
#  Row 4 (Right): Business Parameters Table
# ───────────────────────────────────────────────────────────────

fig.add_trace(
    go.Table(
        header={
            "values": ["<b>Parameter</b>", "<b>Value (USD)</b>", "<b>Impact</b>"],
            "fill_color": COLORS["primary"],
            "font": {"color": "white", "size": 13, "family": "Segoe UI"},
            "align": "left",
            "height": 30,
        },
        cells={
            "values": [
                business_params["Parameter"],
                [f"${v:,.2f}" for v in business_params["Value_USD"]],
                ["Profit ↑", "Loss ↓", "Cost ↓", "Reward ↑"],
            ],
            "fill_color": [["white", "#F3F2F1"] * 2],
            "font": {"size": 11, "family": "Segoe UI", "color": COLORS["text"]},
            "align": "left",
            "height": 28,
        },
    ),
    row=4, col=3,
)

# ═══════════════════════════════════════════════════════════════
#  LAYOUT & STYLING
# ═══════════════════════════════════════════════════════════════

fig.update_layout(
    title={
        "text": (
            "<b style='font-size:28px'>Credit Risk Analytics Dashboard</b><br>"
            "<span style='font-size:14px;color:#605E5C'>Lending Club Loan Portfolio  |  "
            "XGBoost vs Logistic Regression  |  Expected Value Optimization</span>"
        ),
        "x": 0.5, "xanchor": "center",
        "y": 0.985, "yanchor": "top",
        "font": {"family": "Segoe UI", "color": COLORS["text"]},
    },
    height=1500, width=1800,
    paper_bgcolor=COLORS["background"],
    plot_bgcolor=COLORS["card"],
    font={"family": "Segoe UI", "color": COLORS["text"]},
    margin={"l": 70, "r": 70, "t": 130, "b": 60},
    showlegend=True,
    legend={
        "orientation": "h",
        "yanchor": "top", "y": 0.50,
        "xanchor": "center", "x": 0.5,
        "bgcolor": "rgba(255,255,255,0.85)",
        "bordercolor": "#E1DFDD",
        "borderwidth": 1,
    },
)

# Axis styling
fig.update_yaxes(title_text="EV per Loan (USD)", row=2, col=1,
                 gridcolor="#E1DFDD", zeroline=False)
fig.update_xaxes(row=2, col=1, gridcolor="#E1DFDD", zeroline=False,
                 tickfont={"size": 10})

fig.update_yaxes(title_text="Expected Value per Loan (USD)", row=3, col=1,
                 gridcolor="#E1DFDD", zeroline=False)
fig.update_xaxes(title_text="Classification Threshold", row=3, col=1,
                 gridcolor="#E1DFDD", zeroline=False)

fig.update_xaxes(title_text="SHAP Value (mean |impact|)", row=4, col=1,
                 gridcolor="#E1DFDD", zeroline=False)
fig.update_yaxes(row=4, col=1, gridcolor="#E1DFDD", zeroline=False)

# Update subplot title styling
for annotation in fig["layout"]["annotations"]:
    annotation["font"] = {"size": 14, "color": COLORS["text"], "family": "Segoe UI"}
    annotation["xanchor"] = "left"

# ═══════════════════════════════════════════════════════════════
#  SAVE OUTPUTS
# ═══════════════════════════════════════════════════════════════

os.makedirs("docs", exist_ok=True)

# Interactive HTML (for in-browser viewing)
fig.write_html("docs/dashboard.html", include_plotlyjs="cdn")
print("[OK] Saved: docs/dashboard.html")

# Static PNG (for the README); needs the kaleido package
try:
    fig.write_image("docs/dashboard.png", width=1800, height=1400, scale=2)
    print("[OK] Saved: docs/dashboard.png")
except Exception as e:
    print(f"[SKIP] PNG not saved ({type(e).__name__}). Install kaleido or screenshot docs/dashboard.html.")

# The CSVs in docs/data/ come from the notebook and can be imported into Power BI as they are.

print("\nDashboard generation complete!")
print("View interactively: open docs/dashboard.html")
print("Static image: docs/dashboard.png")
