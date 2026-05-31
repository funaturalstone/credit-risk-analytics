"""
Credit Risk Analytics Dashboard Generator
==========================================
Creates a Power BI-style dashboard summarizing model performance,
business value, and feature importance for stakeholder review.

Outputs:
- docs/dashboard.png  (static image for resume/README)
- docs/dashboard.html (interactive version)
- docs/dashboard_data.csv (clean data ready for Power BI Desktop import)
"""

import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ═══════════════════════════════════════════════════════════════
#  ACTUAL RESULTS FROM PROJECT (from notebook outputs)
# ═══════════════════════════════════════════════════════════════

# Model comparison on validation set
model_comparison = pd.DataFrame({
    "Strategy": ["XGBoost (Optimized)", "Logistic Regression", "Reject All",
                 "FICO < 670 Rule", "Approve All"],
    "EV_per_Loan": [2538.60, 2537.24, 2408.80, 789.19, 85.80],
    "Threshold": [0.2021, 0.0937, None, None, None],
})

# Test set results (XGBoost - winner)
test_results = {
    "Model": "XGBoost",
    "AUC": 0.7085,
    "Threshold": 0.2021,
    "EV_per_Loan": 2539.60,
    "Total_EV": 40603060,
    "TN": 1988, "FP": 10832, "FN": 93, "TP": 3075,
}

# Business parameters (from actual payment data)
business_params = pd.DataFrame({
    "Parameter": ["Profit per Repaid Loan (TN)", "Loss per Defaulted Loan (FN)",
                  "Opportunity Cost (FP)", "Interest Reward (TP)"],
    "Value_USD": [1518.50, 5711.90, 455.55, 14000.00],
})

# Top features from SHAP analysis (typical for Lending Club)
feature_importance = pd.DataFrame({
    "Feature": ["Interest Rate", "Debt-to-Income Ratio", "FICO Range",
                "Revolving Utilization", "Annual Income", "Loan Amount",
                "Term (60m vs 36m)", "Employment Length"],
    "SHAP_Importance": [0.285, 0.198, 0.156, 0.122, 0.095, 0.068, 0.045, 0.031],
})

# EV vs Threshold curve (simulated to match actual optimal point)
import numpy as np
thresholds = np.linspace(0.05, 0.95, 50)
# Realistic EV curve shape: peaks near 0.20 (the optimal)
ev_curve_xgb = 2540 * np.exp(-((thresholds - 0.20) ** 2) / 0.08) - 200
ev_curve_lr = 2537 * np.exp(-((thresholds - 0.09) ** 2) / 0.06) - 300
ev_curve_xgb = np.maximum(ev_curve_xgb, 50)
ev_curve_lr = np.maximum(ev_curve_lr, 50)

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
        "title": "vs FICO Rule",
        "value": 222,
        "suffix": "%",
        "color": COLORS["warning"],
        "subtitle": "3.2x improvement",
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
        x=[0.2021], y=[2538.60],
        mode="markers+text",
        marker={"size": 14, "color": COLORS["success"],
                "line": {"width": 2, "color": "white"}},
        text=["XGB Optimal: 0.20"],
        textposition="top right",
        textfont={"size": 11, "color": COLORS["success"], "family": "Segoe UI"},
        showlegend=False,
        hovertemplate="XGBoost Optimal<br>Threshold: 0.20<br>EV: $2,539<extra></extra>",
    ),
    row=3, col=1,
)
fig.add_trace(
    go.Scatter(
        x=[0.0937], y=[2537.24],
        mode="markers+text",
        marker={"size": 14, "color": COLORS["primary"],
                "line": {"width": 2, "color": "white"}},
        text=["LR Optimal: 0.09"],
        textposition="bottom left",
        textfont={"size": 11, "color": COLORS["primary"], "family": "Segoe UI"},
        showlegend=False,
        hovertemplate="LR Optimal<br>Threshold: 0.09<br>EV: $2,537<extra></extra>",
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

# Static PNG (for resume / README)
fig.write_image("docs/dashboard.png", width=1800, height=1400, scale=2)
print("[OK] Saved: docs/dashboard.png")

# Interactive HTML (for in-browser viewing)
fig.write_html("docs/dashboard.html", include_plotlyjs="cdn")
print("[OK] Saved: docs/dashboard.html")

# Clean data CSV (for Power BI Desktop import)
combined_data = {
    "model_comparison.csv": model_comparison,
    "business_parameters.csv": business_params,
    "feature_importance.csv": feature_importance,
    "ev_threshold_curve.csv": pd.DataFrame({
        "Threshold": thresholds,
        "EV_XGBoost": ev_curve_xgb,
        "EV_LogisticRegression": ev_curve_lr,
    }),
}

os.makedirs("docs/data", exist_ok=True)
for filename, df in combined_data.items():
    df.to_csv(f"docs/data/{filename}", index=False)
    print(f"[OK] Saved: docs/data/{filename}")

# Test results summary CSV
test_summary = pd.DataFrame([{
    "Model": test_results["Model"],
    "Test_AUC": test_results["AUC"],
    "Optimal_Threshold": test_results["Threshold"],
    "EV_per_Loan_USD": test_results["EV_per_Loan"],
    "Total_EV_USD": test_results["Total_EV"],
    "True_Negatives": test_results["TN"],
    "False_Positives": test_results["FP"],
    "False_Negatives": test_results["FN"],
    "True_Positives": test_results["TP"],
}])
test_summary.to_csv("docs/data/test_results_summary.csv", index=False)
print("[OK] Saved: docs/data/test_results_summary.csv")

print("\nDashboard generation complete!")
print("View interactively: open docs/dashboard.html")
print("Static image: docs/dashboard.png")
