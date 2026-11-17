"""
Credit Risk Dashboard, financial-report style
=============================================
A second layout of the dashboard in the style of a Power BI financial report.

Every number comes from the project itself:
- docs/data/*.csv            model results saved by the last notebook cell
- data/lending_club.db       loan-level data built by sql/load_data.py

Run from the repo root after the notebook and sql/load_data.py:
    python src/create_dashboard_v2.py
Outputs docs/dashboard_v2.html (and docs/dashboard_v2.png if kaleido works).
"""

import sqlite3

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ═══════════════════════════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════════════════════════

DATA = "docs/data"
strategies = pd.read_csv(f"{DATA}/model_comparison.csv")          # validation set
curve = pd.read_csv(f"{DATA}/ev_threshold_curve.csv")             # validation set
test = pd.read_csv(f"{DATA}/test_results_summary.csv").set_index("Model")

with sqlite3.connect("data/lending_club.db") as con:
    by_grade = pd.read_sql(
        """
        SELECT grade,
               COUNT(*)                                   AS loans,
               100.0 * AVG(loan_status = 'Charged Off')   AS default_rate,
               (1 - AVG(loan_status = 'Charged Off'))
                 * AVG(CASE WHEN loan_status = 'Fully Paid'  THEN total_pymnt - loan_amnt END)
               - AVG(loan_status = 'Charged Off')
                 * AVG(CASE WHEN loan_status = 'Charged Off' THEN loan_amnt - total_pymnt END)
                                                          AS ev_per_loan
        FROM loans GROUP BY grade ORDER BY grade
        """, con)
    by_purpose = pd.read_sql(
        "SELECT purpose, COUNT(*) AS loans FROM loans GROUP BY purpose ORDER BY loans DESC", con)
    by_fico = pd.read_sql(
        """
        SELECT (fico_range_low / 20) * 20                   AS fico_band,
               COUNT(*)                                     AS loans,
               100.0 * AVG(loan_status = 'Charged Off')     AS default_rate
        FROM loans GROUP BY fico_band HAVING COUNT(*) >= 500 ORDER BY fico_band
        """, con)
    total_loans, default_rate = con.execute(
        "SELECT COUNT(*), 100.0 * AVG(loan_status = 'Charged Off') FROM loans").fetchone()

# Top 6 purposes, the rest grouped together
top = by_purpose.head(6).copy()
top.loc[len(top)] = ["all_other_purposes", by_purpose["loans"].iloc[6:].sum()]
top["purpose"] = top["purpose"].str.replace("_", " ").str.capitalize()

xgb, lr = test.loc["XGB"], test.loc["LR"]
best = int(curve["EV_XGBoost"].idxmax())
ev = dict(zip(strategies["Strategy"], strategies["EV_per_Loan"]))
test_size = int(xgb[["TN", "FP", "FN", "TP"]].sum())
recall = xgb["TP"] / (xgb["TP"] + xgb["FN"])

# ═══════════════════════════════════════════════════════════════
#  COLORS
# ═══════════════════════════════════════════════════════════════

C = {
    "header_bg": "#3F3D89",
    "card_bg": "#FFFFFF",
    "page_bg": "#F5F4FA",
    "primary": "#1F2D67",
    "secondary": "#5B9BD5",
    "accent_green": "#5BDDA8",
    "accent_teal": "#3FB8A8",
    "danger": "#D64550",
    "success": "#1BA974",
    "text_dark": "#1F2D67",
    "text_muted": "#6B7280",
    "border": "#E5E7EB",
}
DONUT_COLORS = ["#1F2D67", "#3FB8A8", "#5B9BD5", "#5BDDA8", "#8E9AC1", "#A8DBD2", "#7ABCDB"]
FONT = "Segoe UI"

# ═══════════════════════════════════════════════════════════════
#  CHARTS
# ═══════════════════════════════════════════════════════════════

fig = make_subplots(
    rows=2, cols=3,
    specs=[
        [{"type": "xy", "secondary_y": True}, {"type": "xy"}, {"type": "domain"}],
        [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
    ],
    subplot_titles=(
        "<b>Loans and Default Rate by Grade</b>",
        "<b>Expected Value per Loan by Grade</b>",
        "<b>Loans by Purpose</b>",
        "<b>Default Rate by FICO Band</b>",
        "<b>Strategy Comparison (validation set)</b>",
        "<b>Profit vs Approval Threshold (XGBoost)</b>",
    ),
    row_heights=[0.50, 0.50],
    horizontal_spacing=0.08,
    vertical_spacing=0.18,
)

# 1. Loans (bars) and default rate (line) by grade
fig.add_trace(go.Bar(
    x=by_grade["grade"], y=by_grade["loans"], name="Loans",
    marker_color=C["primary"],
    hovertemplate="<b>Grade %{x}</b><br>Loans: %{y:,}<extra></extra>",
), row=1, col=1, secondary_y=False)
fig.add_trace(go.Scatter(
    x=by_grade["grade"], y=by_grade["default_rate"], name="Default rate (%)",
    mode="lines+markers", line={"color": C["secondary"], "width": 3},
    marker={"size": 8, "color": C["secondary"]},
    hovertemplate="<b>Grade %{x}</b><br>Default rate: %{y:.1f}%<extra></extra>",
), row=1, col=1, secondary_y=True)

# 2. Expected value per loan by grade (red = loses money)
fig.add_trace(go.Bar(
    x=by_grade["grade"], y=by_grade["ev_per_loan"],
    marker_color=[C["success"] if v > 0 else C["danger"] for v in by_grade["ev_per_loan"]],
    text=[f"-${-v:,.0f}" if v < 0 else f"${v:,.0f}" for v in by_grade["ev_per_loan"]], textposition="outside",
    textfont={"size": 10, "color": C["text_dark"]}, showlegend=False,
    hovertemplate="<b>Grade %{x}</b><br>EV per loan: $%{y:,.0f}<extra></extra>",
), row=1, col=2)

# 3. Loans by purpose
fig.add_trace(go.Pie(
    labels=top["purpose"], values=top["loans"], hole=0.55,
    marker={"colors": DONUT_COLORS, "line": {"color": "white", "width": 2}},
    textinfo="label+percent", textfont={"size": 10, "family": FONT},
    textposition="outside", showlegend=False,
    hovertemplate="<b>%{label}</b><br>Loans: %{value:,}<br>Share: %{percent}<extra></extra>",
), row=1, col=3)

# 4. Default rate by FICO band
fig.add_trace(go.Scatter(
    x=[f"{b}-{b + 19}" for b in by_fico["fico_band"]], y=by_fico["default_rate"],
    mode="lines+markers+text", line={"color": C["accent_teal"], "width": 3},
    marker={"size": 8, "color": C["accent_teal"]},
    text=[f"{v:.1f}%" for v in by_fico["default_rate"]], textposition="top center",
    textfont={"size": 9, "color": C["text_dark"]}, showlegend=False,
    hovertemplate="<b>FICO %{x}</b><br>Default rate: %{y:.1f}%<extra></extra>",
), row=2, col=1)

# 5. Strategy comparison
s = strategies.sort_values("EV_per_Loan")
fig.add_trace(go.Bar(
    y=s["Strategy"], x=s["EV_per_Loan"], orientation="h",
    marker_color=[C["primary"] if "Model" in n else "#9CA3AF" for n in s["Strategy"]],
    text=[f"${v:,.0f}" for v in s["EV_per_Loan"]], textposition="outside",
    textfont={"size": 11, "color": C["text_dark"]}, showlegend=False,
    hovertemplate="<b>%{y}</b><br>EV per loan: $%{x:,.2f}<extra></extra>",
), row=2, col=2)

# 6. Expected value at each threshold
fig.add_trace(go.Scatter(
    x=curve["Threshold"], y=curve["EV_XGBoost"], mode="lines",
    line={"color": C["primary"], "width": 3}, name="XGBoost EV per loan",
    showlegend=False,
    hovertemplate="Threshold: %{x:.2f}<br>EV per loan: $%{y:,.0f}<extra></extra>",
), row=2, col=3)
fig.add_annotation(
    x=curve["Threshold"][best], y=curve["EV_XGBoost"][best],
    text=f"<b>Best: {curve['Threshold'][best]:.2f}</b>",
    showarrow=True, arrowhead=2, arrowcolor=C["primary"], ax=40, ay=-30,
    font={"color": C["primary"], "size": 11, "family": FONT}, row=2, col=3,
)

# ═══════════════════════════════════════════════════════════════
#  HEADER + KPI CARDS
# ═══════════════════════════════════════════════════════════════

fig.add_shape(type="rect", xref="paper", yref="paper", x0=0, x1=1, y0=1.21, y1=1.30,
              fillcolor=C["header_bg"], line={"width": 0}, layer="below")
fig.add_annotation(xref="paper", yref="paper", x=0.005, y=1.285, xanchor="left", yanchor="top",
                   text="<b>Credit Risk Dashboard</b>",
                   font={"size": 22, "color": "white", "family": FONT}, showarrow=False)
fig.add_annotation(xref="paper", yref="paper", x=0.005, y=1.235, xanchor="left", yanchor="top",
                   text="Lending Club, 80,000 finished loans  |  XGBoost vs Logistic Regression  |  "
                        "Expected value per loan, measured against not lending",
                   font={"size": 11, "color": "#D8D6E8", "family": FONT}, showarrow=False)

kpi_y_top, kpi_y_bot = 1.18, 1.03
kpis = [
    {"title": "Loans Analyzed", "main": f"{total_loans:,}",
     "side_label": "Test set", "side": f"{test_size:,}",
     "trend_label": "Default rate", "trend": f"{default_rate:.1f}%", "color": C["text_dark"]},
    {"title": "Expected Value per Loan (test)", "main": f"${xgb['EV/Loan ($)']:,.0f}",
     "side_label": "Total on test set", "side": f"${xgb['Test EV ($)'] / 1e6:.2f}M",
     "trend_label": "vs approve all (validation)",
     "trend": f"{ev['XGB Model'] / ev['Approve All']:.1f}x", "color": C["success"]},
    {"title": "Approval Rate (test)", "main": f"{xgb['Approval_Rate']:.0%}",
     "side_label": "Defaults caught", "side": f"{recall:.0%}",
     "trend_label": "vs FICO < 670 rule (validation)",
     "trend": f"{ev['XGB Model'] / ev['FICO < 670 Rule']:.1f}x", "color": C["success"]},
    {"title": "Model AUC (test)", "main": f"{xgb['AUC']:.3f}",
     "side_label": "Best threshold", "side": f"{xgb['Threshold']:.2f}",
     "trend_label": "Logistic Regression AUC", "trend": f"{lr['AUC']:.3f}", "color": C["text_dark"]},
]
starts = [0.005, 0.255, 0.505, 0.755]
for kpi, x0 in zip(kpis, starts):
    x1 = x0 + 0.24
    mid_y = (kpi_y_top + kpi_y_bot) / 2
    fig.add_shape(type="rect", xref="paper", yref="paper", x0=x0, x1=x1, y0=kpi_y_bot, y1=kpi_y_top,
                  fillcolor=C["card_bg"], line={"color": C["border"], "width": 1})
    mid_x = (x0 + x1) / 2 + 0.005
    fig.add_shape(type="line", xref="paper", yref="paper", x0=mid_x, x1=mid_x,
                  y0=kpi_y_bot + 0.015, y1=kpi_y_top - 0.015, line={"color": C["border"], "width": 1})
    notes = [
        (x0 + 0.012, kpi_y_top - 0.015, "left", "top", f"<b>{kpi['title']}</b>", 11, C["text_dark"]),
        (x0 + 0.012, mid_y - 0.01, "left", "middle", f"<b>{kpi['main']}</b>", 24, C["primary"]),
        (x1 - 0.012, kpi_y_top - 0.015, "right", "top", kpi["side_label"], 9, C["text_muted"]),
        (x1 - 0.012, mid_y + 0.018, "right", "middle", f"<b>{kpi['side']}</b>", 14, C["text_dark"]),
        (x1 - 0.012, mid_y - 0.018, "right", "middle", kpi["trend_label"], 9, C["text_muted"]),
        (x1 - 0.012, kpi_y_bot + 0.012, "right", "bottom", f"<b>{kpi['trend']}</b>", 13, kpi["color"]),
    ]
    for x, y, xa, ya, text, size, color in notes:
        fig.add_annotation(xref="paper", yref="paper", x=x, y=y, xanchor=xa, yanchor=ya, text=text,
                           font={"size": size, "color": color, "family": FONT}, showarrow=False)

# ═══════════════════════════════════════════════════════════════
#  LAYOUT
# ═══════════════════════════════════════════════════════════════

fig.update_layout(
    paper_bgcolor=C["page_bg"], plot_bgcolor=C["card_bg"],
    font={"family": FONT, "color": C["text_dark"]},
    height=1200, width=1900,
    margin={"l": 30, "r": 30, "t": 320, "b": 40},
    showlegend=True,
    legend={"orientation": "h", "yanchor": "bottom", "y": 0.52, "xanchor": "left", "x": 0.005,
            "bgcolor": "rgba(255,255,255,0)", "font": {"size": 10}},
    bargap=0.4,
)
axis = {"gridcolor": C["border"], "zeroline": False, "tickfont": {"size": 9}}
fig.update_yaxes(title_text="Loans", row=1, col=1, secondary_y=False, **axis)
fig.update_yaxes(title_text="Default rate (%)", row=1, col=1, secondary_y=True, **axis)
fig.update_xaxes(title_text="Grade", row=1, col=1, **axis)
fig.update_yaxes(title_text="EV per loan ($)", row=1, col=2, **axis)
fig.update_xaxes(title_text="Grade", row=1, col=2, **axis)
fig.update_yaxes(title_text="Default rate (%)", row=2, col=1, **axis)
fig.update_xaxes(title_text="FICO score", row=2, col=1, **axis)
fig.update_yaxes(row=2, col=2, gridcolor=C["border"], zeroline=False,
                 tickfont={"size": 10, "color": C["text_dark"]})
fig.update_xaxes(title_text="EV per loan ($)", row=2, col=2, **axis)
fig.update_yaxes(title_text="EV per loan ($)", row=2, col=3, **axis)
fig.update_xaxes(title_text="Approval threshold (predicted default probability)", row=2, col=3, **axis)

# ═══════════════════════════════════════════════════════════════
#  SAVE
# ═══════════════════════════════════════════════════════════════

fig.write_html("docs/dashboard_v2.html", include_plotlyjs="cdn")
print("[OK] Saved: docs/dashboard_v2.html")
try:
    fig.write_image("docs/dashboard_v2.png", width=1900, height=1200, scale=2)
    print("[OK] Saved: docs/dashboard_v2.png")
except Exception as e:
    print(f"[SKIP] PNG not saved ({type(e).__name__}). Install kaleido or screenshot the HTML.")
