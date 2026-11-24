import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# =========================================================
# ELECTRA | Electricity Forecasting & Intelligence Dashboard
# =========================================================

st.set_page_config(
    page_title="ELECTRA | Energy Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"

BG = "#06111F"
PANEL = "#0B1B2B"
PANEL2 = "#0E2235"
BORDER = "#1B344A"
TEXT = "#E8F1FA"
MUTED = "#C3D2DF"
CYAN = "#16B8F3"
BLUE = "#287DE0"
GREEN = "#21C66B"
YELLOW = "#F5B51B"
ORANGE = "#F28C28"
PURPLE = "#9874EA"
PINK = "#EF6685"

# ------------------------- CSS ----------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root { color-scheme: dark; }
.stApp {
    background: radial-gradient(ellipse at 50% -20%, #102c45 0%, #06111f 55%);
    color: #E8F1FA;
    font-family: Inter, sans-serif;
}
.block-container {max-width: 1800px; padding: 0.8rem 1.15rem 1.4rem;}
header[data-testid="stHeader"] {background: transparent;}
footer {visibility:hidden;}
/* Higher-contrast text for readability */
.stApp, .stApp p, .stApp label, .stApp li,
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stWidgetLabel"] {
    color: #D8E4EE;
}
.stApp input, .stApp textarea, .stApp select {
    color: #F1F7FC !important;
}
.stApp [data-testid="stCaptionContainer"] p {
    color: #C0CFDC !important;
}
.stApp [data-testid="stRadio"] label p {
    color: #E4EDF5 !important;
}

#MainMenu {visibility:hidden;}
.topbar {
    display:flex; align-items:center; justify-content:space-between;
    gap:14px; padding:10px 14px; margin-bottom:14px;
    background:#050D18; border:1px solid #173047; border-radius:12px;
}
.brand {font-size:20px; font-weight:800; letter-spacing:1.2px; color:#F4F9FF;}
.brand span {color:#19A9F5;}
.brandmark {
    display:inline-flex; width:32px; height:32px; border-radius:9px;
    align-items:center; justify-content:center; margin-right:8px;
    background:linear-gradient(145deg,#147BE0,#12C4E8); color:white;
}
.topnote {font-size:11px; color:#C3D2DF; text-align:right;}
.pagehead {padding:2px 4px 12px;}
.pagehead h2 {font-size:19px; margin:0; color:#F4F8FC; font-weight:700;}
.pagehead p {font-size:12px; color:#C4D2DF; margin:3px 0 0;}
.kpi {
    background:linear-gradient(145deg,#10263A,#0A1929);
    border:1px solid #1C3B53; border-radius:12px;
    padding:13px 14px 11px; min-height:105px;
    box-shadow:0 5px 18px rgba(0,0,0,.16);
}
.kpi-label {font-size:10px; letter-spacing:.45px; color:#D0DCE7; text-transform:uppercase;}
.kpi-value {font-size:23px; line-height:1.35; font-weight:700; color:#F4F8FC; margin-top:8px;}
.kpi-foot {font-size:10px; color:#C3D2DF; margin-top:3px;}
.kpi-icon {float:right; font-size:22px; opacity:.95;}
.panel {
    background:linear-gradient(145deg,#0D2031,#091827);
    border:1px solid #1B354B; border-radius:12px;
    padding:13px 14px 8px; height:100%;
}
.panel-title {font-size:12px; color:#EAF3FA; font-weight:700; letter-spacing:.35px; margin-bottom:6px;}
.small-muted {font-size:10px; color:#B6C7D6;}
.section-title {font-size:13px; font-weight:700; color:#E7F0F8; margin:14px 2px 8px;}
.footerline {border-top:1px solid #183047; margin-top:18px; padding:13px 0 2px;
    text-align:center; color:#B6C7D6; font-size:11px; letter-spacing:.3px;}
div[data-testid="stHorizontalBlock"] {gap:10px;}
.stPlotlyChart {border-radius:8px;}
div[data-testid="stDataFrame"] {border:1px solid #1B354B; border-radius:8px;}
.stButton button, .stDownloadButton button {
    background:#0D3555; border:1px solid #1D6C9E; color:#E8F7FF;
    border-radius:7px; font-size:12px;
}
.stButton button:hover, .stDownloadButton button:hover {
    background:#12517D; border-color:#21B8F3; color:white;
}
[data-testid="stRadio"] > div {gap:4px;}
[data-testid="stRadio"] label {
    background:#0B1A29; border:1px solid #1B344A; border-radius:8px;
    padding:6px 12px; margin:0;
}
[data-testid="stRadio"] label p {font-size:11px; font-weight:600;}
[data-testid="stMetric"] {
    background:#0D2031; border:1px solid #1B354B;
    padding:10px; border-radius:10px;
}
[data-testid="stMetricValue"] {font-size:20px; color:#EAF6FF;}
[data-testid="stMetricLabel"] {font-size:10px; color:#D0DCE7;}
</style>
""", unsafe_allow_html=True)

# ---------------------- Data loading ----------------------

@st.cache_data
def load_data():
    historical_path = DATA_DIR / "PJME_hourly.csv"
    comparison_path = BASE_DIR / "model_comparison.csv"
    forecast_path = BASE_DIR / "future_forecast.csv"

    missing = [str(p) for p in [historical_path, comparison_path, forecast_path] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required project file(s): " + ", ".join(missing))

    hist = pd.read_csv(historical_path)
    comp = pd.read_csv(comparison_path)
    fc = pd.read_csv(forecast_path)

    hist["Datetime"] = pd.to_datetime(hist["Datetime"])
    fc["Datetime"] = pd.to_datetime(fc["Datetime"])
    hist = hist.sort_values("Datetime")
    fc = fc.sort_values("Datetime")

    return hist, comp, fc

try:
    historical, comparison, forecast = load_data()
except Exception as exc:
    st.error(f"Could not load project data: {exc}")
    st.info("Check that dataset/PJME_hourly.csv, model_comparison.csv and future_forecast.csv are beside app.py in the expected folders.")
    st.stop()

if not {"Model", "MAE", "RMSE", "R2 Score"}.issubset(comparison.columns):
    st.error("model_comparison.csv must contain Model, MAE, RMSE and R2 Score columns.")
    st.stop()
if not {"Datetime", "Predicted_PJME_MW"}.issubset(forecast.columns):
    st.error("future_forecast.csv must contain Datetime and Predicted_PJME_MW columns.")
    st.stop()

baseline_rows = comparison[comparison["Model"].str.lower() == "baseline"]
improved_rows = comparison[comparison["Model"].str.lower() == "improved"]
baseline = baseline_rows.iloc[0] if not baseline_rows.empty else comparison.iloc[0]
improved = improved_rows.iloc[0] if not improved_rows.empty else comparison.iloc[-1]

peak_row = forecast.loc[forecast["Predicted_PJME_MW"].idxmax()]
low_row = forecast.loc[forecast["Predicted_PJME_MW"].idxmin()]
avg_hist = historical["PJME_MW"].mean()
max_hist = historical["PJME_MW"].max()
avg_fc = forecast["Predicted_PJME_MW"].mean()
mae_reduction = ((baseline["MAE"] - improved["MAE"]) / baseline["MAE"] * 100) if baseline["MAE"] else 0

# ----------------------- Plot theme ------------------------

def polish(fig, height=260, legend=True):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=TEXT, size=10),
        margin=dict(l=8, r=8, t=28, b=8),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="left", x=0, font=dict(size=9, color="#C5D4E1")
        ) if legend else dict(visible=False),
        xaxis=dict(
            gridcolor="rgba(112,145,170,.13)", linecolor="#29445B",
            tickfont=dict(size=9, color=MUTED), zerolinecolor="#29445B"
        ),
        yaxis=dict(
            gridcolor="rgba(112,145,170,.13)", linecolor="#29445B",
            tickfont=dict(size=9, color=MUTED), zerolinecolor="#29445B"
        ),
        hoverlabel=dict(bgcolor="#10283D", font_color="white")
    )
    return fig

def panel_title(title, subtitle=None):
    st.markdown(f'<div class="panel-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="small-muted">{subtitle}</div>', unsafe_allow_html=True)

def kpi(label, value, foot, icon, accent):
    st.markdown(f"""
    <div class="kpi">
      <div class="kpi-icon" style="color:{accent}">{icon}</div>
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-foot">{foot}</div>
    </div>
    """, unsafe_allow_html=True)

# ----------------------- Top header ------------------------

st.markdown(f"""
<div class="topbar">
  <div class="brand"><span class="brandmark">⚡</span>ELECTRA <span style="color:#C4D2DF;font-size:10px;font-weight:500;letter-spacing:1px;">ENERGY INTELLIGENCE</span></div>
  <div class="topnote">MACHINE LEARNING • ELECTRICITY DEMAND<br><span style="color:#21C66B;">● MODEL RESULTS LOADED</span></div>
</div>
""", unsafe_allow_html=True)

nav = st.radio(
    "Dashboard navigation",
    ["Dashboard", "Historical", "Model Performance", "Forecast"],
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("""
<div class="pagehead">
  <h2>ENERGY MANAGEMENT & FORECASTING SYSTEM</h2>
  <p>Electricity consumption analytics · Random Forest regression · 24-observation forecast</p>
</div>
""", unsafe_allow_html=True)

# ---------------------- KPI strip --------------------------

kcols = st.columns(6)
kpi_items = [
    ("HISTORICAL AVERAGE", f"{avg_hist:,.0f} MW", "Across available dataset", "ϟ", YELLOW),
    ("HISTORICAL PEAK", f"{max_hist:,.0f} MW", "Maximum recorded demand", "↗", CYAN),
    ("FORECAST AVERAGE", f"{avg_fc:,.0f} MW", "Next 24 observations", "◉", GREEN),
    ("FORECAST PEAK", f"{peak_row['Predicted_PJME_MW']:,.0f} MW", peak_row["Datetime"].strftime("%d %b %Y, %H:%M"), "⚡", YELLOW),
    ("IMPROVED MAE", f"{improved['MAE']:,.1f} MW", f"{mae_reduction:.1f}% lower than baseline MAE", "◎", PURPLE),
    ("MODEL R²", f"{improved['R2 Score']:.4f}", "Improved model test score", "◈", GREEN),
]
for col, item in zip(kcols, kpi_items):
    with col:
        kpi(*item)

# ------------------------ Dashboard -------------------------

if nav == "Dashboard":
    st.markdown('<div class="section-title">LIVE-STYLE ANALYTICS OVERVIEW <span style="color:#B6C7D6;font-weight:400;">| historical dataset and saved forecast</span></div>', unsafe_allow_html=True)

    # Top three-panel row
    left, middle, right = st.columns([1.18, 0.92, 1.55], gap="small")

    with left:
        with st.container(border=True):
            panel_title("HISTORICAL DEMAND PROFILE", "Recent 7 days of recorded electricity consumption")
            recent = historical.tail(168)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=recent["Datetime"], y=recent["PJME_MW"], mode="lines",
                line=dict(color=CYAN, width=1.7), name="Demand",
                fill="tozeroy", fillcolor="rgba(22,184,243,.10)"
            ))
            fig.update_layout(xaxis_title=None, yaxis_title="MW", showlegend=False)
            st.plotly_chart(polish(fig, 245, False), use_container_width=True, config={"displayModeBar": False})

    with middle:
        with st.container(border=True):
            panel_title("DEMAND DISTRIBUTION", "Share of observations by demand range")
            bins = pd.cut(historical["PJME_MW"], bins=5)
            dist = bins.value_counts().sort_index()
            labels = [f"{int(x.left/1000)}–{int(x.right/1000)}k MW" for x in dist.index]
            fig = go.Figure(go.Pie(
                labels=labels, values=dist.values, hole=.66,
                marker=dict(colors=[BLUE, CYAN, GREEN, YELLOW, PURPLE],
                            line=dict(color=PANEL, width=2)),
                textinfo="percent", textfont=dict(size=9, color="white"),
                hovertemplate="%{label}<br>%{value:,} records<extra></extra>"
            ))
            fig.add_annotation(text=f"{len(historical):,}<br><span style='font-size:9px'>records</span>",
                               x=.5, y=.5, showarrow=False,
                               font=dict(size=16, color=TEXT))
            fig.update_layout(showlegend=True, legend=dict(orientation="v", x=1.0, y=.5, font=dict(size=8)))
            st.plotly_chart(polish(fig, 245), use_container_width=True, config={"displayModeBar": False})

    with right:
        with st.container(border=True):
            panel_title("HISTORICAL TREND", "Monthly average demand")
            monthly = historical.set_index("Datetime")["PJME_MW"].resample("MS").mean().dropna()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly.index, y=monthly.values, mode="lines",
                name="Monthly average", line=dict(color=YELLOW, width=2),
                fill="tozeroy", fillcolor="rgba(245,181,27,.07)"
            ))
            fig.update_layout(xaxis_title=None, yaxis_title="Average MW", showlegend=False)
            st.plotly_chart(polish(fig, 245, False), use_container_width=True, config={"displayModeBar": False})

    # Second row
    c1, c2, c3 = st.columns([1.05, 1.45, 1.05], gap="small")

    with c1:
        with st.container(border=True):
            panel_title("MODEL PERFORMANCE", "Baseline vs improved model")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=["Baseline", "Improved"], y=[baseline["MAE"], improved["MAE"]],
                marker_color=[BLUE, GREEN], text=[f"{baseline['MAE']:,.0f}", f"{improved['MAE']:,.0f}"],
                textposition="outside", name="MAE"
            ))
            fig.update_layout(showlegend=False, yaxis_title="MAE (MW)")
            st.plotly_chart(polish(fig, 220, False), use_container_width=True, config={"displayModeBar": False})

    with c2:
        with st.container(border=True):
            panel_title("NEXT 24 OBSERVATIONS", "Recursive electricity demand forecast")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=forecast["Datetime"], y=forecast["Predicted_PJME_MW"],
                name="Predicted demand", marker_color=BLUE, opacity=.78
            ))
            fig.add_trace(go.Scatter(
                x=forecast["Datetime"], y=forecast["Predicted_PJME_MW"],
                name="Trend", mode="lines", line=dict(color=CYAN, width=2)
            ))
            fig.update_layout(xaxis_title=None, yaxis_title="MW")
            st.plotly_chart(polish(fig, 220), use_container_width=True, config={"displayModeBar": False})

    with c3:
        with st.container(border=True):
            panel_title("FORECAST INSIGHTS", "Highlights from the saved forecast")
            st.markdown(f"""
            <div style="padding:7px 2px;border-bottom:1px solid #1B344A;">
              <span style="color:#D0DCE7;font-size:10px;">▲ PEAK DEMAND</span><br>
              <b style="font-size:19px;color:#F5B51B;">{peak_row['Predicted_PJME_MW']:,.0f} MW</b><br>
              <span style="font-size:10px;color:#C4D2DF;">{peak_row['Datetime'].strftime('%d %b %Y · %H:%M')}</span>
            </div>
            <div style="padding:7px 2px;border-bottom:1px solid #1B344A;">
              <span style="color:#D0DCE7;font-size:10px;">▼ MINIMUM DEMAND</span><br>
              <b style="font-size:19px;color:#21C66B;">{low_row['Predicted_PJME_MW']:,.0f} MW</b><br>
              <span style="font-size:10px;color:#C4D2DF;">{low_row['Datetime'].strftime('%d %b %Y · %H:%M')}</span>
            </div>
            <div style="padding:7px 2px;">
              <span style="color:#D0DCE7;font-size:10px;">◎ FORECAST RANGE</span><br>
              <b style="font-size:17px;color:#E8F1FA;">{peak_row['Predicted_PJME_MW']-low_row['Predicted_PJME_MW']:,.0f} MW</b>
            </div>
            """, unsafe_allow_html=True)

    # Bottom wide historical vs predicted
    with st.container(border=True):
        panel_title("HISTORICAL ACTUAL VS FUTURE PREDICTION", "Final 48 historical observations followed by the 24 predicted observations")
        hist_recent = historical.tail(48)[["Datetime", "PJME_MW"]].rename(columns={"PJME_MW": "Historical"})
        pred = forecast[["Datetime", "Predicted_PJME_MW"]].rename(columns={"Predicted_PJME_MW": "Predicted"})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_recent["Datetime"], y=hist_recent["Historical"],
                                 mode="lines", name="Historical actual",
                                 line=dict(color=CYAN, width=2.2)))
        fig.add_trace(go.Scatter(x=pred["Datetime"], y=pred["Predicted"],
                                 mode="lines+markers", name="Forecast",
                                 line=dict(color=GREEN, width=2.5, dash="solid"),
                                 marker=dict(size=5)))
        fig.add_vline(x=pred["Datetime"].iloc[0], line_dash="dash", line_color=YELLOW,
                      annotation_text="Forecast begins", annotation_position="top")
        fig.update_layout(xaxis_title="Date and time", yaxis_title="Electricity demand (MW)")
        st.plotly_chart(polish(fig, 285), use_container_width=True, config={"displayModeBar": False})

elif nav == "Historical":
    st.markdown('<div class="section-title">HISTORICAL CONSUMPTION ANALYTICS</div>', unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    min_date, max_date = historical["Datetime"].min().date(), historical["Datetime"].max().date()
    with d1:
        start = st.date_input("From date", value=max(min_date, max_date - pd.Timedelta(days=30)), min_value=min_date, max_value=max_date)
    with d2:
        end = st.date_input("To date", value=max_date, min_value=min_date, max_value=max_date)
    if start > end:
        st.warning("The start date must be on or before the end date.")
    else:
        data = historical[historical["Datetime"].dt.date.between(start, end)]
        if data.empty:
            st.warning("No observations found for this date range.")
        else:
            a, b, c = st.columns(3)
            a.metric("Observations", f"{len(data):,}")
            b.metric("Average demand", f"{data['PJME_MW'].mean():,.1f} MW")
            c.metric("Peak demand", f"{data['PJME_MW'].max():,.1f} MW")
            fig = go.Figure(go.Scatter(x=data["Datetime"], y=data["PJME_MW"], mode="lines",
                                       line=dict(color=CYAN, width=1.5), name="Demand"))
            fig.update_layout(title="Electricity consumption over selected period", xaxis_title="Date", yaxis_title="MW")
            st.plotly_chart(polish(fig, 420, False), use_container_width=True)
            hour = data.assign(Hour=data["Datetime"].dt.hour).groupby("Hour")["PJME_MW"].mean()
            fig = go.Figure(go.Bar(x=hour.index, y=hour.values, marker_color=BLUE))
            fig.update_layout(title="Average demand by hour of day", xaxis_title="Hour", yaxis_title="Average MW")
            st.plotly_chart(polish(fig, 320, False), use_container_width=True)
            st.dataframe(data.tail(200), use_container_width=True, hide_index=True)

elif nav == "Model Performance":
    st.markdown('<div class="section-title">MODEL EVALUATION & COMPARISON</div>', unsafe_allow_html=True)
    st.caption("Evaluation metrics from the saved chronological test-set comparison.")
    metric_cols = st.columns(3)
    for col, metric, title, better in [
        (metric_cols[0], "MAE", "Mean Absolute Error (MW)", "lower"),
        (metric_cols[1], "RMSE", "Root Mean Squared Error (MW)", "lower"),
        (metric_cols[2], "R2 Score", "R² Score", "higher")
    ]:
        with col:
            with st.container(border=True):
                panel_title(title)
                fig = go.Figure(go.Bar(
                    x=comparison["Model"], y=comparison[metric],
                    marker_color=[BLUE, GREEN],
                    text=comparison[metric].round(4), textposition="outside"
                ))
                fig.update_layout(showlegend=False, yaxis_title=metric)
                st.plotly_chart(polish(fig, 280, False), use_container_width=True)
    st.dataframe(comparison.round(4), use_container_width=True, hide_index=True)
    st.success(f"Improved model MAE is {mae_reduction:.2f}% lower than the baseline MAE.")
    st.caption("R² is a variance-explained score, not a percentage accuracy measure.")

elif nav == "Forecast":
    st.markdown('<div class="section-title">24-OBSERVATION RECURSIVE FORECAST</div>', unsafe_allow_html=True)
    a, b, c, d = st.columns(4)
    a.metric("Maximum", f"{forecast['Predicted_PJME_MW'].max():,.1f} MW")
    b.metric("Minimum", f"{forecast['Predicted_PJME_MW'].min():,.1f} MW")
    c.metric("Average", f"{avg_fc:,.1f} MW")
    d.metric("Forecast observations", len(forecast))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast["Datetime"], y=forecast["Predicted_PJME_MW"],
                             mode="lines+markers", name="Predicted demand",
                             line=dict(color=GREEN, width=2.5), marker=dict(size=7),
                             fill="tozeroy", fillcolor="rgba(33,198,107,.08)"))
    fig.update_layout(title="Predicted electricity demand", xaxis_title="Date and time", yaxis_title="MW")
    st.plotly_chart(polish(fig, 420, False), use_container_width=True)
    st.dataframe(forecast.round(2), use_container_width=True, hide_index=True)
    st.download_button("⬇ Download forecast CSV", forecast.to_csv(index=False).encode("utf-8"),
                       "electricity_forecast.csv", "text/csv")

# -------------------------- Footer -------------------------

st.markdown("""
<div class="footerline">
  <b style="color:#1CB6F0;letter-spacing:1.5px;">ELECTRA</b> &nbsp; ENERGY INTELLIGENCE
  <br><span style="font-size:10px;">Machine Learning Powered Electricity Forecasting · Random Forest Regression · PJM Hourly Energy Dataset</span>
</div>
""", unsafe_allow_html=True)

st.caption("Data note: PJM historical data ends on 3 August 2018. The displayed 24-observation forecast continues from that final timestamp and is a historical demonstration, not live grid monitoring.")
