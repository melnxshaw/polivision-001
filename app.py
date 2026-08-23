import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from data_pipeline import consolidate, load_combined_clean_csv, ENRICHED_COLS, totals_series, category_mix
from forecasting import time_ordered_backtest, final_forecast

st.set_page_config(page_title="Polivision — Life Insurance Intelligence",
                    layout="wide", page_icon="🧭", initial_sidebar_state="expanded")

# ============================================================
#  DESIGN SYSTEM
#  Base: warm saffron-on-dark (from the earlier "Saffron" build)
#  Brand accent: a tricolor gradient (saffron -> warm white -> green),
#  evoking the Indian flag's palette WITHOUT reproducing the flag itself
#  (no stripes, no chakra) — used only as a refined text/accent gradient
#  so the app reads as a global, professional product, not a flag graphic.
# ============================================================
PALETTE = {
    "bg0": "#0E0B08", "bg1": "#161109", "surface": "rgba(255,255,255,0.035)",
    "surface_border": "rgba(255,196,102,0.14)",
    "amber": "#F5A623", "amber_deep": "#D4780C", "cream": "#FFE7BD",
    "text": "#FBF3E7", "text_dim": "#B8A88F",
    "good": "#7BC67E", "bad": "#E2685A",
}
# Brand gradient — vibrant multi-color accent for the Polivision wordmark & key CTAs
BRAND_GRADIENT = "linear-gradient(100deg, #6366F1 0%, #06B6D4 30%, #22C55E 55%, #F59E0B 78%, #EC4899 100%)"
# Working gradient for dashboard chrome (buttons, tab underline) — same family, less stripe-like
GRADIENT_MAIN = f"linear-gradient(120deg, {PALETTE['amber_deep']} 0%, {PALETTE['amber']} 55%, {PALETTE['cream']} 100%)"
PLOTLY_COLORWAY = [PALETTE["amber"], "#E2685A", "#7BC67E", "#D4780C",
                   "#FFE7BD", "#C97B3F", "#F2C57C", "#8C6A45"]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
.stApp {{
    background: radial-gradient(circle at 20% -10%, #241a0d 0%, {PALETTE['bg0']} 45%, #050403 100%);
    color: {PALETTE['text']};
}}
h2, h3 {{ font-family: 'Fraunces', serif !important; color: {PALETTE['cream']}; }}

/* Brand wordmark — tricolor gradient, used sparingly (logo/hero only) */
.brand-mark {{
    font-family: 'Fraunces', serif; font-weight: 700;
    background: {BRAND_GRADIENT}; -webkit-background-clip: text;
    -webkit-text-fill-color: transparent; background-clip: text;
}}
.brand-hairline {{
    height: 3px; width: 84px; border-radius: 3px; background: {BRAND_GRADIENT};
    margin: 10px 0 22px 0;
}}

/* Hero */
.hero-wrap {{ padding: 34px 10px 10px 10px; }}
.hero-headline {{
    font-family: 'Fraunces', serif; font-weight: 700;
    font-size: 3rem; line-height: 1.05; letter-spacing: -0.01em;
    background: {GRADIENT_MAIN}; -webkit-background-clip: text;
    -webkit-text-fill-color: transparent; background-clip: text;
    margin-bottom: 6px;
}}
.hero-sub {{ color: {PALETTE['text_dim']}; font-size: 1.05rem; margin-bottom: 18px; }}

/* Stat / feature cards */
.stat-card {{
    background: {PALETTE['surface']};
    border: 1px solid {PALETTE['surface_border']};
    border-radius: 14px; padding: 18px 20px;
    transition: border-color 0.2s ease, transform 0.2s ease;
    height: 100%;
}}
.stat-card:hover {{ border-color: {PALETTE['amber']}; transform: translateY(-2px); }}
.stat-label {{ color: {PALETTE['text_dim']}; font-size: 0.74rem; text-transform: uppercase;
               letter-spacing: 0.08em; font-weight: 600; margin-bottom: 6px; }}
.stat-value {{ font-family: 'Fraunces', serif; font-size: 1.9rem; font-weight: 700; color: {PALETTE['cream']}; }}
.stat-delta-pos {{ color: {PALETTE['good']}; font-size: 0.82rem; font-weight: 600; }}
.stat-delta-neg {{ color: {PALETTE['bad']}; font-size: 0.82rem; font-weight: 600; }}

.feature-card {{
    background: {PALETTE['surface']}; border: 1px solid {PALETTE['surface_border']};
    border-radius: 16px; padding: 22px 22px; height: 100%;
}}
.feature-title {{ font-family: 'Fraunces', serif; color: {PALETTE['cream']}; font-size: 1.15rem;
                   font-weight: 700; margin-bottom: 8px; }}
.feature-body {{ color: {PALETTE['text_dim']}; font-size: 0.92rem; line-height: 1.55; }}
.step-num {{
    display:inline-flex; align-items:center; justify-content:center;
    width: 30px; height: 30px; border-radius: 50%;
    background: {BRAND_GRADIENT}; color: #241a0d; font-weight: 700; font-size: 0.85rem;
    margin-bottom: 10px;
}}

.eyebrow {{
    display: inline-block; color: {PALETTE['amber']}; font-weight: 700;
    font-size: 0.76rem; letter-spacing: 0.1em; text-transform: uppercase;
    border-bottom: 2px solid {PALETTE['amber']}; padding-bottom: 3px; margin-bottom: 10px;
}}

.stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {PALETTE['surface_border']}; }}
.stTabs [data-baseweb="tab"] {{
    background: transparent; color: {PALETTE['text_dim']}; font-weight: 600; padding: 10px 18px;
}}
.stTabs [aria-selected="true"] {{
    color: {PALETTE['amber']} !important;
    border-bottom: 3px solid {PALETTE['amber']} !important;
}}

[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #1A1309, #0A0705);
    border-right: 1px solid {PALETTE['surface_border']};
}}
.stDownloadButton button, .stButton button {{
    background: {GRADIENT_MAIN} !important; color: #241a0d !important;
    font-weight: 700 !important; border: none !important; border-radius: 10px !important;
}}
[data-testid="stDataFrame"] {{ border: 1px solid {PALETTE['surface_border']}; border-radius: 10px; }}
hr {{ border-color: {PALETTE['surface_border']}; }}
</style>
""", unsafe_allow_html=True)


def stat_card(label, value, delta=None, positive=True):
    delta_html = ""
    if delta is not None:
        cls = "stat-delta-pos" if positive else "stat-delta-neg"
        arrow = "▲" if positive else "▼"
        delta_html = f'<div class="{cls}">{arrow} {delta}</div>'
    return f"""<div class="stat-card"><div class="stat-label">{label}</div>
    <div class="stat-value">{value}</div>{delta_html}</div>"""


def feature_card(title, body):
    return f"""<div class="feature-card"><div class="feature-title">{title}</div>
    <div class="feature-body">{body}</div></div>"""


def step_card(num, title, body):
    return f"""<div class="feature-card"><div class="step-num">{num}</div>
    <div class="feature-title">{title}</div><div class="feature-body">{body}</div></div>"""


def eyebrow(text):
    st.markdown(f'<span class="eyebrow">{text}</span>', unsafe_allow_html=True)


def style_fig(fig, height=420):
    fig.update_layout(
        height=height, colorway=PLOTLY_COLORWAY,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE["text"], family="Inter"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=40, l=10, r=10, b=10),
    )
    fig.update_xaxes(gridcolor="rgba(255,196,102,0.08)", zerolinecolor="rgba(255,196,102,0.08)")
    fig.update_yaxes(gridcolor="rgba(255,196,102,0.08)", zerolinecolor="rgba(255,196,102,0.08)")
    return fig


# ============================================================
#  PAGE 1 — INTRODUCTION
# ============================================================
def render_intro_page():
    st.markdown(f'<div class="brand-mark" style="font-size:2.6rem;">Polivision</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-hairline"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-headline" style="font-size:2.2rem;">See where insurance sales are headed.</div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Polivision turns scattered monthly life-insurance new-business reports '
        'into one clear, forecasting-ready view — trend, seasonality, category mix, and a forward look, '
        'all in one place.</div>', unsafe_allow_html=True)

    st.write("")
    eyebrow("What We Do")
    st.subheader("From raw monthly reports to a living forecast")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(feature_card(
            "Consolidate",
            "We collect and clean monthly life-insurance new-business data across insurers and "
            "premium categories into one continuous, analysis-ready timeline."
        ), unsafe_allow_html=True)
    with c2:
        st.markdown(feature_card(
            "Analyze",
            "We surface the patterns that matter — seasonal peaks, insurer market share shifts, "
            "and which premium categories are growing or shrinking."
        ), unsafe_allow_html=True)
    with c3:
        st.markdown(feature_card(
            "Forecast",
            "We project future premium and policy volumes with a confidence range, so planning "
            "is based on evidence rather than guesswork."
        ), unsafe_allow_html=True)

    st.write("")
    eyebrow("How We Do It")
    st.subheader("A transparent, time-aware pipeline")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.markdown(step_card("1", "Clean & Consolidate",
                               "Monthly reports are parsed, standardized, and merged into a single "
                               "insurer-by-category timeline."), unsafe_allow_html=True)
    with d2:
        st.markdown(step_card("2", "Explore",
                               "Trend, seasonality, market share, and category mix are visualized "
                               "so patterns are easy to see, not just compute."), unsafe_allow_html=True)
    with d3:
        st.markdown(step_card("3", "Validate",
                               "Every forecasting model is backtested on a strict time-ordered split — "
                               "never trained on future data — before it's trusted."), unsafe_allow_html=True)
    with d4:
        st.markdown(step_card("4", "Forecast",
                               "The best-performing model projects forward with a confidence "
                               "interval, not a false-precision single number."), unsafe_allow_html=True)

    st.write("")
    eyebrow("What You Get")
    st.subheader("A live, explorable intelligence dashboard")
    e1, e2 = st.columns(2)
    with e1:
        st.markdown(feature_card(
            "Full visibility",
            "Industry trend, seasonality, insurer market share, and premium-category mix — "
            "all in interactive, explorable charts, including a 3D view across insurer and time."
        ), unsafe_allow_html=True)
    with e2:
        st.markdown(feature_card(
            "Evidence, not guesses",
            "A transparent model comparison (MAE, RMSE, WAPE, Bias) shows exactly how accurate "
            "each forecasting approach really is, backed by real backtested numbers."
        ), unsafe_allow_html=True)

    st.write("")
    eyebrow("How We Help You")
    st.subheader("Built for decisions, not just charts")
    st.markdown(feature_card(
        "Plan with confidence",
        "Whether you're setting a monthly sales target, comparing insurer performance, or "
        "deciding where to focus a product push, Polivision gives you a data-backed starting "
        "point — a forecast with a known accuracy track record and a clear confidence range, "
        "instead of relying on last year's memory alone."
    ), unsafe_allow_html=True)

    st.write("")
    st.write("")
    _, mid, _ = st.columns([1, 1, 1])
    with mid:
        if st.button("Enter the Dashboard →", use_container_width=True):
            st.switch_page(dashboard_page_ref)
    st.caption("You can also jump straight to the Dashboard any time from the sidebar.")


# ============================================================
#  PAGE 2 — DASHBOARD (all analysis, unchanged functionality)
# ============================================================
def render_dashboard_page():
    st.sidebar.markdown("### 🟠 Data")

    @st.cache_data
    def get_base_data():
        return load_combined_clean_csv("combined_clean.csv")

    base_df = get_base_data()
    st.sidebar.success(f"Loaded {base_df['Month'].nunique()} months · {base_df['Insurer'].nunique()} insurers")

    extra_files = st.sidebar.file_uploader(
        "Add newer monthly files (optional)",
        type=["xls", "xlsx", "csv"], accept_multiple_files=True,
        help="Upload additional raw council .xls files or CSVs to extend the timeline.",
    )

    master_df = base_df
    if extra_files:
        files = {f.name: f.read() for f in extra_files}
        new_data = consolidate(files)
        if not new_data.empty:
            master_df = pd.concat([base_df, new_data], ignore_index=True)
            master_df = master_df.drop_duplicates(subset=["Month", "Insurer", "BusinessType"], keep="last")
            master_df = master_df.sort_values(["Month", "Insurer", "BusinessType"]).reset_index(drop=True)
            st.sidebar.success(f"Extended to {master_df['Month'].nunique()} months total")
        else:
            st.sidebar.error("Couldn't parse the uploaded file(s) — check the format.")

    totals_df = totals_series(master_df)
    mix_df = category_mix(master_df)
    totals_df["MonthDate"] = pd.to_datetime(totals_df["Month"], format="%Y-%m")
    mix_df["MonthDate"] = pd.to_datetime(mix_df["Month"], format="%Y-%m")

    industry = (totals_df.groupby("MonthDate", as_index=False)
                .agg(Premium_Rs_Crore=("Premium_Rs_Crore", "sum"),
                     No_of_Policies=("No_of_Policies", "sum")))
    industry = industry.sort_values("MonthDate").set_index("MonthDate").asfreq("MS").reset_index()
    industry[["Premium_Rs_Crore", "No_of_Policies"]] = industry[["Premium_Rs_Crore", "No_of_Policies"]].interpolate()
    n_months = industry["MonthDate"].nunique()

    latest = industry.iloc[-1]
    prev_year_row = industry[industry["MonthDate"] == latest["MonthDate"] - pd.DateOffset(years=1)]
    yoy = ((latest["Premium_Rs_Crore"] / prev_year_row["Premium_Rs_Crore"].values[0]) - 1) * 100 \
        if len(prev_year_row) else None

    st.markdown('<div class="hero-wrap">', unsafe_allow_html=True)
    st.markdown(f'<span class="brand-mark" style="font-size:1.5rem;">Polivision</span> '
                f'<span style="color:{PALETTE["text_dim"]};font-size:1rem;"> · Dashboard</span>',
                unsafe_allow_html=True)
    st.markdown('<div class="hero-headline">Grow Your Insight.</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Life Insurance New-Business Sales — Trend, Category Mix &amp; Forecasting</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(stat_card("Latest Month", latest["MonthDate"].strftime("%b %Y")), unsafe_allow_html=True)
    with c2:
        st.markdown(stat_card("Industry NBP (₹ cr)", f"{latest['Premium_Rs_Crore']:,.0f}",
                              f"{abs(yoy):.1f}% YoY" if yoy is not None else None,
                              positive=(yoy is None or yoy >= 0)), unsafe_allow_html=True)
    with c3:
        st.markdown(stat_card("Policies Issued", f"{latest['No_of_Policies']:,.0f}"), unsafe_allow_html=True)
    with c4:
        st.markdown(stat_card("Active Insurers", f"{totals_df['Insurer'].nunique()}"), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    tabs = st.tabs(["Overview", "Category Mix", "3D Landscape", "Backtesting", "Forecast",
                    "Insights", "FAQ / Assumptions"])

    # ---------------- TAB 1: Overview ----------------
    with tabs[0]:
        eyebrow("Sales Trend")
        st.subheader("Industry Trend Over Time")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=industry["MonthDate"], y=industry["Premium_Rs_Crore"],
                                  mode="lines", name="Industry NBP", line=dict(width=3, color=PALETTE["amber"]),
                                  fill="tozeroy", fillcolor="rgba(245,166,35,0.12)"))
        st.plotly_chart(style_fig(fig), use_container_width=True)

        colA, colB = st.columns(2)
        with colA:
            eyebrow("Seasonality")
            st.subheader("Average by Calendar Month")
            seasonal = industry.copy()
            seasonal["MonthNum"] = seasonal["MonthDate"].dt.month
            seasonal_avg = seasonal.groupby("MonthNum")["Premium_Rs_Crore"].mean().reset_index()
            seasonal_avg["MonthName"] = pd.to_datetime(seasonal_avg["MonthNum"], format="%m").dt.strftime("%b")
            fig2 = px.bar(seasonal_avg, x="MonthName", y="Premium_Rs_Crore")
            fig2.update_traces(marker_color=PALETTE["amber"])
            st.plotly_chart(style_fig(fig2, 350), use_container_width=True)
        with colB:
            eyebrow("Market Share")
            st.subheader("Insurer Split (latest month)")
            latest_month = totals_df["MonthDate"].max()
            share = totals_df[totals_df["MonthDate"] == latest_month].nlargest(10, "Premium_Rs_Crore")
            fig3 = px.pie(share, names="Insurer", values="Premium_Rs_Crore", hole=0.55)
            st.plotly_chart(style_fig(fig3, 350), use_container_width=True)

        eyebrow("Leaderboard")
        st.subheader("Top 8 Insurers — Contribution Over Time")
        top_insurers = (totals_df.groupby("Insurer")["Premium_Rs_Crore"].sum()
                         .sort_values(ascending=False).head(8).index.tolist())
        trend_df = totals_df[totals_df["Insurer"].isin(top_insurers)]
        fig4 = px.area(trend_df, x="MonthDate", y="Premium_Rs_Crore", color="Insurer")
        st.plotly_chart(style_fig(fig4, 440), use_container_width=True)

    # ---------------- TAB 2: Category Mix ----------------
    with tabs[1]:
        eyebrow("Product Mix")
        st.subheader("Premium Category Breakdown")
        st.caption("Individual vs Group, Single vs Non-Single premium.")

        cat_totals = (mix_df.groupby(["MonthDate", "BusinessType"], as_index=False)
                      .agg(Premium_Rs_Crore=("Premium_Current_Month", "sum")))
        fig_mix = px.area(cat_totals, x="MonthDate", y="Premium_Rs_Crore", color="BusinessType")
        st.plotly_chart(style_fig(fig_mix, 440), use_container_width=True)

        latest_month_mix = mix_df["MonthDate"].max()
        latest_mix = (mix_df[mix_df["MonthDate"] == latest_month_mix]
                      .groupby("BusinessType", as_index=False)
                      .agg(Premium_Rs_Crore=("Premium_Current_Month", "sum")))
        col1, col2 = st.columns([1, 1.3])
        with col1:
            fig_pie = px.pie(latest_mix, names="BusinessType", values="Premium_Rs_Crore",
                              title=f"Split — {latest_month_mix.strftime('%b %Y')}", hole=0.5)
            st.plotly_chart(style_fig(fig_pie, 380), use_container_width=True)
        with col2:
            eyebrow("Vault Table")
            st.subheader("Insurer × Category")
            pivot = (mix_df[mix_df["MonthDate"] == latest_month_mix]
                     .pivot_table(index="Insurer", columns="BusinessType",
                                  values="Premium_Current_Month", aggfunc="sum", fill_value=0))
            st.dataframe(pivot.style.format("{:,.1f}"), use_container_width=True, height=380)

    # ---------------- TAB 3: 3D Landscape ----------------
    with tabs[2]:
        eyebrow("Explore")
        st.subheader("3D Sales Landscape — Insurer × Time × Premium")
        st.caption("Rotate/zoom to explore how each insurer's monthly premium has evolved.")

        top_n = st.slider("Number of top insurers to plot", 5, 15, 10)
        top_list = (totals_df.groupby("Insurer")["Premium_Rs_Crore"].sum()
                    .sort_values(ascending=False).head(top_n).index.tolist())
        plot_df = totals_df[totals_df["Insurer"].isin(top_list)].copy()
        plot_df["InsurerShort"] = plot_df["Insurer"].str.slice(0, 22)
        insurer_order = plot_df.groupby("InsurerShort")["Premium_Rs_Crore"].sum().sort_values(ascending=False).index.tolist()
        insurer_idx = {name: i for i, name in enumerate(insurer_order)}
        plot_df["InsurerIdx"] = plot_df["InsurerShort"].map(insurer_idx)
        plot_df["TimeIdx"] = (plot_df["MonthDate"].dt.year - plot_df["MonthDate"].dt.year.min()) * 12 + plot_df["MonthDate"].dt.month

        fig3d = go.Figure(data=[go.Scatter3d(
            x=plot_df["TimeIdx"], y=plot_df["InsurerIdx"], z=plot_df["Premium_Rs_Crore"],
            mode="markers",
            marker=dict(
                size=4, color=plot_df["Premium_Rs_Crore"],
                colorscale=[[0, PALETTE["amber_deep"]], [0.5, PALETTE["amber"]], [1, PALETTE["cream"]]],
                opacity=0.85, colorbar=dict(title="₹ crore")
            ),
            text=plot_df["Insurer"] + "<br>" + plot_df["Month"],
            hovertemplate="%{text}<br>Premium: ₹%{z:,.0f} cr<extra></extra>",
        )])
        fig3d.update_layout(
            height=620, paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=PALETTE["text"]),
            scene=dict(
                xaxis=dict(title="Time (months since start)", backgroundcolor="rgba(0,0,0,0)",
                           gridcolor="rgba(255,196,102,0.10)"),
                yaxis=dict(title="Insurer", tickvals=list(insurer_idx.values()),
                           ticktext=list(insurer_idx.keys()), backgroundcolor="rgba(0,0,0,0)",
                           gridcolor="rgba(255,196,102,0.10)"),
                zaxis=dict(title="Premium (₹ crore)", backgroundcolor="rgba(0,0,0,0)",
                           gridcolor="rgba(255,196,102,0.10)"),
            ),
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig3d, use_container_width=True)

    # ---------------- TAB 4: Backtesting ----------------
    with tabs[3]:
        eyebrow("Model Validation")
        st.subheader("Time-Ordered Backtest — Candidate Model Comparison")
        st.caption("Rolling-origin backtest: each fold trains only on past months and forecasts forward — no shuffling, no leakage.")

        target_metric = st.selectbox("Backtest target", ["Premium_Rs_Crore", "No_of_Policies"], key="bt_target")
        horizon = st.slider("Forecast horizon per fold (months)", 1, 6, 3)
        max_min_train = max(6, n_months - horizon - 1)
        min_train = st.slider("Minimum training window (months)", 6, max_min_train, min(18, max_min_train))

        series = industry.set_index("MonthDate")[target_metric]
        with st.spinner("Running rolling-origin backtest across all candidate models..."):
            bt_results = time_ordered_backtest(series, horizon=horizon, min_train=min_train)

        if bt_results.empty:
            st.warning("Not enough history for this horizon/training-window combination.")
        else:
            st.dataframe(bt_results.style.highlight_min(subset=["MAE", "RMSE", "WAPE_%"], color="#2b230f")
                         .highlight_min(subset=["Bias_%"], color="#332417"),
                         use_container_width=True)
            best_model = bt_results.iloc[0]["Model"]
            st.success(f"✅ Best performing model by WAPE: **{best_model}**")
            st.caption("MAE / RMSE / WAPE — lower is better. Bias — closer to 0% is better.")

    # ---------------- TAB 5: Forecast ----------------
    with tabs[4]:
        eyebrow("Forward-Looking")
        st.subheader("Forecast")

        fc_target = st.selectbox("Forecast target", ["Premium_Rs_Crore", "No_of_Policies"], key="fc_target")
        fc_horizon = st.slider("Forecast horizon (months ahead)", 1, 12, 6)

        series = industry.set_index("MonthDate")[fc_target]
        with st.spinner("Fitting SARIMA and Holt-Winters models..."):
            fc = final_forecast(series, horizon=fc_horizon)

        hist_df = industry[["MonthDate", fc_target]].rename(columns={fc_target: "Actual"})
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=hist_df["MonthDate"], y=hist_df["Actual"],
                                   mode="lines", name="Actual", line=dict(color=PALETTE["amber"], width=2.5)))
        future_dates = pd.to_datetime(fc["Month"], format="%Y-%m")
        fig5.add_trace(go.Scatter(x=future_dates, y=fc["Forecast_SARIMA"],
                                   mode="lines+markers", name="Forecast (SARIMA)",
                                   line=dict(color=PALETTE["bad"], dash="dash", width=2.5)))
        fig5.add_trace(go.Scatter(x=future_dates, y=fc["Forecast_HoltWinters"],
                                   mode="lines", name="Forecast (Holt-Winters)",
                                   line=dict(color=PALETTE["good"], dash="dot", width=2)))
        fig5.add_trace(go.Scatter(x=future_dates, y=fc["Forecast_XGBoost"],
                                   mode="lines", name="Forecast (XGBoost)",
                                   line=dict(color="#6366F1", dash="dashdot", width=2)))
        fig5.add_trace(go.Scatter(x=future_dates, y=fc["Forecast_Prophet"],
                                   mode="lines", name="Forecast (Prophet)",
                                   line=dict(color="#EC4899", dash="longdash", width=2)))
        fig5.add_trace(go.Scatter(
            x=list(future_dates) + list(future_dates[::-1]),
            y=list(fc["Upper_80"]) + list(fc["Lower_80"][::-1]),
            fill="toself", fillcolor="rgba(245,166,35,0.14)", line=dict(width=0),
            name="80% Confidence Interval (SARIMA)", showlegend=True))
        st.plotly_chart(style_fig(fig5, 460), use_container_width=True)
        st.caption("SARIMA carries the shaded 80% confidence band; the other models are shown "
                   "as point forecasts for comparison.")

        st.dataframe(fc.style.format({
            "Forecast_SARIMA": "{:,.0f}", "Lower_80": "{:,.0f}", "Upper_80": "{:,.0f}",
            "Forecast_HoltWinters": "{:,.0f}", "Forecast_XGBoost": "{:,.0f}",
            "Forecast_Prophet": "{:,.0f}"}), use_container_width=True)

        csv = fc.to_csv(index=False).encode()
        st.download_button("⬇️ Download forecast as CSV", csv, "nbp_forecast.csv", "text/csv")

    # ---------------- TAB 6: Insights ----------------
    with tabs[5]:
        eyebrow("Auto-Generated")
        st.subheader("Business Insights")
        bullets = []

        if n_months >= 24:
            last12 = industry.tail(12)
            prev12 = industry.iloc[-24:-12]
            yoy_growth = ((last12["Premium_Rs_Crore"].sum() / prev12["Premium_Rs_Crore"].sum()) - 1) * 100
            direction = "grew" if yoy_growth >= 0 else "declined"
            bullets.append(f"Trailing 12-month industry NBP **{direction} {abs(yoy_growth):.1f}%** YoY.")

        top_insurer_latest = (totals_df[totals_df["MonthDate"] == totals_df["MonthDate"].max()]
                               .sort_values("Premium_Rs_Crore", ascending=False).iloc[0])
        bullets.append(f"**{top_insurer_latest['Insurer']}** leads market share in the latest month "
                        f"with ₹{top_insurer_latest['Premium_Rs_Crore']:,.0f} crore.")

        if n_months >= 12:
            peak_month = (industry.assign(MonthNum=industry["MonthDate"].dt.month)
                          .groupby("MonthNum")["Premium_Rs_Crore"].mean().idxmax())
            peak_month_name = pd.to_datetime(str(peak_month), format="%m").strftime("%B")
            bullets.append(f"**{peak_month_name}** is consistently the strongest month for new business.")

        growing = (totals_df.groupby("Insurer").apply(
            lambda d: d.sort_values("MonthDate")["Premium_Rs_Crore"].pct_change().mean() * 100
        ).dropna().sort_values(ascending=False))
        if len(growing) > 0:
            bullets.append(f"**{growing.index[0]}** shows the strongest average month-on-month growth momentum.")

        latest_month_for_mix = mix_df["MonthDate"].max()
        top_category = (mix_df[mix_df["MonthDate"] == latest_month_for_mix]
                         .groupby("BusinessType")["Premium_Current_Month"].sum().idxmax())
        bullets.append(f"**{top_category}** is the largest premium category in the latest month.")

        for b in bullets:
            st.markdown(f"- {b}")

        eyebrow("Leaderboard")
        st.subheader("Growth Ranking (avg MoM % change)")
        st.dataframe(growing.reset_index().rename(columns={0: "Avg_MoM_Growth_%"}).round(2),
                     use_container_width=True)

    # ---------------- TAB 7: FAQ / Assumptions ----------------
    with tabs[6]:
        eyebrow("Documentation")
        st.subheader("Frequently Asked Questions")
        with st.expander("What data powers this dashboard?"):
            st.markdown("""Real Life Insurance Council "Detailed New Business Performance" data,
            cleaned via a custom preprocessing pipeline — 67 months (Jan 2021–Jul 2026), 27 insurers,
            5 premium categories per insurer plus a Total row.""")
        with st.expander("Why are PRIVATE TOTAL / GRAND TOTAL rows excluded?"):
            st.markdown("""They are sums across insurers, not a single insurer — keeping them would
            double-count against the real per-insurer rows.""")
        with st.expander("How is the backtest kept honest (no data leakage)?"):
            st.markdown("""A rolling-origin, time-ordered split is used — every fold trains only on
            past months and forecasts forward, never trained on future data.""")
        with st.expander("Why WAPE and Bias instead of plain MAPE?"):
            st.markdown("""Individual insurer-month values can be zero or very small, which makes
            MAPE unstable. WAPE (error as % of total volume) and Bias (over/under-forecast direction)
            are more robust for this kind of data.""")
        with st.expander("Can I add newer months myself?"):
            st.markdown("""Yes — upload raw council `.xls` files or CSVs in the sidebar any time;
            they merge with the built-in dataset and deduplicate automatically.""")
        with st.expander("Why do the model lists include both statistical and ML models?"):
            st.markdown("""The backtest compares classical statistical models (Naive, Seasonal Naive,
            Moving Average, Holt-Winters, SARIMA) against machine-learning approaches (XGBoost on
            lag/rolling features, Prophet) side by side on the same rolling-origin folds — so the
            choice of "which kind of model wins" is decided by evidence, not assumption.""")
        with st.expander("What are the known limitations?"):
            st.markdown("""Forecasts assume the recent trend/seasonality broadly continues — they do
            not anticipate regulatory changes, new product launches, or macro shocks. Forecasts are
            directional planning inputs, not guarantees.""")


# ============================================================
#  NAVIGATION — sidebar page switcher
# ============================================================
intro_page_ref = st.Page(render_intro_page, title="Introduction", icon="🧭", default=True)
dashboard_page_ref = st.Page(render_dashboard_page, title="Dashboard", icon="📊")

nav = st.navigation([intro_page_ref, dashboard_page_ref])
nav.run()
