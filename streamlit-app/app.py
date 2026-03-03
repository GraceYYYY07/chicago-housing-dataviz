"""
app.py
Housing Density, Income, and Uninsured Rates in Chicago’s South Side

This dashboard is descriptive (not causal). It helps identify where uninsured
residents are concentrated and how priority lists change under different
thresholds and weighting scenarios.

Data input (no data is fabricated):
- data/derived-data/merged_tract.geojson
  Produced by preprocessing.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import altair as alt
import folium
import streamlit as st
from streamlit_folium import st_folium
from pathlib import Path


# =============================================================================
# Theme constants (Chicago-style)
# =============================================================================
CHI_BLUE = "#0B4F9E"
CHI_RED = "#E4002B"
LIGHT_GRAY = "#F5F7FA"
TEXT_DARK = "#111111"


# =============================================================================
# Page config
# =============================================================================
st.set_page_config(
    page_title="Housing Density, Income, and Uninsured Rates in Chicago’s South Side",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Styling (simple, human, less “template”)
# =============================================================================
st.markdown(f"""
<style>
    .stApp {{ background-color: #ffffff; }}
    .main .block-container {{ padding: 1.4rem 2rem; }}

    h1 {{
        color: {TEXT_DARK} !important;
        border-bottom: 3px solid {CHI_RED};
        padding-bottom: 10px;
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
    }}

    h2, h3 {{
        color: {TEXT_DARK} !important;
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
    }}

    [data-testid="metric-container"] {{
        background-color: {LIGHT_GRAY};
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 14px;
    }}

    [data-testid="metric-container"] [data-testid="stMetricValue"] {{
        color: {CHI_BLUE} !important;
        font-size: 1.6rem;
        font-weight: 700;
    }}

    [data-testid="stSidebar"] {{
        background-color: #FAFBFD;
    }}

    [data-testid="stSidebar"] * {{
        color: {TEXT_DARK} !important;
    }}

    .stCaption {{
        color: #4B5563 !important;
    }}

    /* Make tables feel less “stock” */
    .stDataFrame {{
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        overflow: hidden;
    }}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Data loading
# =============================================================================
@st.cache_data
def load_data() -> gpd.GeoDataFrame:
    ROOT = Path(__file__).resolve().parents[1]
    DATA_PATH = ROOT / "data" / "derived-data" / "merged_tract.geojson"
    gdf = gpd.read_file(DATA_PATH)

    numeric_cols = [
        "addr_count", "med_hh_inc", "pct_no_hlt", "pop_0_17", "tot_pop",
        "area_sqkm", "addr_per_sqkm", "pop_per_sqkm"
    ]
    for c in numeric_cols:
        if c in gdf.columns:
            gdf[c] = pd.to_numeric(gdf[c], errors="coerce")

    # pct_no_hlt likely in [0,1]
    gdf["uninsured_pct"] = gdf["pct_no_hlt"] * 100 if "pct_no_hlt" in gdf.columns else np.nan

    # Estimated uninsured residents (KEY policy metric)
    # If tot_pop missing, stays NaN (no fabrication)
    if "tot_pop" in gdf.columns:
        gdf["est_uninsured"] = (gdf["uninsured_pct"] / 100.0) * gdf["tot_pop"]
    else:
        gdf["est_uninsured"] = np.nan

    if "pop_0_17" in gdf.columns and "tot_pop" in gdf.columns:
        gdf["child_pct"] = gdf["pop_0_17"] / gdf["tot_pop"].replace(0, np.nan) * 100
    else:
        gdf["child_pct"] = np.nan

    # Friendly tract label
    if "NAMELSAD" in gdf.columns and gdf["NAMELSAD"].notna().any():
        gdf["tract_name"] = gdf["NAMELSAD"].astype(str)
    elif "TRACTCE" in gdf.columns:
        gdf["tract_name"] = "Census Tract " + gdf["TRACTCE"].astype(str)
    else:
        gdf["tract_name"] = gdf["GEOID"].astype(str)

    # Income quartile label (fallback if missing)
    if "income_quartile" not in gdf.columns:
        if "med_hh_inc" in gdf.columns and gdf["med_hh_inc"].notna().any():
            gdf["income_quartile"] = pd.qcut(
                gdf["med_hh_inc"], q=4,
                labels=["Q1 (Lowest)", "Q2", "Q3", "Q4 (Highest)"]
            )
        else:
            gdf["income_quartile"] = "undefined"

    return gdf


def winsorize(s: pd.Series, lo: float = 0.01, hi: float = 0.99) -> pd.Series:
    s = s.replace([np.inf, -np.inf], np.nan)
    if s.dropna().empty:
        return s
    a, b = s.quantile(lo), s.quantile(hi)
    return s.clip(a, b)


# =============================================================================
# Load data
# =============================================================================
gdf = load_data()


# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    st.markdown("## Settings")
    st.caption("Adjust filters and see how the priority list changes.")

    st.markdown("---")

    # Map metric
    map_metric_options = {
        "Uninsured rate (%)": "uninsured_pct",
        "Median household income ($)": "med_hh_inc",
        "Address density (addresses per sq km)": "addr_per_sqkm",
    }
    map_metric_label = st.selectbox("Map layer", list(map_metric_options.keys()), index=0)
    map_metric = map_metric_options[map_metric_label]

    st.markdown("---")
    st.markdown("### Priority rules (thresholds)")

    st.caption(
        "We use **quantiles** because policy teams often need a transparent way to pick the "
        "**top X%** tracts when resources are limited. For example, a 0.75 density threshold "
        "means “top 25% densest tracts (within the current filtered set).”"
    )

    density_q = st.slider(
        "High density (quantile)",
        0.50, 0.95, 0.75, 0.05,
        help="Higher = stricter. 0.75 flags the top 25% densest tracts."
    )
    low_income_q = st.slider(
        "Low income (quantile)",
        0.05, 0.50, 0.25, 0.05,
        help="Lower = stricter. 0.25 flags the bottom 25% income tracts."
    )
    uninsured_q = st.slider(
        "High uninsured (quantile)",
        0.50, 0.95, 0.75, 0.05,
        help="Higher = stricter. 0.75 flags the top 25% uninsured-rate tracts."
    )

    st.markdown("---")
    st.markdown("### Scenario weighting (priority score)")

    st.caption(
        "Income tends to explain uninsured rates more strongly than density. "
        "So the default weights emphasize **uninsured severity** and **low income**, "
        "while density is used mainly for **reach efficiency** (where outreach can reach more people)."
    )

    w_uninsured = st.slider("Weight: uninsured severity", 0.0, 1.0, 0.50, 0.05)
    w_income = st.slider("Weight: low income", 0.0, 1.0, 0.35, 0.05)
    w_density = st.slider("Weight: reach efficiency (density)", 0.0, 1.0, 0.15, 0.05)

    st.markdown("---")
    st.markdown("### Plot options")

    clip_outliers = st.checkbox("Clip extreme values (1%–99%)", value=True)
    use_log_density = st.checkbox("Use log scale for density plots", value=True)

    st.markdown("---")
    with st.expander("Limitations (plain English)", expanded=False):
        st.write(
            "- This tool is **descriptive**, not causal.\n"
            "- ACS 5-year estimates can be noisy for small areas.\n"
            "- Address density is a **proxy** for residential concentration (not official housing units).\n"
            "- Some tracts may have very low/zero residential address points (e.g., industrial or park areas). "
            "We treat those as **missing** for density plots."
        )


# =============================================================================
# Filter dataset
# =============================================================================
filtered = gdf.copy()

if filtered.empty:
    st.title("Housing Density, Income, and Uninsured Rates in Chicago’s South Side")
    st.warning("No tracts match the current income filter. Please widen the income range.")
    st.stop()

# Quantile thresholds within filtered set
d_series = filtered["addr_per_sqkm"].replace([np.inf, -np.inf], np.nan).dropna()
i_series = filtered["med_hh_inc"].replace([np.inf, -np.inf], np.nan).dropna()
u_series = filtered["uninsured_pct"].replace([np.inf, -np.inf], np.nan).dropna()

d_th = d_series.quantile(density_q) if not d_series.empty else np.nan
i_th = i_series.quantile(low_income_q) if not i_series.empty else np.nan
u_th = u_series.quantile(uninsured_q) if not u_series.empty else np.nan

filtered["priority_flag"] = (
    (filtered["addr_per_sqkm"] >= d_th) &
    (filtered["med_hh_inc"] <= i_th) &
    (filtered["uninsured_pct"] >= u_th)
)

# =============================================================================
# Header (less academic, more action)
# =============================================================================
st.title("Housing Density, Income, and Uninsured Rates in Chicago’s South Side")
st.write(
   "This dashboard explores how housing density and neighborhood income relate to uninsured rates at the Census Tract level. It provides descriptive, tract-level evidence for policy discussion."
)

st.markdown("---")


# =============================================================================
# KPIs (add estimated uninsured)
# =============================================================================
total_pop = filtered["tot_pop"].replace([np.inf, -np.inf], np.nan).sum(skipna=True)
avg_income = filtered["med_hh_inc"].replace([np.inf, -np.inf], np.nan).mean()
avg_unins = filtered["uninsured_pct"].replace([np.inf, -np.inf], np.nan).mean()
avg_density = filtered["addr_per_sqkm"].replace([np.inf, -np.inf], np.nan).mean()

total_est_uninsured = filtered["est_uninsured"].replace([np.inf, -np.inf], np.nan).sum(skipna=True)

priority = filtered[filtered["priority_flag"]].copy()
priority_est_uninsured = priority["est_uninsured"].replace([np.inf, -np.inf], np.nan).sum(skipna=True)
priority_count = int(priority.shape[0])

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Avg median income", f"${avg_income:,.0f}")
c2.metric("Avg uninsured rate", f"{avg_unins:.1f}%")
c3.metric("Estimated uninsured (all tracts)", f"{total_est_uninsured:,.0f}")
c4.metric("Priority tracts", f"{priority_count:,}")
c5.metric("Estimated uninsured (priority tracts)", f"{priority_est_uninsured:,.0f}")

st.markdown("---")


# =============================================================================
tabs = st.tabs(["Map", "Relationships", "Priority list"])


# =============================================================================
# TAB 1: MAP
# =============================================================================
with tabs[0]:
    left, right = st.columns([3, 2])

    with left:
        st.subheader(f"Map: {map_metric_label}")

        center_lat = float(filtered.geometry.centroid.y.mean())
        center_lon = float(filtered.geometry.centroid.x.mean())
        m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="OpenStreetMap")

        # Choose a map palette aligned with blue/red
        # - Uninsured / estimated uninsured: Reds
        # - Income: Blues (reversed so higher income darker)
        # - Density: Blues (darker = more dense)
        if map_metric in ["uninsured_pct", "est_uninsured"]:
            fill_color = "Reds"
        elif map_metric == "med_hh_inc":
            fill_color = "Blues"
        else:
            fill_color = "Blues"

        folium.Choropleth(
            geo_data=filtered.__geo_interface__,
            data=filtered[["GEOID", map_metric]],
            columns=["GEOID", map_metric],
            key_on="feature.properties.GEOID",
            fill_color=fill_color,
            fill_opacity=0.75,
            line_opacity=0.2,
            nan_fill_color="#B0B7C3",
            legend_name=map_metric_label
        ).add_to(m)

        tooltip_fields = ["tract_name", "med_hh_inc", "uninsured_pct", "est_uninsured", "addr_per_sqkm", "tot_pop"]
        tooltip_aliases = ["Tract:", "Median income ($):", "Uninsured (%):", "Est. uninsured:", "Address density (/km²):", "Population:"]

        folium.GeoJson(
            filtered.__geo_interface__,
            style_function=lambda x: {"fillOpacity": 0, "weight": 0.6, "color": "#4B5563"},
            tooltip=folium.GeoJsonTooltip(
                fields=tooltip_fields,
                aliases=tooltip_aliases,
                localize=True,
                sticky=True,
                labels=True,
                style="background-color:#ffffff;color:#111111;font-size:12px;border:1px solid #CBD5E1;border-radius:8px;padding:10px;"
            )
        ).add_to(m)

        # Priority outlines in Chicago red
        if not priority.empty:
            folium.GeoJson(
                priority.__geo_interface__,
                style_function=lambda x: {"fillOpacity": 0, "weight": 3, "color": CHI_RED},
                name="Priority tracts"
            ).add_to(m)

        map_key = f"map_{map_metric}_{density_q}_{low_income_q}_{uninsured_q}"
        st_folium(m, height=560, use_container_width=True, key=map_key)

    with right:
        st.subheader("How to read this")
        st.write(
            "- Each polygon is a Census Tract.\n"
            "- Orange/red shading = higher uninsured (or higher estimated uninsured, depending on layer).\n"
            f"- Red outlines = tracts flagged by your current thresholds.\n\n"
            "**What density means here:** density is used to locate where outreach may reach more residents per visit/site. "
            "It is not assumed to cause uninsurance."
        )

        # Simple distribution chart
        dist = filtered[map_metric].replace([np.inf, -np.inf], np.nan).dropna()
        dist_df = pd.DataFrame({"value": dist})

        hist = alt.Chart(dist_df).mark_bar(opacity=0.9, color=CHI_BLUE).encode(
            x=alt.X("value:Q", bin=alt.Bin(maxbins=25), title=map_metric_label),
            y=alt.Y("count():Q", title="Number of tracts")
        ).properties(height=220).configure_axis(
            labelColor=TEXT_DARK, titleColor=TEXT_DARK, gridColor="#E5E7EB"
        ).configure_view(stroke=None)

        st.altair_chart(hist, use_container_width=True)


# =============================================================================
# TAB 2: RELATIONSHIPS
# =============================================================================
with tabs[1]:
    st.subheader("Relationships (visual evidence)")

    df = filtered[[
        "tract_name", "income_quartile", "med_hh_inc", "uninsured_pct",
        "addr_per_sqkm", "tot_pop", "est_uninsured"
    ]].replace([np.inf, -np.inf], np.nan).copy()

    # Clip for readability
    if clip_outliers:
        df["med_hh_inc_c"] = winsorize(df["med_hh_inc"])
        df["uninsured_pct_c"] = winsorize(df["uninsured_pct"])
        df["addr_per_sqkm_c"] = winsorize(df["addr_per_sqkm"])
    else:
        df["med_hh_inc_c"] = df["med_hh_inc"]
        df["uninsured_pct_c"] = df["uninsured_pct"]
        df["addr_per_sqkm_c"] = df["addr_per_sqkm"]

    # --- A) Income vs uninsured
    st.write("**A) Income vs uninsured rate** (income tends to be the strongest separator)")

    df_a = df.dropna(subset=["med_hh_inc_c", "uninsured_pct_c", "tot_pop"]).copy()
    if df_a.empty:
        st.warning("Not enough valid data to plot income vs uninsured under current filters.")
    else:
        # Blue gradient for income quartiles (Chicago-ish)
        income_scale = alt.Scale(
            domain=["Q1 (Lowest)", "Q2", "Q3", "Q4 (Highest)"],
            range=["#A6C8FF", "#6FA8FF", CHI_BLUE, "#08306B"]
        )

        base_a = alt.Chart(df_a).mark_circle(opacity=0.75).encode(
            x=alt.X("med_hh_inc_c:Q", title="Median household income ($)"),
            y=alt.Y("uninsured_pct_c:Q", title="Uninsured rate (%)"),
            color=alt.Color("income_quartile:N", scale=income_scale, title="Income quartile"),
            size=alt.Size("tot_pop:Q", title="Population", scale=alt.Scale(range=[20, 280])),
            tooltip=[
                alt.Tooltip("tract_name:N", title="Tract"),
                alt.Tooltip("med_hh_inc:Q", title="Income ($)", format=",.0f"),
                alt.Tooltip("uninsured_pct:Q", title="Uninsured (%)", format=".1f"),
                alt.Tooltip("est_uninsured:Q", title="Est. uninsured", format=",.0f"),
            ]
        ).properties(height=340)

        reg_a = base_a.transform_regression("med_hh_inc_c", "uninsured_pct_c").mark_line(
            color=CHI_RED, strokeWidth=2, opacity=0.8
        )

        st.altair_chart(
            (base_a + reg_a)
            .configure_axis(labelColor=TEXT_DARK, titleColor=TEXT_DARK, gridColor="#E5E7EB")
            .configure_view(stroke=None)
            .configure_legend(labelColor=TEXT_DARK, titleColor=TEXT_DARK),
            use_container_width=True
        )

    st.markdown("")

    # --- B) Density vs uninsured (treat 0 as missing)
    st.write("**B) Address density vs uninsured rate** (density is mainly a reach / targeting lens)")

    df_b = df.copy()
    # Treat 0 or negative density as missing for plotting (common for industrial/park-like tracts)
    zero_or_missing = df_b["addr_per_sqkm_c"].fillna(0) <= 0
    excluded_n = int(zero_or_missing.sum())
    df_b.loc[zero_or_missing, "addr_per_sqkm_c"] = np.nan

    # Drop missing for this plot only
    df_b = df_b.dropna(subset=["addr_per_sqkm_c", "uninsured_pct_c", "tot_pop"])

    if excluded_n > 0:
        st.caption(f"Note: {excluded_n} tracts have zero/very-low address density and are treated as missing in this plot.")

    if df_b.empty:
        st.warning("Not enough valid density data to plot under current filters.")
    else:
        if use_log_density:
            df_b["x_density"] = np.log1p(df_b["addr_per_sqkm_c"])
            x_title = "Address density (log1p)"
        else:
            df_b["x_density"] = df_b["addr_per_sqkm_c"]
            x_title = "Address density (addresses per sq km)"

        base_b = alt.Chart(df_b).mark_circle(opacity=0.75).encode(
            x=alt.X("x_density:Q", title=x_title),
            y=alt.Y("uninsured_pct_c:Q", title="Uninsured rate (%)"),
            color=alt.Color("income_quartile:N", scale=income_scale, title="Income quartile"),
            size=alt.Size("tot_pop:Q", title="Population", scale=alt.Scale(range=[20, 280])),
            tooltip=[
                alt.Tooltip("tract_name:N", title="Tract"),
                alt.Tooltip("addr_per_sqkm:Q", title="Density (/km²)", format=",.0f"),
                alt.Tooltip("uninsured_pct:Q", title="Uninsured (%)", format=".1f"),
                alt.Tooltip("est_uninsured:Q", title="Est. uninsured", format=",.0f"),
            ]
        ).properties(height=340)

        reg_b = base_b.transform_regression("x_density", "uninsured_pct_c").mark_line(
            color=CHI_RED, strokeWidth=2, opacity=0.8
        )

        st.altair_chart(
            (base_b + reg_b)
            .configure_axis(labelColor=TEXT_DARK, titleColor=TEXT_DARK, gridColor="#E5E7EB")
            .configure_view(stroke=None)
            .configure_legend(labelColor=TEXT_DARK, titleColor=TEXT_DARK),
            use_container_width=True
        )

    # =============================================================================
    # Summary chart 1: Income quartile -> average uninsured rate
    # =============================================================================
    st.markdown("---")
    st.subheader("Summary: uninsured rate by income quartile")

    df_bar = filtered[["income_quartile", "uninsured_pct", "tot_pop"]].replace([np.inf, -np.inf], np.nan).dropna()
    df_bar = df_bar[df_bar["income_quartile"].astype(str).isin(["Q1 (Lowest)", "Q2", "Q3", "Q4 (Highest)"])]

    if df_bar.empty:
        st.warning("Not enough valid data to summarize uninsured rates by income quartile.")
    else:
        # Weighted mean uninsured by population (more policy-relevant)
        grouped = (
            df_bar.assign(w=df_bar["tot_pop"])
            .groupby("income_quartile", as_index=False)
            .apply(lambda g: pd.Series({
                "avg_uninsured_wt": np.average(g["uninsured_pct"], weights=g["w"])
            }))
            .reset_index()
        )

        # Ensure quartile ordering
        order = ["Q1 (Lowest)", "Q2", "Q3", "Q4 (Highest)"]
        grouped["income_quartile"] = pd.Categorical(grouped["income_quartile"], categories=order, ordered=True)
        grouped = grouped.sort_values("income_quartile")

        bar1 = alt.Chart(grouped).mark_bar(color=CHI_BLUE).encode(
            x=alt.X("income_quartile:N", title="Income quartile (within filtered tracts)", sort=order),
            y=alt.Y("avg_uninsured_wt:Q", title="Population-weighted uninsured rate (%)"),
            tooltip=[
                alt.Tooltip("income_quartile:N", title="Quartile"),
                alt.Tooltip("avg_uninsured_wt:Q", title="Avg uninsured (%)", format=".2f"),
            ]
        ).properties(height=320)

        labels1 = alt.Chart(grouped).mark_text(
            dy=-8, color=TEXT_DARK
        ).encode(
            x=alt.X("income_quartile:N", sort=order),
            y=alt.Y("avg_uninsured_wt:Q"),
            text=alt.Text("avg_uninsured_wt:Q", format=".1f")
        )

        st.altair_chart(
            (bar1 + labels1)
            .configure_axis(labelColor=TEXT_DARK, titleColor=TEXT_DARK, gridColor="#E5E7EB")
            .configure_view(stroke=None),
            use_container_width=True
        )

        st.caption(
             "Lower-income quartiles tend to have higher uninsured rates on average, "
             "though the relationship is not perfectly monotonic across quartiles."
        )


# =============================================================================
# TAB 3: PRIORITY LIST
# =============================================================================
with tabs[2]:
    st.subheader("Priority list (actionable tracts)")

    st.write(
        "These tracts meet your current threshold rules (high density + low income + high uninsured). "
        "The score below ranks flagged tracts under your **scenario weights**."
    )

    pr = filtered[filtered["priority_flag"]].copy()
    if pr.empty:
        st.warning("No tracts meet the current thresholds. Adjust the quantiles in the sidebar.")
        st.stop()

    dfp = pr[["tract_name", "med_hh_inc", "uninsured_pct", "addr_per_sqkm", "tot_pop", "est_uninsured"]].copy()
    dfp = dfp.replace([np.inf, -np.inf], np.nan).dropna(subset=["med_hh_inc", "uninsured_pct", "tot_pop"])

    # z-scores (within priority set) for transparent ranking
    def z(s: pd.Series) -> pd.Series:
        s = s.astype(float)
        sd = s.std(ddof=0)
        if np.isclose(sd, 0) or np.isnan(sd):
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - s.mean()) / sd

    # Density: treat non-positive as missing for scoring; fill missing density z as 0 (neutral)
    dens = dfp["addr_per_sqkm"].copy()
    dens = dens.where(dens > 0, np.nan)

    dfp["z_uninsured"] = z(dfp["uninsured_pct"])
    dfp["z_income_low"] = z(-dfp["med_hh_inc"])  # lower income -> higher score
    dfp["z_density"] = z(dens).fillna(0)

    # Normalize weights (avoid “sum != 1” confusion)
    w_sum = w_uninsured + w_income + w_density
    if w_sum == 0:
        w_uninsured_n, w_income_n, w_density_n = 0.50, 0.35, 0.15
    else:
        w_uninsured_n, w_income_n, w_density_n = w_uninsured / w_sum, w_income / w_sum, w_density / w_sum

    dfp["priority_score"] = (
        w_uninsured_n * dfp["z_uninsured"] +
        w_income_n * dfp["z_income_low"] +
        w_density_n * dfp["z_density"]
    )

    # Add a policy-friendly field: est uninsured
    # (already exists) but ensure readable
    dfp["est_uninsured"] = dfp["est_uninsured"].round(0)

    # Top table
    top = dfp.sort_values("priority_score", ascending=False).head(15).copy()
    top = top.rename(columns={
        "tract_name": "Tract",
        "med_hh_inc": "Median income ($)",
        "uninsured_pct": "Uninsured (%)",
        "addr_per_sqkm": "Address density (/km²)",
        "tot_pop": "Population",
        "est_uninsured": "Estimated uninsured (count)",
        "priority_score": "Priority score"
    })

    top["Median income ($)"] = top["Median income ($)"].round(0).astype(int)
    top["Population"] = top["Population"].round(0).astype(int)
    top["Uninsured (%)"] = top["Uninsured (%)"].round(2)
    top["Address density (/km²)"] = top["Address density (/km²)"].round(2)
    top["Priority score"] = top["Priority score"].round(3)
    top["Estimated uninsured (count)"] = top["Estimated uninsured (count)"].fillna(0).astype(int)

    st.caption(
        f"Current normalized weights: uninsured={w_uninsured_n:.2f}, low income={w_income_n:.2f}, density={w_density_n:.2f}. "
        "Higher density weight prioritizes reach efficiency; higher uninsured/income weights prioritize severity/need."
    )

    st.dataframe(top, use_container_width=True, hide_index=True)

    # Download full list
    export_df = dfp.sort_values("priority_score", ascending=False).rename(columns={
        "tract_name": "Tract",
        "med_hh_inc": "Median income ($)",
        "uninsured_pct": "Uninsured (%)",
        "addr_per_sqkm": "Address density (/km²)",
        "tot_pop": "Population",
        "est_uninsured": "Estimated uninsured (count)",
        "priority_score": "Priority score"
    })

    csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download flagged tracts (CSV)",
        data=csv,
        file_name="priority_tracts.csv",
        mime="text/csv"
    )