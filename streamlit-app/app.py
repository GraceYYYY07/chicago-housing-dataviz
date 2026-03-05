"""
app.py
Housing Density, Income, and Uninsured Rates in Chicago's South Side

This dashboard is descriptive (not causal). It helps identify where uninsured
residents are concentrated and how priority lists change under different
thresholds and weighting scenarios.

Data input:
- data/derived-data/merged_tract.geojson  (produced by preprocessing.py)
  Address data source: cleaned_F_ADD1_not0.csv (~46,899 residential buildings)
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
CHI_BLUE  = "#0B4F9E"
CHI_RED   = "#E4002B"
LIGHT_GRAY = "#F5F7FA"
TEXT_DARK  = "#111111"


# =============================================================================
# Page config
# =============================================================================
st.set_page_config(
    page_title="Housing Density, Income, and Uninsured Rates in Chicago's South Side",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

    .stCaption {{ color: #4B5563 !important; }}

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
        "addr_count", "unit_count",
        "med_hh_inc", "pct_no_hlt", "pop_0_17", "tot_pop",
        "area_sqkm", "addr_per_sqkm", "unit_per_sqkm", "pop_per_sqkm"
    ]
    for c in numeric_cols:
        if c in gdf.columns:
            gdf[c] = pd.to_numeric(gdf[c], errors="coerce")

    # pct_no_hlt is stored in [0, 1]
    gdf["uninsured_pct"] = (
        gdf["pct_no_hlt"] * 100 if "pct_no_hlt" in gdf.columns else np.nan
    )

    # Estimated uninsured residents
    if "tot_pop" in gdf.columns:
        gdf["est_uninsured"] = (gdf["uninsured_pct"] / 100.0) * gdf["tot_pop"]
    else:
        gdf["est_uninsured"] = np.nan

    if "pop_0_17" in gdf.columns and "tot_pop" in gdf.columns:
        gdf["child_pct"] = (
            gdf["pop_0_17"] / gdf["tot_pop"].replace(0, np.nan) * 100
        )
    else:
        gdf["child_pct"] = np.nan

    # Friendly tract label
    if "NAMELSAD" in gdf.columns and gdf["NAMELSAD"].notna().any():
        gdf["tract_name"] = gdf["NAMELSAD"].astype(str)
    elif "TRACTCE" in gdf.columns:
        gdf["tract_name"] = "Census Tract " + gdf["TRACTCE"].astype(str)
    else:
        gdf["tract_name"] = gdf["GEOID"].astype(str)

    # Income quartile (fallback if missing)
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
HAS_UNITS = "unit_per_sqkm" in gdf.columns and gdf["unit_per_sqkm"].notna().any()


# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    st.markdown("## Settings")

    # Map metric selector — include unit density if available
    map_metric_options = {
        "Uninsured rate (%)": "uninsured_pct",
        "Median household income ($)": "med_hh_inc",
        "Address density (buildings / km²)": "addr_per_sqkm",
    }
    if HAS_UNITS:
        map_metric_options["Housing unit density (units / km²)"] = "unit_per_sqkm"

    map_metric_label = st.selectbox(
        "Map layer", list(map_metric_options.keys()), index=0
    )
    map_metric = map_metric_options[map_metric_label]

    st.markdown("---")
    st.markdown("### Priority rules (thresholds)")

    density_q = st.slider(
        "High density (quantile)", 0.50, 0.95, 0.75, 0.05,
        help="Higher = stricter. 0.75 flags the top 25% densest tracts."
    )
    low_income_q = st.slider(
        "Low income (quantile)", 0.05, 0.50, 0.25, 0.05,
        help="Lower = stricter. 0.25 flags the bottom 25% income tracts."
    )
    uninsured_q = st.slider(
        "High uninsured (quantile)", 0.50, 0.95, 0.75, 0.05,
        help="Higher = stricter. 0.75 flags the top 25% uninsured-rate tracts."
    )

    st.markdown("---")
    st.markdown("### Scenario weighting (priority score)")

    w_uninsured = st.slider("Weight: uninsured severity",        0.0, 1.0, 0.50, 0.05)
    w_income    = st.slider("Weight: low income",                0.0, 1.0, 0.35, 0.05)
    w_density   = st.slider("Weight: reach efficiency (density)", 0.0, 1.0, 0.15, 0.05)

    st.markdown("---")
    st.markdown("### Plot options")

    clip_outliers   = st.checkbox("Clip extreme values (1%–99%)", value=True)
    use_log_density = st.checkbox("Use log scale for density plots", value=True)


# =============================================================================
# Filter dataset
# =============================================================================
filtered = gdf.copy()

if filtered.empty:
    st.title("Housing Density, Income, and Uninsured Rates in Chicago's South Side")
    st.warning("No tracts match the current filter. Please widen the settings.")
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
    (filtered["med_hh_inc"]    <= i_th) &
    (filtered["uninsured_pct"] >= u_th)
)


# =============================================================================
# Header
# =============================================================================
st.title("Housing Density, Income, and Uninsured Rates in Chicago's South Side")
st.write(
    "This dashboard explores how housing density and neighborhood income relate to "
    "uninsured rates at the Census Tract level. "
    "It provides descriptive, tract-level evidence for policy discussion."
)
st.markdown("---")


# =============================================================================
# KPIs
# =============================================================================
total_pop            = filtered["tot_pop"].replace([np.inf, -np.inf], np.nan).sum(skipna=True)
avg_income           = filtered["med_hh_inc"].replace([np.inf, -np.inf], np.nan).mean()
avg_unins            = filtered["uninsured_pct"].replace([np.inf, -np.inf], np.nan).mean()
avg_density          = filtered["addr_per_sqkm"].replace([np.inf, -np.inf], np.nan).mean()
total_est_uninsured  = filtered["est_uninsured"].replace([np.inf, -np.inf], np.nan).sum(skipna=True)

priority               = filtered[filtered["priority_flag"]].copy()
priority_est_uninsured = priority["est_uninsured"].replace([np.inf, -np.inf], np.nan).sum(skipna=True)
priority_count         = int(priority.shape[0])

# Show total buildings KPI from the new dataset
total_buildings = int(filtered["addr_count"].replace([np.inf, -np.inf], np.nan).sum(skipna=True))

cols = st.columns(5)
cols[0].metric("Avg median income",                  f"${avg_income:,.0f}")
cols[1].metric("Avg uninsured rate",                 f"{avg_unins:.1f}%")
cols[2].metric("Estimated uninsured (all tracts)",   f"{total_est_uninsured:,.0f}")
cols[3].metric("Priority tracts",                    f"{priority_count:,}")
cols[4].metric("Estimated uninsured (priority tracts)", f"{priority_est_uninsured:,.0f}")

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
        m = folium.Map(location=[center_lat, center_lon],
                       zoom_start=11, tiles="OpenStreetMap")

        if map_metric in ["uninsured_pct", "est_uninsured"]:
            fill_color = "Reds"
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

        tooltip_fields   = ["tract_name", "med_hh_inc", "uninsured_pct",
                             "est_uninsured", "addr_per_sqkm", "tot_pop"]
        tooltip_aliases  = ["Tract:", "Median income ($):", "Uninsured (%):",
                             "Est. uninsured:", "Address density (/km²):", "Population:"]

        if HAS_UNITS:
            tooltip_fields.insert(5, "unit_per_sqkm")
            tooltip_aliases.insert(5, "Unit density (/km²):")

        folium.GeoJson(
            filtered.__geo_interface__,
            style_function=lambda x: {
                "fillOpacity": 0, "weight": 0.6, "color": "#4B5563"
            },
            tooltip=folium.GeoJsonTooltip(
                fields=tooltip_fields,
                aliases=tooltip_aliases,
                localize=True,
                sticky=True,
                labels=True,
                style=(
                    "background-color:#ffffff;color:#111111;"
                    "font-size:12px;border:1px solid #CBD5E1;"
                    "border-radius:8px;padding:10px;"
                )
            )
        ).add_to(m)

        if not priority.empty:
            folium.GeoJson(
                priority.__geo_interface__,
                style_function=lambda x: {
                    "fillOpacity": 0, "weight": 3, "color": CHI_RED
                },
                name="Priority tracts"
            ).add_to(m)

        map_key = f"map_{map_metric}_{density_q}_{low_income_q}_{uninsured_q}"
        st_folium(m, height=560, use_container_width=True, key=map_key)

    with right:
        dist = filtered[map_metric].replace([np.inf, -np.inf], np.nan).dropna()
        dist_df = pd.DataFrame({"value": dist})

        hist = (
            alt.Chart(dist_df)
            .mark_bar(opacity=0.9, color=CHI_BLUE)
            .encode(
                x=alt.X("value:Q", bin=alt.Bin(maxbins=25), title=map_metric_label),
                y=alt.Y("count():Q", title="Number of tracts")
            )
            .properties(height=220)
            .configure_axis(
                labelColor=TEXT_DARK, titleColor=TEXT_DARK, gridColor="#E5E7EB"
            )
            .configure_view(stroke=None)
        )
        st.altair_chart(hist, use_container_width=True)


# =============================================================================
# TAB 2: RELATIONSHIPS
# =============================================================================
with tabs[1]:
    st.subheader("Relationships (visual evidence)")

    base_cols = [
        "tract_name", "income_quartile", "med_hh_inc",
        "uninsured_pct", "addr_per_sqkm", "tot_pop", "est_uninsured"
    ]
    if HAS_UNITS:
        base_cols.append("unit_per_sqkm")

    df = (
        filtered[base_cols]
        .replace([np.inf, -np.inf], np.nan)
        .copy()
    )

    if clip_outliers:
        df["med_hh_inc_c"]    = winsorize(df["med_hh_inc"])
        df["uninsured_pct_c"] = winsorize(df["uninsured_pct"])
        df["addr_per_sqkm_c"] = winsorize(df["addr_per_sqkm"])
    else:
        df["med_hh_inc_c"]    = df["med_hh_inc"]
        df["uninsured_pct_c"] = df["uninsured_pct"]
        df["addr_per_sqkm_c"] = df["addr_per_sqkm"]

    income_scale = alt.Scale(
        domain=["Q1 (Lowest)", "Q2", "Q3", "Q4 (Highest)"],
        range=["#A6C8FF", "#6FA8FF", CHI_BLUE, "#08306B"]
    )

    # --- A) Income vs uninsured
    st.write("**A) Income vs uninsured rate** (income tends to be the strongest separator)")

    df_a = df.dropna(subset=["med_hh_inc_c", "uninsured_pct_c", "tot_pop"]).copy()
    if df_a.empty:
        st.warning("Not enough valid data to plot income vs uninsured under current filters.")
    else:
        base_a = (
            alt.Chart(df_a)
            .mark_circle(opacity=0.75)
            .encode(
                x=alt.X("med_hh_inc_c:Q", title="Median household income ($)"),
                y=alt.Y("uninsured_pct_c:Q", title="Uninsured rate (%)"),
                color=alt.Color("income_quartile:N", scale=income_scale,
                                title="Income quartile"),
                size=alt.Size("tot_pop:Q", title="Population",
                              scale=alt.Scale(range=[20, 280])),
                tooltip=[
                    alt.Tooltip("tract_name:N",    title="Tract"),
                    alt.Tooltip("med_hh_inc:Q",    title="Income ($)",        format=",.0f"),
                    alt.Tooltip("uninsured_pct:Q", title="Uninsured (%)",     format=".1f"),
                    alt.Tooltip("est_uninsured:Q", title="Est. uninsured",    format=",.0f"),
                ]
            )
            .properties(height=340)
        )
        reg_a = (
            base_a
            .transform_regression("med_hh_inc_c", "uninsured_pct_c")
            .mark_line(color=CHI_RED, strokeWidth=2, opacity=0.8)
        )
        st.altair_chart(
            (base_a + reg_a)
            .configure_axis(labelColor=TEXT_DARK, titleColor=TEXT_DARK,
                            gridColor="#E5E7EB")
            .configure_view(stroke=None)
            .configure_legend(labelColor=TEXT_DARK, titleColor=TEXT_DARK),
            use_container_width=True
        )

    st.markdown("")

    # --- B) Address density vs uninsured
    st.write(
        "**B) Address density vs uninsured rate** "
        "(density = buildings per km² from city address database)"
    )

    df_b = df.copy()
    zero_or_missing = df_b["addr_per_sqkm_c"].fillna(0) <= 0
    excluded_n = int(zero_or_missing.sum())
    df_b.loc[zero_or_missing, "addr_per_sqkm_c"] = np.nan
    df_b = df_b.dropna(subset=["addr_per_sqkm_c", "uninsured_pct_c", "tot_pop"])

    if excluded_n > 0:
        st.write(
            f"Note: {excluded_n} tracts have zero/very-low address density "
            "and are treated as missing in this plot."
        )

    if df_b.empty:
        st.warning("Not enough valid density data to plot under current filters.")
    else:
        if use_log_density:
            df_b["x_density"] = np.log1p(df_b["addr_per_sqkm_c"])
            x_title = "Address density (log1p, buildings/km²)"
        else:
            df_b["x_density"] = df_b["addr_per_sqkm_c"]
            x_title = "Address density (buildings per km²)"

        tt_b = [
            alt.Tooltip("tract_name:N",      title="Tract"),
            alt.Tooltip("addr_per_sqkm:Q",   title="Density (/km²)",    format=",.1f"),
            alt.Tooltip("uninsured_pct:Q",   title="Uninsured (%)",     format=".1f"),
            alt.Tooltip("est_uninsured:Q",   title="Est. uninsured",    format=",.0f"),
        ]
        if HAS_UNITS:
            tt_b.insert(2, alt.Tooltip("unit_per_sqkm:Q",
                                        title="Unit density (/km²)", format=",.1f"))

        base_b = (
            alt.Chart(df_b)
            .mark_circle(opacity=0.75)
            .encode(
                x=alt.X("x_density:Q", title=x_title),
                y=alt.Y("uninsured_pct_c:Q", title="Uninsured rate (%)"),
                color=alt.Color("income_quartile:N", scale=income_scale,
                                title="Income quartile"),
                size=alt.Size("tot_pop:Q", title="Population",
                              scale=alt.Scale(range=[20, 280])),
                tooltip=tt_b
            )
            .properties(height=340)
        )
        reg_b = (
            base_b
            .transform_regression("x_density", "uninsured_pct_c")
            .mark_line(color=CHI_RED, strokeWidth=2, opacity=0.8)
        )
        st.altair_chart(
            (base_b + reg_b)
            .configure_axis(labelColor=TEXT_DARK, titleColor=TEXT_DARK,
                            gridColor="#E5E7EB")
            .configure_view(stroke=None)
            .configure_legend(labelColor=TEXT_DARK, titleColor=TEXT_DARK),
            use_container_width=True
        )




# =============================================================================
# TAB 3: PRIORITY LIST
# =============================================================================
with tabs[2]:
    st.subheader("Priority list (actionable tracts)")
    st.write(
        "These tracts meet your current threshold rules "
        "(high density + low income + high uninsured). "
        "The score below ranks flagged tracts under your **scenario weights**."
    )

    pr = filtered[filtered["priority_flag"]].copy()
    if pr.empty:
        st.warning("No tracts meet the current thresholds. "
                   "Adjust the quantiles in the sidebar.")
        st.stop()

    show_cols = [
        "tract_name", "med_hh_inc", "uninsured_pct",
        "addr_per_sqkm", "tot_pop", "est_uninsured"
    ]
    if HAS_UNITS:
        show_cols.insert(4, "unit_per_sqkm")

    dfp = pr[show_cols].copy()
    dfp = dfp.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["med_hh_inc", "uninsured_pct", "tot_pop"]
    )

    def z(s: pd.Series) -> pd.Series:
        s = s.astype(float)
        sd = s.std(ddof=0)
        if np.isclose(sd, 0) or np.isnan(sd):
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - s.mean()) / sd

    dens = dfp["addr_per_sqkm"].copy()
    dens = dens.where(dens > 0, np.nan)

    dfp["z_uninsured"]  = z(dfp["uninsured_pct"])
    dfp["z_income_low"] = z(-dfp["med_hh_inc"])   # lower income → higher score
    dfp["z_density"]    = z(dens).fillna(0)

    w_sum = w_uninsured + w_income + w_density
    if w_sum == 0:
        w_u, w_i, w_d = 0.50, 0.35, 0.15
    else:
        w_u, w_i, w_d = w_uninsured / w_sum, w_income / w_sum, w_density / w_sum

    dfp["priority_score"] = (
        w_u * dfp["z_uninsured"] +
        w_i * dfp["z_income_low"] +
        w_d * dfp["z_density"]
    )

    dfp["est_uninsured"] = dfp["est_uninsured"].round(0)

    rename_map = {
        "tract_name":    "Tract",
        "med_hh_inc":    "Median income ($)",
        "uninsured_pct": "Uninsured (%)",
        "addr_per_sqkm": "Address density (/km²)",
        "tot_pop":       "Population",
        "est_uninsured": "Estimated uninsured (count)",
        "priority_score":"Priority score",
    }
    if HAS_UNITS:
        rename_map["unit_per_sqkm"] = "Unit density (/km²)"

    top = (
        dfp.sort_values("priority_score", ascending=False)
        .head(15)
        .copy()
        .rename(columns=rename_map)
    )

    top["Median income ($)"]         = top["Median income ($)"].round(0).astype(int)
    top["Population"]                = top["Population"].round(0).astype(int)
    top["Uninsured (%)"]             = top["Uninsured (%)"].round(2)
    top["Address density (/km²)"]    = top["Address density (/km²)"].round(2)
    top["Priority score"]            = top["Priority score"].round(3)
    top["Estimated uninsured (count)"] = (
        top["Estimated uninsured (count)"].fillna(0).astype(int)
    )
    if HAS_UNITS and "Unit density (/km²)" in top.columns:
        top["Unit density (/km²)"] = top["Unit density (/km²)"].round(2)

    st.caption(
        f"Current normalized weights: uninsured={w_u:.2f}, "
        f"low income={w_i:.2f}, density={w_d:.2f}. "
        "Higher density weight prioritizes reach efficiency; "
        "higher uninsured/income weights prioritize severity/need."
    )
    st.dataframe(top, use_container_width=True, hide_index=True)

    # Download full list
    export_df = (
        dfp.sort_values("priority_score", ascending=False)
        .rename(columns=rename_map)
    )
    csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download flagged tracts (CSV)",
        data=csv,
        file_name="priority_tracts.csv",
        mime="text/csv"
    )
