"""
figures.py

Generate two static, publication-style figures for the project:

Figure 1 (PNG)
    A choropleth map of median household income by Census Tract, with
    address-count contour lines overlaid (derived from tract centroids).

Figure 2 (HTML)
    An interactive scatter plot (Altair): address density vs. uninsured rate,
    colored by income quartile and sized by total population.

Outputs (written to derived-data/):
    - figure1_choropleth.png
    - figure2_scatter.html

How to run:
    python figures.py
"""

import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

import altair as alt
from scipy.interpolate import griddata

# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DERIVED_DIR = DATA_DIR / "derived-data"

DATA_PATH = DERIVED_DIR / "merged_tract.geojson"
OUT_DIR = DERIVED_DIR


def _require_columns(df: pd.DataFrame, cols: list, name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} is missing required columns: {missing}")


def make_figure_1(gdf: gpd.GeoDataFrame, out_dir: Path) -> str:
    """
    Figure 1: Choropleth of median household income + address-count contours.
    Uses addr_count (building count per tract) for contour lines.
    Highlight: high address count + low income tracts.
    """
    _require_columns(
        gdf,
        ["geometry", "med_hh_inc", "addr_count"],
        "GeoDataFrame for Figure 1",
    )

    print("\n[Figure 1] Building choropleth + contours...")

    gdf_plot = gdf.copy().to_crs(epsg=3857)

    vmin = gdf_plot["med_hh_inc"].quantile(0.05)
    vmax = gdf_plot["med_hh_inc"].quantile(0.95)

    fig, ax = plt.subplots(1, 1, figsize=(10.5, 12))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    gdf_plot.plot(
        column="med_hh_inc",
        cmap="RdYlBu",
        linewidth=0.25,
        edgecolor="#c7c7c7",
        legend=False,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        missing_kwds={"color": "#efefef"},
    )

    # Contours from tracts that have address data
    # With the new larger dataset, use higher contour levels
    density_tracts = gdf_plot[gdf_plot["addr_count"] > 0].copy()
    if len(density_tracts) > 10:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            density_tracts["cx"] = density_tracts.geometry.centroid.x
            density_tracts["cy"] = density_tracts.geometry.centroid.y

        x = density_tracts["cx"].to_numpy()
        y = density_tracts["cy"].to_numpy()
        z = density_tracts["addr_count"].to_numpy(dtype=float)

        xi = np.linspace(x.min(), x.max(), 200)
        yi = np.linspace(y.min(), y.max(), 200)
        xi, yi = np.meshgrid(xi, yi)

        zi = griddata((x, y), z, (xi, yi), method="cubic")

        # Levels updated for the larger dataset (~46k buildings across South Side)
        p25 = float(np.nanpercentile(z, 25))
        p50 = float(np.nanpercentile(z, 50))
        p75 = float(np.nanpercentile(z, 75))
        p90 = float(np.nanpercentile(z, 90))
        p95 = float(np.nanpercentile(z, 95))
        levels = sorted(set([round(v) for v in [p25, p50, p75, p90, p95] if v > 0]))

        contour = ax.contour(
            xi, yi, zi,
            levels=levels,
            colors=["#3b3b3b"],
            alpha=0.5,
            linewidths=[0.8, 1.0, 1.1, 1.3, 1.6],
        )
        ax.clabel(contour, inline=True, fontsize=8, fmt="%d", colors="#3b3b3b")

    # Highlight: high address count + low income
    addr_hi = float(gdf_plot["addr_count"].quantile(0.75))
    inc_lo  = float(gdf_plot["med_hh_inc"].quantile(0.25))
    highlight = gdf_plot[
        (gdf_plot["addr_count"] >= addr_hi) &
        (gdf_plot["med_hh_inc"]  <= inc_lo)
    ].copy()
    if len(highlight) > 0:
        highlight.plot(ax=ax, color="none", edgecolor="#FF6B35", linewidth=2.0)

    # Colorbar
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(cmap="RdYlBu", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, shrink=0.75)
    cbar.set_label("Median household income ($)", fontsize=11)

    legend_elements = [
        Line2D([0], [0], color="#3b3b3b", alpha=0.6, linewidth=1.2,
               label="Address-count contour lines"),
        Line2D([0], [0], color="#FF6B35", linewidth=2.0,
               label="High address count & low income (top 25% / bottom 25%)"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", frameon=True)

    ax.set_title(
        "Chicago South Side\n"
        "Median Household Income by Census Tract (with Address-Count Contours)",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.set_axis_off()

    plt.tight_layout()
    out_path = out_dir / "figure1_choropleth.png"
    plt.savefig(str(out_path), dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"  ✓ Saved: {out_path}")
    return str(out_path)


def make_figure_2(gdf: gpd.GeoDataFrame, out_dir: Path) -> str:
    """
    Figure 2: Altair interactive scatter (address density vs uninsured rate).
    Uses addr_per_sqkm (buildings per km²) for x-axis — more comparable
    across tracts of different sizes than raw addr_count.
    """
    required = ["addr_per_sqkm", "pct_no_hlt", "tot_pop",
                "med_hh_inc", "income_quartile", "GEOID"]
    _require_columns(gdf, required, "GeoDataFrame for Figure 2")

    print("\n[Figure 2] Building Altair scatter...")

    df = gdf[gdf["addr_per_sqkm"] > 0].copy()

    df["addr_per_sqkm"]  = df["addr_per_sqkm"].astype(float)
    df["uninsured_pct"]  = df["pct_no_hlt"].astype(float) * 100.0
    df["tot_pop"]        = df["tot_pop"].astype(float)
    df["med_hh_inc"]     = df["med_hh_inc"].astype(float)

    # Also expose unit_per_sqkm in tooltip if available
    has_unit = "unit_per_sqkm" in df.columns
    if has_unit:
        df["unit_per_sqkm"] = df["unit_per_sqkm"].astype(float)

    if "geometry" in df.columns:
        df = df.drop(columns=["geometry"])

    # Quartile label mapping (handle both English and legacy Chinese labels)
    mapping = {
        "Q1低收入":   "Q1 (Lowest)",
        "Q2中低收入": "Q2",
        "Q3中高收入": "Q3",
        "Q4高收入":   "Q4 (Highest)",
    }
    df["income_quartile_en"] = (
        df["income_quartile"].map(mapping).fillna(df["income_quartile"])
    )

    domain = ["Q1 (Lowest)", "Q2", "Q3", "Q4 (Highest)"]
    color_scale = alt.Scale(
        domain=domain,
        range=["#d73027", "#fc8d59", "#91bfdb", "#4575b4"],
    )

    tooltip = [
        alt.Tooltip("GEOID:N",            title="Tract GEOID"),
        alt.Tooltip("addr_per_sqkm:Q",    title="Buildings / km²",  format=",.1f"),
        alt.Tooltip("uninsured_pct:Q",    title="Uninsured (%)",     format=".1f"),
        alt.Tooltip("med_hh_inc:Q",       title="Median income ($)", format=",.0f"),
        alt.Tooltip("tot_pop:Q",          title="Population",        format=","),
        alt.Tooltip("income_quartile_en:N", title="Income quartile"),
    ]
    if has_unit:
        tooltip.insert(2, alt.Tooltip("unit_per_sqkm:Q",
                                       title="Housing units / km²", format=",.1f"))

    base = (
        alt.Chart(df)
        .mark_circle(opacity=0.75, stroke="#ffffff", strokeWidth=0.5)
        .encode(
            x=alt.X(
                "addr_per_sqkm:Q",
                title="Address density (buildings per km²)",
                scale=alt.Scale(zero=False),
            ),
            y=alt.Y(
                "uninsured_pct:Q",
                title="Uninsured rate (%)",
            ),
            size=alt.Size(
                "tot_pop:Q",
                title="Total population",
                scale=alt.Scale(range=[30, 500]),
                legend=alt.Legend(title="Population", orient="bottom-right"),
            ),
            color=alt.Color(
                "income_quartile_en:N",
                title="Income quartile",
                scale=color_scale,
                legend=alt.Legend(title="Income quartile", orient="top-right"),
            ),
            tooltip=tooltip,
        )
        .properties(
            width=700,
            height=420,
            title=alt.TitleParams(
                text="Address Density and Uninsured Rate by Census Tract (Chicago South Side)",
                subtitle=(
                    "Point size = population; color = income quartile. "
                    "Dashed line shows overall linear trend."
                ),
                fontSize=14,
            ),
        )
    )

    reg = (
        base
        .transform_regression("addr_per_sqkm", "uninsured_pct")
        .mark_line(color="#111111", strokeWidth=2, strokeDash=[5, 3], opacity=0.7)
    )

    loess = (
        base
        .transform_loess(
            "addr_per_sqkm", "uninsured_pct",
            groupby=["income_quartile_en"],
            bandwidth=0.5,
        )
        .mark_line(strokeWidth=1.5, opacity=0.35)
        .encode(color=alt.Color("income_quartile_en:N",
                                scale=color_scale, legend=None))
    )

    chart = (
        (base + reg + loess)
        .configure(background="white")
        .configure_axis(
            labelColor="#111111", titleColor="#111111",
            gridColor="#e6e6e6",  domainColor="#999999", tickColor="#999999",
        )
        .configure_legend(
            labelColor="#111111", titleColor="#111111",
            fillColor="white",    strokeColor="#dddddd",
        )
        .configure_title(color="#111111", subtitleColor="#444444")
        .configure_view(stroke=None)
    )

    out_path = out_dir / "figure2_scatter.html"
    chart.save(str(out_path))
    print(f"  ✓ Saved: {out_path}")
    return str(out_path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading merged tract dataset...")
    gdf = gpd.read_file(DATA_PATH)
    print(f"  Tracts: {len(gdf):,}")
    if "addr_count" in gdf.columns:
        print(f"  Tracts with addresses: {(gdf['addr_count'] > 0).sum():,}")
    if "unit_count" in gdf.columns:
        print(f"  Total housing units:   {gdf['unit_count'].sum():,.0f}")

    fig1 = make_figure_1(gdf, OUT_DIR)
    fig2 = make_figure_2(gdf, OUT_DIR)

    print("\nDone.")
    print(f"Figure 1: {fig1}")
    print(f"Figure 2: {fig2}")


if __name__ == "__main__":
    main()
