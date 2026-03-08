"""
preprocessing.py
Chicago South Side: Residential Density & Community Health Inequality
Data preparation script

Data source:
- Address data: address_data.csv  (488,689 residential buildings)
  Columns used: lon, lat, NO_OF_UNIT
- Census tract shapefile: il_tract.shp  (Cook County subset)

Outputs:
- derived-data/merged_tract.geojson
  Contains tract-level ACS variables + address/unit counts + area + density.

How to run:
    python preprocessing.py
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path

# -----------------------------------------------------------------------------
# Robust project paths (works locally + Streamlit Cloud)
# This file lives in: <root>/code/preprocessing.py
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw-data"
DERIVED_DIR = DATA_DIR / "derived-data"
DERIVED_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = RAW_DIR / "address_data.csv"      # NEW address dataset
SHP_PATH = RAW_DIR / "il_tract.shp"
OUT_PATH = DERIVED_DIR / "merged_tract.geojson"


print("=" * 70)
print("Step 1) Load address CSV (lat/lon columns)")
print("=" * 70)

df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["lat", "lon"])
print(f"Total rows loaded: {len(df):,}")
print(f"Coordinate columns: lat [{df['lat'].min():.4f}, {df['lat'].max():.4f}], "
      f"lon [{df['lon'].min():.4f}, {df['lon'].max():.4f}]")

print("\n" + "=" * 70)
print("Step 2) Convert addresses to a GeoDataFrame (points)")
print("=" * 70)

geometry = [Point(lon, lat) for lon, lat in zip(df["lon"], df["lat"])]
addr_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

print("\n" + "=" * 70)
print("Step 3) Load census tract shapefile and subset Cook County (031)")
print("=" * 70)

tracts = gpd.read_file(SHP_PATH)
cook_tracts = tracts[tracts["COUNTYFP"] == "031"].copy()
cook_tracts = cook_tracts.to_crs("EPSG:4326")
print(f"Cook County tracts: {len(cook_tracts):,}")

print("\n" + "=" * 70)
print("Step 4) Spatial join: assign each address point to a tract")
print("=" * 70)

joined = gpd.sjoin(
    addr_gdf,
    cook_tracts[["GEOID", "geometry"]],
    how="left",
    predicate="within"
)

addr_counts = (
    joined
    .groupby("GEOID")
    .agg(
        addr_count=("BLDG_ID", "count"),
        unit_count=("NO_OF_UNIT", "sum"),
    )
    .reset_index()
)
print(f"Tracts with >=1 address: {len(addr_counts):,}")
print(f"Total building records matched: {addr_counts['addr_count'].sum():,}")
print(f"Total housing units matched:    {addr_counts['unit_count'].sum():,.0f}")

print("\n" + "=" * 70)
print("Step 5) Clean ACS variables (treat invalid values as missing)")
print("=" * 70)

cols_check = ["med_hh_inc", "pct_no_hlt", "pop_0_17", "tot_pop"]

for col in cols_check:
    if col not in cook_tracts.columns:
        raise ValueError(f"Missing required column in tract shapefile: {col}")

for col in cols_check:
    if col == "med_hh_inc":
        cook_tracts.loc[cook_tracts[col] <= 0, col] = np.nan
    elif col in ["pop_0_17", "tot_pop"]:
        cook_tracts.loc[cook_tracts[col] <= 0, col] = np.nan
    elif col == "pct_no_hlt":
        cook_tracts.loc[cook_tracts[col] < 0, col] = np.nan

    missing = int(cook_tracts[col].isna().sum())
    median_val = float(cook_tracts[col].median())
    cook_tracts[col] = cook_tracts[col].fillna(median_val)
    print(f"{col}: missing {missing:,} -> imputed with median {median_val:.3f}")

print("\n" + "=" * 70)
print("Step 6) Merge address counts and compute tract area & densities")
print("=" * 70)

merged = cook_tracts.merge(addr_counts, on="GEOID", how="left")
merged["addr_count"] = merged["addr_count"].fillna(0).astype(int)
merged["unit_count"] = merged["unit_count"].fillna(0).astype(float)

area_gdf = merged.to_crs(epsg=3435)
merged["area_sqkm"] = area_gdf.geometry.area / 1_000_000
merged.loc[merged["area_sqkm"] <= 0, "area_sqkm"] = np.nan

merged["addr_per_sqkm"] = merged["addr_count"] / merged["area_sqkm"]
merged["unit_per_sqkm"] = merged["unit_count"] / merged["area_sqkm"]
merged["pop_per_sqkm"]  = merged["tot_pop"]    / merged["area_sqkm"]

print("Added fields: area_sqkm, addr_per_sqkm, unit_per_sqkm, pop_per_sqkm")

print("\n" + "=" * 70)
print("Step 7) Approximate South Side subset (bounding box on tract centroid)")
print("=" * 70)

merged["centroid_lat"] = merged.geometry.centroid.y
merged["centroid_lon"] = merged.geometry.centroid.x

south_side = merged[
    (merged["centroid_lat"] >= 41.63) &
    (merged["centroid_lat"] <= 41.87) &
    (merged["centroid_lon"] >= -87.76) &
    (merged["centroid_lon"] <= -87.52)
].copy()

south_side = south_side.drop(columns=["centroid_lat", "centroid_lon"])
print(f"South Side tracts (approx): {len(south_side):,}")
print(f"  Tracts with >=1 address:  {(south_side['addr_count'] > 0).sum():,}")
print(f"  Total buildings:           {south_side['addr_count'].sum():,}")
print(f"  Total housing units:       {south_side['unit_count'].sum():,.0f}")

print("\n" + "=" * 70)
print("Step 8) Create income quartiles for visualization")
print("=" * 70)

south_side["income_quartile"] = pd.qcut(
    south_side["med_hh_inc"], q=4,
    labels=["Q1 (Lowest)", "Q2", "Q3", "Q4 (Highest)"]
)

print("\n" + "=" * 70)
print("Step 9) Save output GeoJSON")
print("=" * 70)

south_side.to_file(OUT_PATH, driver="GeoJSON")
print(f"✅ Saved: {OUT_PATH}")
print("✅ preprocessing.py done.")
