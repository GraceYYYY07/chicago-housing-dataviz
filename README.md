# Housing Density, Income, and Uninsured Rates in Chicago's South Side

**Course:** Data Visualization (PPHA 30538, 2026 Winter) | **Team:** team40
**Group Member** Grace Yao, Jiamin Zhang, Yuxin Zheng 
**Streamlit URL**:https://final-project-team40.streamlit.app/
---

## Research Question

Do residents of dense, low-income neighborhoods on Chicago's South Side face greater barriers to healthcare coverage? This project examines how housing density and neighborhood income are associated with uninsured rates at the census tract level.

Lower-income households, who are both less able to afford coverage out-of-pocket and more likely to work in jobs that do not offer employer-sponsored plans. Housing density introduces an additional dimension: neighborhoods characterized by high density may disproportionately house transient residents and workers in informal or low-benefit employment sectors, potentially contributing to elevated rates of uninsurance.
These dynamics motivate the central descriptive question of this project: Across census tracts in Chicago's South Side, how do housing density and neighborhood income levels relate to the proportion of residents lacking health insurance coverage?

---

## Data Sources

| Dataset | Description | Source |
|---|---|---|
| `address_data.csv` | ~488,689 residential building addresses (replaces `obama_addresses_mappable_t.csv`) | [Chicago Data Portal – Building Footprints](https://data.cityofchicago.org/Buildings/Building-Footprints/syp8-uezg/about_data) |
| `il_tract.shp` | Illinois Census tract shapefile (Cook County subset) | U.S. Census Bureau | ([ACS American Community Survey](https://data.census.gov/))

The shapefile provides tract-level income, uninsured rate, and geographic boundaries. The address dataset is used to construct housing density as a proxy variable.
---

## Data Processing Pipeline (6 Steps)

1. **Geocoding** — Each address is converted to a geographic point and projected to EPSG:4326 for coordinate alignment with the shapefile.
2. **Spatial Join** — Address points are matched to the Census tract polygon containing them, assigning a tract ID to each record.
3. **Density Calculation** — Addresses are grouped and counted by tract, yielding a housing density proxy.
4. **Merge** — Address counts are left-joined back to the shapefile; tracts with no matched addresses receive a density of zero rather than being dropped.
5. **Filter & Enrich** — The merged dataset is filtered to South Side tracts only.
6. **Income Quartile Construction** — Tracts are split into four equal income groups to enable comparison across the income distribution.

---

## Static Figures

**Figure 1 — Spatial Overlap of Income and Housing Density** (GeoPandas)

Before modeling uninsured rates, we first examine the relationship between residential density and income, since density may be correlated with economic structure. Density contour lines are overlaid on the tract map; tracts that are simultaneously low-income and high-density are outlined in orange.

**Figure 2 — Income, Density, and Uninsured Rates**(Altair)

Scatter plots showing how each independent variable relates to uninsured rates across tracts. Income shows a strong negative association with uninsured rates; density alone does not predict uninsured rates as clearly.

---

## Streamlit Dashboard

Static figures provide a fixed perspective, but policymakers may define "high priority" differently. The dashboard lets users set their own thresholds and explore the data interactively.

**Sidebar controls:**
- **Metric selector** — base map of income, uninsured rate, child population, or address density
- **Priority threshold sliders** — define what counts as a priority tract (defaults: top 25% density, bottom 25% income, top 25% uninsured rate)

Tracts meeting all three thresholds are highlighted with orange outlines.
**Tab 1 — Interactive Map** (Folium): Pan, zoom, and hover over any tract to see all indicators. Priority tracts are outlined in orange.
**Tab 2 — Scatter Plots**: Income vs. uninsured rate, and density vs. uninsured rate, for the filtered selection.
**Tab 3 — Priority Tract Table**: A ranked list of tracts meeting all three priority conditions, sorted by a composite score combining density and uninsured signals. Downloadable as CSV.

---

## Key Findings
Income is negatively associated with uninsured rates at the tract level: lower-income neighborhoods consistently show higher uninsured rates. Housing density shows a weaker standalone relationship with uninsured rates, but areas with both high density and low income tend to cluster geographically and account for a disproportionate share of high-uninsured tracts. The dashboard enables policymakers to identify and prioritize these overlapping areas.

## Revise After Reflection
> **Why using density as one of the independent variable?** Housing density is included as a key predictor on the grounds that densely populated neighborhoods are more likely to contain informal housing arrangements, highly mobile resident populations, and workers employed in industries that seldom provide employer-sponsored coverage. Through this channel, greater residential density could translate into higher uninsured rates at the neighborhood level. We therefore treat housing density as a potentially important determinant of tract-level insurance coverage in our analysis.

> **Why address points?** Address coordinates offer a finer spatial proxy for residential concentration than census household counts, and better capture how buildings are distributed across tracts. Density is measured as building addresses per square kilometer.

> **Why is there zero density data and how we fixed it:** In an earlier version of the project we used the dataset obama_addresses_mappable_t.csv, which only covered the area surrounding the Obama Presidential Center. Because that dataset represented only a small portion of Chicago’s South Side and produced many tracts with zero density values, we replaced it with a larger address dataset covering the broader region (address_data.csv) 

## Usage

1) Install dependencies
```bash
pip install -r requirements.txt
```

2) Build tract-level dataset (generates `derived-data/merged_tract.geojson`)
```bash
python code/preprocessing.py
```

3) (Optional) Regenerate static figures
```bash
python code/figures.py
```

4) Run the dashboard
```bash
streamlit run streamlit-app/app.py
```

## Notes
- Address density is computed as: **buildings per square kilometer** (`addr_per_sqkm`).
- Housing unit density (`unit_per_sqkm`) uses the `NO_OF_UNIT` field from the address dataset.
- Address/unit density is a proxy (city address database records, not official census housing units).
- Findings are descriptive and correlational.
