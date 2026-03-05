# Chicago South Side Inequality Dashboard

This dashboard explores **address density** (a proxy for housing concentration), **income**, and **uninsured rates** at the **Census Tract** level in Chicago's South Side.

## Data Sources

| File | Description |
|------|-------------|
| `data/raw-data/cleaned_F_ADD1_not0.csv` | ~46,899 residential building addresses with lat/lon and unit counts (replaces older `obama_addresses_mappable_t.csv`) |
| `data/raw-data/il_tract.shp` | Illinois Census tract shapefile (Cook County subset) |

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
