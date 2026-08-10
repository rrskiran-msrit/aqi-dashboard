# Air Quality Intelligence Dashboard

Lightweight, data-only Streamlit dashboard generated from the CPCB Workbench output bundle.

## Run locally

```powershell
cd "C:\Users\rrski\Desktop\Air Quality Datasets\Streamlit_code_10th_Aug\air_quality_dashboard"
streamlit run streamlit_app.py
```

## Included cities

- Jaipur: 6 stations
- Hyderabad: 14 stations
- Bengaluru: 13 stations

## Add another city

Run the compact importer against the new modeled-results folder:

```powershell
python add_city.py "C:\path\to\new_city_results"
```

The importer reads `streamlit_bundle/station_index.csv`, validates each station, copies only the
required compact files, and updates the dashboard city registry. Required station files are
`station_hourly.parquet` and `station_config.json`. Metrics and prediction CSVs are optional.

The deployment bundle intentionally excludes generated images, duplicate hourly CSV files, model
weights, training logs, and the original result ZIP. Charts are generated from Parquet/CSV data.
