# PIIANN AQI Dashboard

PIIANN AQI Dashboard is an interactive, data-driven air-quality monitoring and model-analysis application built with Streamlit. It presents historical hourly air-quality observations, station-to-station comparisons, pollutant behaviour, data-quality information, and validation results from conventional and physics-informed machine-learning models.

The project is intended to make detailed air-quality analysis easier to explore for researchers, students, policymakers, environmental professionals, and members of the public. It complements the underlying scientific workflow by converting its processed results into an accessible web application.

This dashboard and report are submitted as part of the thesis work titled **“Development of Physics-Guided AI Platform for Local Air Quality Index Prediction and Monitoring in Indian Cities.”** The thesis is submitted to the **Centre for Distance and Online Education, Andhra University**, in partial fulfilment of the requirements for the award of the **Master of Computer Applications** degree.

**Researcher:** Rama Siva Kiran Reddy (Reg. No. A24CA0239)  
**Research guide:** Dr. Manish Kumar, Associate Professor, School of Computer Science and Engineering, R V University, Bangalore – 560059  
**Institution:** Centre for Distance and Online Education, Andhra University, Visakhapatnam  
**Year:** 2026

> **Important:** The current deployment is a historical and model-validation dashboard. It is not a live CPCB sensor feed and must not be interpreted as a real-time warning system or medical service.

## Project objectives

The dashboard is being developed to:

- provide a unified view of hourly air quality across multiple Indian cities;
- compare monitoring stations using the same selected period and map timestamp;
- communicate AQI categories using CPCB-aligned colours and public-health guidance;
- explore PM2.5, PM10, NO₂, SO₂, CO, and O₃ behaviour;
- identify temporal patterns at hourly, daily, monthly, and seasonal scales;
- compare conventional, deep-learning, and physics-informed AQI models;
- report missing data and station coverage alongside every comparison;
- support reproducible analysis and the public communication of research results; and
- provide a scalable structure into which additional city result bundles can be imported.

## Cities and stations currently included

| City | Monitoring stations |
|---|---:|
| Ahmedabad | 9 |
| Bengaluru | 13 |
| Bhopal | 3 |
| Bhubaneswar | 2 |
| Chandigarh | 3 |
| Chennai | 9 |
| Delhi | 38 |
| Guwahati | 4 |
| Hyderabad | 14 |
| Jaipur | 6 |
| Kanpur | 4 |
| Kochi | 1 |
| Kolkata | 7 |
| Lucknow | 6 |
| Mumbai | 19 |
| Nagpur | 4 |
| Patna | 6 |
| Pune | 7 |
| Raipur | 4 |
| Surat | 1 |
| **Total** | **160** |

The station registry contains **20 Indian cities and 160 monitoring stations**. The application discovers cities dynamically from `data/station_index.csv`; therefore, every imported city is available through the city selector without a separate code change.

## Dataset scale and processing volume

The complete multi-city workflow is based on **raw hourly air-quality data**. Source files were discovered, validated, standardized, quality-checked, and consolidated into one compact hourly Parquet dataset for each monitoring station. The following inventory was calculated directly from the current 20-city project files on **15 August 2026**:

| Measure | Current verified total |
|---|---:|
| Cities | 20 |
| Monitoring stations | 160 |
| Raw annual source-file records found in the validated city manifests | 784 |
| Standardized station-hour records analysed | **6,658,128 (6.66 million)** |
| Non-missing AQI station-hour observations | **2,076,914 (2.08 million)** |
| Non-missing measurements across PM2.5, PM10, NO2, SO2, CO and O3 | **13,207,214 (13.21 million)** |
| Distinct hourly timestamps represented across the combined dataset | 43,368 |
| Overall temporal extent | 1 January 2021 to 12 December 2025 |
| Files in the 20 processed city result directories | **12,936** |

The **6.66 million** figure represents station-hour records: the same clock timestamp recorded at different monitoring stations is counted once for each station because each is an independently analysed observation. The **43,368** figure counts distinct clock timestamps only once across the complete project. Therefore, these two quantities describe different aspects of the dataset and should not be interchanged.

The project contains more than 10,000 files when all generated research artifacts are included. That **12,936-file** inventory includes raw-data records and the resulting cleaned datasets, audit files, model outputs, metrics, figures and reports; it should not be described as 12,936 downloaded raw files. The validated manifests currently document **784 raw annual source files**, all marked as found.

### Standardized hourly records by city

| City | Stations | Station-hour records |
|---|---:|---:|
| Ahmedabad | 9 | 390,312 |
| Bengaluru | 13 | 537,432 |
| Bhopal | 3 | 112,584 |
| Bhubaneswar | 2 | 51,696 |
| Chandigarh | 3 | 130,104 |
| Chennai | 9 | 390,288 |
| Delhi | 38 | 1,647,984 |
| Guwahati | 4 | 164,712 |
| Hyderabad | 14 | 607,152 |
| Jaipur | 6 | 207,648 |
| Kanpur | 4 | 155,904 |
| Kochi | 1 | 43,368 |
| Kolkata | 7 | 303,576 |
| Lucknow | 6 | 251,448 |
| Mumbai | 19 | 823,992 |
| Nagpur | 4 | 120,912 |
| Patna | 6 | 260,208 |
| Pune | 7 | 259,488 |
| Raipur | 4 | 173,472 |
| Surat | 1 | 25,848 |
| **Total** | **160** | **6,658,128** |

## Application pages

- **City dashboard:** maps, rankings, temporal patterns, pollution burden, city-wide model evidence and reliability.
- **Station explorer:** detailed AQI, pollutant, model and data-quality tabs for an individual station.
- **Research visuals:** manuscript-style station–pollutant bubble tables, ranked dot plots, calendar/temporal heatmaps, seasonal distributions and correlation-dot matrices.
- **References:** searchable author–year bibliography with direct article/PDF links.
- **About & acknowledgement:** formal MCA project-submission details and acknowledgement.

The City Dashboard begins with the publication-quality national study-area figure and an interactive major-city study map. Hovering over a mapped city displays its station count, latest historical median AQI, AQI category, station range, and most recent available record. The complete 20-city dataset remains accessible through the dynamic city selector. The 250 km circles indicate nominal study domains rather than pollutant dispersion or administrative boundaries.

## Methodological framework diagrams

The diagrams below summarize the dashboard workflow and the PIIANN modelling methodology. Select either image to open its full-resolution GitHub version.

### PIIANN AQI Dashboard methodological framework

[![PIIANN AQI Dashboard methodological framework](https://raw.githubusercontent.com/rrskiran-msrit/aqi-dashboard/main/PIIANN_AQI_Methodology_Flow_Diagram.png)](https://github.com/rrskiran-msrit/aqi-dashboard/blob/main/PIIANN_AQI_Methodology_Flow_Diagram.png)

This framework describes the progression from processed hourly observations and the multi-city station registry through user selection, city/station analysis, model evidence, interpretation, data-quality safeguards, and downloadable research outputs.

### PIIANN model methodological framework

[![PIIANN model methodological framework](https://raw.githubusercontent.com/rrskiran-msrit/aqi-dashboard/main/PIIANN_Methodology_Flow_Diagram.png)](https://github.com/rrskiran-msrit/aqi-dashboard/blob/main/PIIANN_Methodology_Flow_Diagram.png)

This framework documents the physics-guided artificial-intelligence workflow used to construct, validate, compare, and interpret AQI prediction models.

## Dashboard capabilities

### City overview

The city-level interface compares all monitoring stations in a selected city.

- Interactive AQI map with CPCB-category marker colours
- Hover details for station, AQI, category, dominant pollutant, and observation time
- Shared map date and hour for fair station comparison
- Adjustable nearest-observation tolerance
- City median AQI and reporting-station count
- Cleanest and highest-AQI station at the selected snapshot
- Rankings by mean AQI, median AQI, healthy-hour share, or unhealthy-hour share
- Minimum-coverage safeguard for station rankings
- Station-by-hour AQI heatmap
- Station-by-month AQI heatmap
- AQI-category exposure comparison
- Poor-or-worse and Good/Satisfactory hour percentages
- Station-level AQI data-coverage comparison
- Downloadable city-comparison table

### City model laboratory

The City Model Lab summarizes AQI validation performance across every station.

- Complete station-by-model evidence table
- Mean city-level MAE, RMSE, R², and bias
- Number of stations evaluated by each model
- Count of stations at which each model was best by RMSE
- Station × model RMSE heatmap
- Explicit indication of whether row-level predictions were saved
- Downloadable city-wide model evidence

Models represented in the result bundles include:

1. Persistence baseline
2. Linear Regression
3. Gradient Boosting
4. Random Forest
5. Gated Recurrent Unit (GRU)
6. Physics-Informed Neural Network (PINN)
7. Physics-Informed Integrated Artificial Neural Network (PIIANN)

All seven models have aggregate validation metrics where training completed. In the current result bundles, row-level observed-versus-predicted values are generally saved only for GRU, PINN, and PIIANN. Conventional model metrics are still included in the city and station comparisons.

### Station explorer

The station-level interface provides detailed investigation of one monitoring location.

- Date-range filtering
- Latest available historical AQI and category
- AQI health-information panel
- Hourly AQI trajectory
- Daily and monthly aggregation
- Diurnal profile by hour of day
- Seasonal profile by month
- AQI-category distribution
- Interactive pollutant comparison
- Monitoring-station map
- Model ranking by validation performance
- Observed-versus-predicted scatterplot where rows were saved
- Variable-level data coverage
- Filtered CSV download
- GRU, PINN, and PIIANN available in the point-level model-diagnostics selector
- Model-specific MAE, RMSE, R², bias, and validation-row cards
- Residual histogram and AQI-category confusion matrix where prediction rows were saved
- Manuscript-style descriptive table with confidence intervals, percentiles, skewness, and kurtosis
- Robust-scaled multi-variable box plots
- Hour × weekday and month × year heatmaps
- Seasonal density/violin view
- Exceedance-frequency and empirical cumulative-distribution curves
- Principal component analysis with seasonal score plot and component loadings
- Consecutive Poor+ pollution-episode table
- Robust anomaly screening and annotated anomaly time series

Point-level prediction diagnostics are displayed only for GRU, PINN, and PIIANN because the compact bundles contain observed and predicted rows for these models. Random Forest, Gradient Boosting, Linear Regression, and Persistence remain in the aggregate model-comparison chart but are temporarily excluded from the diagnostic selector.

## AQI categories

The dashboard uses the following AQI categories and visual colours:

| AQI | Category | Dashboard colour |
|---:|---|---|
| 0–50 | Good | Green |
| 51–100 | Satisfactory | Light green |
| 101–200 | Moderate | Yellow |
| 201–300 | Poor | Orange |
| 301–400 | Very Poor | Red |
| 401–500 | Severe | Purple |

Grey map markers indicate that no AQI observation was found within the selected timestamp tolerance. Health messages are informational and are not a substitute for guidance from public-health authorities or qualified medical professionals.

## Data and analytical design

Each station bundle can contain:

- a standardized hourly timeline;
- AQI and AQI category;
- PM2.5, PM10, NO₂, SO₂, CO, and O₃;
- additional pollutants where available;
- temperature, relative humidity, wind, rainfall, pressure, or solar variables where available;
- station latitude and longitude;
- AQI model-validation metrics;
- row-level AQI predictions for selected models;
- pollutant predictions; and
- a station configuration file describing features and model settings.

The web application uses compact Parquet files for hourly data and CSV/JSON files for metrics and metadata. All charts are generated dynamically. Pre-rendered PNG figures, duplicate hourly CSV files, training logs, original result ZIP files, and neural-network weights are intentionally excluded from the deployment bundle.

## Fair-comparison safeguards

Air-quality comparisons can be misleading when stations have different observation periods or missing-data rates. The PIIANN AQI Dashboard therefore applies the following safeguards:

- map values are matched to one shared date and hour;
- users choose an acceptable nearest-observation tolerance;
- stations without a nearby observation are shown in grey;
- rankings use a common selected period;
- a minimum AQI coverage threshold can exclude unreliable rankings;
- coverage is displayed alongside model and air-quality results; and
- model performance is kept separate from environmental cleanliness.

A station with the best prediction model is not necessarily the cleanest station. Similarly, a station with a low mean AQI should not be declared the cleanest without considering its observation coverage.

## Prediction interpretation and limitations

The underlying modeling workflow uses the preceding hourly history to evaluate next-hour AQI prediction. The current saved prediction tables represent held-out model-validation results; they are not continuously refreshed future forecasts.

The current research configuration includes a 24-hour history window and a one-hour prediction horizon. Some experiment outputs were produced using a random train/validation/test split. Operational future forecasting should instead use chronological or walk-forward validation, regularly updated observations, explicit forecast timestamps, and uncertainty estimates.

The dashboard therefore labels saved prediction results as **model validation**, not as a live forecast.

## Repository structure

```text
air-quality-piiml-dashboard/
├── streamlit_app.py             # Main Streamlit application
├── add_city.py                  # Compact city-bundle importer
├── PIIANN_AQI_Methodology_Flow_Diagram.png
├── PIIANN_Methodology_Flow_Diagram.png
├── assets/
│   └── Study-area.png           # Publication-quality India study-area figure
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
├── .streamlit/
│   └── config.toml              # Streamlit theme and server configuration
└── data/
    ├── station_index.csv        # City and station registry
    ├── ahmedabad/
    ├── bengaluru/
    ├── bhopal/
    ├── bhubaneswar/
    ├── chandigarh/
    ├── chennai/
    ├── delhi/
    ├── guwahati/
    ├── hyderabad/
    ├── jaipur/
    ├── kanpur/
    ├── kochi/
    ├── kolkata/
    ├── lucknow/
    ├── mumbai/
    ├── nagpur/
    ├── patna/
    ├── pune/
    ├── raipur/
    ├── surat/
    └── city_summary/
```

Each city directory contains one subdirectory per monitoring station:

```text
data/<city>/<station>/
├── station_hourly.parquet       # Standardized hourly observations
├── station_config.json          # Station, feature, and model configuration
├── model_metrics.csv            # Validation metrics for available models
├── aqi_predictions.csv          # Saved AQI prediction rows
└── pollutant_predictions.csv    # Saved pollutant prediction rows, where available
```

Each compact station directory normally contains:

```text
station_hourly.parquet
station_config.json
model_metrics.csv
aqi_predictions.csv          # Optional
pollutant_predictions.csv    # Optional
```

## Installation

Python 3.12 or 3.13 is recommended. Python 3.14 may not have compatible prebuilt wheels for every pinned data dependency.

### Windows Command Prompt

```bat
py -3.13 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

### PowerShell

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

The application normally opens at:

```text
http://localhost:8501
```

Node.js and `npm install` are not required. This is a Python Streamlit application.

## Importing another city

The new city must first be processed by the modeling workflow so that it contains a valid `streamlit_bundle` and `station_index.csv`.

Run:

```bat
python add_city.py "C:\path\to\new_city_results"
```

The importer:

1. reads `streamlit_bundle/station_index.csv`;
2. validates required station files;
3. copies compact hourly data, configurations, metrics, and available predictions;
4. excludes figures, model weights, logs, and duplicate data; and
5. updates `data/station_index.csv` without duplicating existing city-station entries.

After importing a city, restart Streamlit or clear its data cache.

## Deploying on Streamlit Community Cloud

1. Create a public GitHub repository.
2. Upload the repository contents, but do not upload `.venv`, `__pycache__`, logs, result ZIP files, or original training folders.
3. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/).
4. Select the GitHub repository and branch.
5. Set the entrypoint to `streamlit_app.py`.
6. Select a supported Python version, preferably Python 3.12 or 3.13.
7. Choose an available `*.streamlit.app` subdomain.
8. Deploy and set the application to public access.
9. Test the public URL in a private/incognito browser before sharing it.

## Reproducibility and responsible use

- Dashboard results depend on the supplied processed station bundles.
- Missing values remain missing and are not presented as observations.
- Station coverage varies by city, station, variable, and period.
- Model metrics should only be compared under compatible validation settings.
- Correlation does not establish a pollution source or causal relationship.
- The application should not be used for emergency response or individual medical decisions.
- A public dashboard URL should be accompanied by a versioned source-code/data archive for scholarly publication.

For manuscript publication, consider archiving a release through Zenodo and citing both the live Streamlit application and the archived DOI.

## Suggested citation

```text
Reddy, R. S. K. (2026). PIIANN AQI Dashboard [Software].
GitHub: https://github.com/rrskiran-msrit/aqi-dashboard
Live application: [Streamlit application URL].
Archived version: [Zenodo DOI].
```

## Research acknowledgement

This dashboard is the presentation layer for a broader air-quality processing and physics-informed machine-learning workflow. It was prepared as part of the MCA thesis submission of **Rama Siva Kiran Reddy** under the guidance of **Dr. Manish Kumar**. The author acknowledges the Centre for Distance and Online Education, Andhra University; the laboratory support provided through the Department of Chemical Engineering, M S Ramaiah Institute of Technology, Bengaluru; and the Central Pollution Control Board, New Delhi, for the air-quality data used in the research.

The thesis and associated scientific report should be consulted for complete information about data provenance, preprocessing, feature construction, model architecture, training, validation, uncertainty, and interpretation.

## Licence

This project is released under the **MIT License**. See the repository's [`LICENSE`](https://github.com/rrskiran-msrit/aqi-dashboard/blob/main/LICENSE) file for the complete licence text.

The MIT License applies to the project software. Air-quality datasets and other third-party materials remain subject to the terms and attribution requirements of their respective providers.

## Contact

**Rama Siva Kiran Reddy**  
Email: [rrskiran@msrit.edu](mailto:rrskiran@msrit.edu)

For questions about the research, dashboard, reproducibility, or academic collaboration, please use the email address above.
