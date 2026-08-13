from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
ASSET_DIR = APP_DIR / "assets"

CITY_CENTRES = {
    "Delhi": (28.6139, 77.2090), "Mumbai": (19.0760, 72.8777),
    "Kolkata": (22.5726, 88.3639), "Chennai": (13.0827, 80.2707),
    "Hyderabad": (17.3850, 78.4867), "Bengaluru": (12.9716, 77.5946),
    "Jaipur": (26.9124, 75.7873),
}

AQI_BANDS = [
    (0, 50, "Good", "#22A06B"),
    (51, 100, "Satisfactory", "#84B547"),
    (101, 200, "Moderate", "#F2B134"),
    (201, 300, "Poor", "#F1783C"),
    (301, 400, "Very Poor", "#D64B55"),
    (401, 500, "Severe", "#8D3B72"),
]

POLLUTANTS = {
    "pm25": "PM₂.₅",
    "pm10": "PM₁₀",
    "no2": "NO₂",
    "so2": "SO₂",
    "co": "CO",
    "o3": "O₃",
}

HEALTH = {
    "Good": "Air quality is considered satisfactory. Enjoy normal outdoor activity.",
    "Satisfactory": "Sensitive people may experience minor discomfort after prolonged exposure.",
    "Moderate": "People with lung, heart or asthma conditions should reduce prolonged exertion.",
    "Poor": "Reduce prolonged outdoor activity, especially for children and sensitive groups.",
    "Very Poor": "Avoid strenuous outdoor activity. Sensitive groups should remain indoors.",
    "Severe": "Avoid outdoor exertion. Keep windows closed and follow local health advice.",
}


st.set_page_config(
    page_title="Air Quality Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {font-family: 'DM Sans', sans-serif;}
.stApp {background: radial-gradient(circle at 95% 0%, #dff7ed 0, transparent 27%), #f4f7f6;}
[data-testid="stSidebar"] {background: #10261f; border-right: 1px solid #244138;}
[data-testid="stSidebar"] * {color: #eef8f4;}
[data-testid="stSidebar"] label {color: #a9c7bc !important; font-weight: 600;}
.block-container {padding-top: 1.7rem; max-width: 1500px;}
.hero {background: linear-gradient(120deg,#10261f 0%,#174b3a 58%,#167754 100%); color:white;
  border-radius:24px; padding:28px 32px; box-shadow:0 16px 42px rgba(17,61,47,.16);
  animation:rise .55s ease-out both; position:relative; overflow:hidden;}
.hero:after {content:""; position:absolute; width:230px; height:230px; right:-60px; top:-100px;
  border:1px solid rgba(255,255,255,.17); border-radius:50%; box-shadow:0 0 0 38px rgba(255,255,255,.035);}
.eyebrow {letter-spacing:.13em; text-transform:uppercase; color:#96dbc0; font-size:.72rem; font-weight:700;}
.hero h1 {font-size:2rem; margin:.35rem 0 .25rem; letter-spacing:-.035em;}
.hero p {color:#cde4dc; margin:0; max-width:720px;}
.metric-card {background:rgba(255,255,255,.95); border:1px solid #e1ebe7; border-radius:18px;
  padding:18px 19px; min-height:132px; box-shadow:0 8px 28px rgba(24,62,50,.06);
  animation:rise .5s ease-out both; transition:transform .18s ease, box-shadow .18s ease;}
.metric-card:hover {transform:translateY(-3px); box-shadow:0 13px 32px rgba(24,62,50,.11);}
.metric-icon {font-size:1.15rem; display:inline-flex; width:34px; height:34px; align-items:center;
  justify-content:center; background:#e8f6f0; border-radius:10px;}
.metric-label {color:#65756f; font-size:.78rem; font-weight:600; margin-top:12px;}
.metric-value {font-size:1.72rem; font-weight:700; color:#14231d; line-height:1.1;}
.metric-note {color:#81908b; font-size:.72rem; margin-top:5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.health-card {border-radius:17px; padding:17px 20px; color:#fff; margin:8px 0 15px;}
.section-kicker {color:#16815d; font-weight:700; text-transform:uppercase; letter-spacing:.1em; font-size:.7rem;}
.section-title {font-weight:700; font-size:1.25rem; margin:.15rem 0 .7rem; color:#183029;}
.data-note {background:#eef5f2; border-left:3px solid #15a776; padding:11px 14px; border-radius:0 10px 10px 0;
 color:#50635c; font-size:.82rem;}
[data-baseweb="tab-list"] {gap:8px;}
[data-baseweb="tab"] {background:white; border-radius:12px; padding:8px 16px; border:1px solid #e4ece9;}
[aria-selected="true"] {background:#173d31 !important; color:white !important;}
@keyframes rise {from {opacity:0; transform:translateY(9px)} to {opacity:1; transform:translateY(0)}}
@media (prefers-reduced-motion: reduce) {* {animation:none !important; transition:none !important;}}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_index() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "station_index.csv")


@st.cache_data(show_spinner=False, max_entries=4)
def load_station(folder: str) -> tuple[pd.DataFrame, dict]:
    station_dir = DATA_DIR / folder
    df = pd.read_parquet(station_dir / "station_hourly.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    with (station_dir / "station_config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    return df, config


@st.cache_data(show_spinner=False, max_entries=8)
def load_optional_csv(folder: str, filename: str) -> pd.DataFrame:
    path = DATA_DIR / folder / filename
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def aqi_style(value: float) -> tuple[str, str]:
    if pd.isna(value):
        return "Unavailable", "#66756f"
    for low, high, category, color in AQI_BANDS:
        if low <= value <= high:
            return category, color
    return "Severe", "#8D3B72"


def latest_value(df: pd.DataFrame, column: str) -> tuple[float, pd.Timestamp | None]:
    if column not in df:
        return np.nan, None
    q = df[["timestamp", column]].dropna()
    return (float(q.iloc[-1][column]), q.iloc[-1]["timestamp"]) if len(q) else (np.nan, None)


def metric_card(icon: str, label: str, value: str, note: str = "") -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-icon">{icon}</div>'
        f'<div class="metric-label">{label}</div><div class="metric-value">{value}</div>'
        f'<div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def section(kicker: str, title: str) -> None:
    st.markdown(
        f'<div class="section-kicker">{kicker}</div><div class="section-title">{title}</div>',
        unsafe_allow_html=True,
    )


def downsample(df: pd.DataFrame, limit: int = 6000) -> pd.DataFrame:
    if len(df) <= limit:
        return df
    return df.iloc[np.linspace(0, len(df) - 1, limit).astype(int)]


def descriptive_statistics(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if values.empty:
            continue
        sem = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.nan
        rows.append({
            "Variable": "AQI" if column == "aqi" else POLLUTANTS.get(column, column.replace("_", " ").title()),
            "N": len(values), "Coverage (%)": 100 * len(values) / max(len(df), 1),
            "Mean": values.mean(), "95% CI low": values.mean() - 1.96 * sem,
            "95% CI high": values.mean() + 1.96 * sem, "Median": values.median(),
            "SD": values.std(), "Min": values.min(), "P05": values.quantile(.05),
            "Q1": values.quantile(.25), "Q3": values.quantile(.75), "P95": values.quantile(.95),
            "Max": values.max(), "IQR": values.quantile(.75) - values.quantile(.25),
            "Skewness": values.skew(), "Kurtosis": values.kurt(),
            "Missing": int(df[column].isna().sum()), "Unique": values.nunique(),
        })
    return pd.DataFrame(rows)


def pca_scores(df: pd.DataFrame, columns: list[str], limit: int = 6000) -> tuple[pd.DataFrame, pd.DataFrame, list[float]]:
    complete = df[["timestamp", *columns]].dropna()
    complete = downsample(complete, limit)
    if len(complete) < 3 or len(columns) < 2:
        return pd.DataFrame(), pd.DataFrame(), []
    matrix = complete[columns].astype(float)
    std = matrix.std(ddof=0).replace(0, np.nan)
    z = ((matrix - matrix.mean()) / std).dropna(axis=1)
    used = list(z.columns)
    if len(used) < 2:
        return pd.DataFrame(), pd.DataFrame(), []
    u, singular, vt = np.linalg.svd(z.to_numpy(), full_matrices=False)
    variance = singular ** 2
    ratios = (variance / variance.sum()).tolist()
    scores = complete.loc[z.index, ["timestamp"]].copy()
    scores["PC1"] = u[:, 0] * singular[0]
    scores["PC2"] = u[:, 1] * singular[1]
    scores["season"] = scores["timestamp"].dt.month.map({12:"Winter",1:"Winter",2:"Winter",3:"Pre-monsoon",4:"Pre-monsoon",5:"Pre-monsoon",6:"Monsoon",7:"Monsoon",8:"Monsoon",9:"Monsoon",10:"Post-monsoon",11:"Post-monsoon"})
    loadings = pd.DataFrame({"Variable": ["AQI" if c == "aqi" else POLLUTANTS.get(c, c) for c in used], "PC1": vt[0], "PC2": vt[1]})
    return scores, loadings, ratios


def episode_table(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.set_index("timestamp")["aqi"].resample("D").mean().dropna().reset_index()
    if daily.empty:
        return daily
    daily["polluted"] = daily["aqi"] > 200
    daily["group"] = (daily["polluted"] != daily["polluted"].shift()).cumsum()
    episodes = daily[daily["polluted"]].groupby("group").agg(
        Start=("timestamp", "min"), End=("timestamp", "max"), Duration_days=("timestamp", "size"),
        Mean_AQI=("aqi", "mean"), Peak_AQI=("aqi", "max"),
    ).reset_index(drop=True)
    return episodes.sort_values(["Duration_days", "Peak_AQI"], ascending=False)


@st.cache_data(show_spinner=False, max_entries=1)
def national_city_snapshot(index_records: tuple[tuple[str, str, str], ...]) -> pd.DataFrame:
    """Summarize the latest historical AQI available at every registered station."""
    rows = []
    for city_name, station_name, folder_name in index_records:
        path = DATA_DIR / folder_name / "station_hourly.parquet"
        try:
            q = pd.read_parquet(path, columns=["timestamp", "aqi"])
        except Exception:
            continue
        q["timestamp"] = pd.to_datetime(q["timestamp"], errors="coerce")
        latest = q.dropna(subset=["timestamp", "aqi"]).sort_values("timestamp").tail(1)
        if len(latest):
            rows.append({"city": city_name, "station": station_name, "timestamp": latest.iloc[0]["timestamp"], "aqi": float(latest.iloc[0]["aqi"])})
    latest = pd.DataFrame(rows)
    if latest.empty:
        return latest
    result = latest.groupby("city", as_index=False).agg(
        stations=("station", "nunique"), city_aqi=("aqi", "median"),
        minimum_aqi=("aqi", "min"), maximum_aqi=("aqi", "max"), latest_observation=("timestamp", "max"),
    )
    result["category"] = result["city_aqi"].map(lambda value: aqi_style(float(value))[0])
    result["color"] = result["city_aqi"].map(lambda value: aqi_style(float(value))[1])
    result["latitude"] = result["city"].map(lambda name: CITY_CENTRES.get(name, (np.nan, np.nan))[0])
    result["longitude"] = result["city"].map(lambda name: CITY_CENTRES.get(name, (np.nan, np.nan))[1])
    result["aqi_display"] = result["city_aqi"].round().astype(int).astype(str)
    result["range_display"] = result.apply(lambda r: f'{r.minimum_aqi:.0f}–{r.maximum_aqi:.0f}', axis=1)
    result["time_display"] = result["latest_observation"].dt.strftime("%d %b %Y · %H:%M")
    result["fill_color"] = result["color"].map(hex_to_rgba)
    return result.dropna(subset=["latitude", "longitude"])


def render_study_area(index_df: pd.DataFrame) -> None:
    section("National study coverage", "Seven-city study area")
    image_path = ASSET_DIR / "Study-area.png"
    if image_path.exists():
        st.image(str(image_path), caption="Figure 1. National study-area framework used to guide the selection and expansion of Indian urban air-quality domains.", use_container_width=True)
        st.markdown(
            '<div class="data-note"><b>Interpretation.</b> The static publication figure presents the broader study design and nominal 250 km city-centred domains. '
            'The operational dashboard below is restricted to the seven cities whose station bundles have been processed: Delhi, Mumbai, Kolkata, Chennai, Hyderabad, Bengaluru and Jaipur. '
            'The dotted radii represent study domains; they are not pollutant-plume boundaries or administrative limits.</div>',
            unsafe_allow_html=True,
        )
        st.download_button("Download full-resolution study-area figure", image_path.read_bytes(), "PIIANN_AQI_Study_Area.png", "image/png")

    records = tuple((str(r.city), str(r.station), str(r.station_folder).replace("\\", "/")) for r in index_df.itertuples())
    with st.spinner("Preparing seven-city historical snapshot…"):
        national = national_city_snapshot(records)
    if national.empty:
        return
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("7", "Processed cities", f'{national["city"].nunique()}', "Major Indian urban centres")
    with c2: metric_card("●", "Monitoring stations", f'{int(national["stations"].sum())}', "Compact project registry")
    with c3: metric_card("↕", "Historical AQI range", f'{national["minimum_aqi"].min():.0f}–{national["maximum_aqi"].max():.0f}', "Across latest station records")
    with c4: metric_card("◷", "Latest data date", national["latest_observation"].max().strftime("%d %b %Y"), "Most recent station record")

    section("Interactive national view", "Hover over a city for its latest historical summary")
    radius_layer = pdk.Layer(
        "ScatterplotLayer", national, get_position="[longitude, latitude]", get_radius=250000,
        get_fill_color=[44, 124, 94, 20], get_line_color=[34, 102, 79, 120], stroked=True,
        filled=True, line_width_min_pixels=1, pickable=False,
    )
    marker_layer = pdk.Layer(
        "ScatterplotLayer", national, get_position="[longitude, latitude]", get_radius=45000,
        get_fill_color="fill_color", get_line_color=[255, 255, 255, 240], stroked=True,
        line_width_min_pixels=2, radius_min_pixels=8, radius_max_pixels=18, pickable=True, auto_highlight=True,
    )
    label_layer = pdk.Layer(
        "TextLayer", national, get_position="[longitude, latitude]", get_text="city",
        get_size=15, get_color=[17, 47, 38, 230], get_pixel_offset=[0, -22],
        get_alignment_baseline="bottom", pickable=False,
    )
    tooltip = {"html": "<b style='font-size:16px'>{city}</b><br/><b>Median AQI:</b> {aqi_display} ({category})<br/><b>Station range:</b> {range_display}<br/><b>Stations:</b> {stations}<br/><b>Latest record:</b> {time_display}<br/><span style='font-size:11px'>Historical station snapshot—not a live feed</span>", "style": {"backgroundColor": "#10261f", "color": "white", "borderRadius": "10px"}}
    st.pydeck_chart(pdk.Deck(layers=[radius_layer, marker_layer, label_layer], initial_view_state=pdk.ViewState(latitude=22.5, longitude=79.0, zoom=3.5), tooltip=tooltip, map_style=None), use_container_width=True)
    st.caption("Marker colour follows the CPCB AQI category of the median latest station AQI. Dates can differ between stations and cities; use each city dashboard for a shared time-aligned comparison.")


@st.cache_data(show_spinner=False, max_entries=4)
def city_date_bounds(folders: tuple[str, ...]) -> tuple[pd.Timestamp, pd.Timestamp]:
    bounds = []
    for folder_name in folders:
        q = pd.read_parquet(DATA_DIR / folder_name / "station_hourly.parquet", columns=["timestamp"])
        q["timestamp"] = pd.to_datetime(q["timestamp"], errors="coerce")
        bounds.append((q["timestamp"].min(), q["timestamp"].max()))
    return min(x[0] for x in bounds), max(x[1] for x in bounds)


@st.cache_data(show_spinner=False, max_entries=3)
def build_city_analysis(
    station_records: tuple[tuple[str, str], ...],
    start_iso: str,
    end_iso: str,
    snapshot_iso: str,
    tolerance_hours: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    start_ts, end_ts, snapshot_ts = map(pd.Timestamp, (start_iso, end_iso, snapshot_iso))
    summaries, snapshots, categories, hourly_rows, monthly_rows = [], [], [], [], []
    required = ["timestamp", "aqi", "aqi_category", "latitude", "longitude", *POLLUTANTS.keys()]

    for station_name, folder_name in station_records:
        path = DATA_DIR / folder_name / "station_hourly.parquet"
        schema_names = set(pd.read_parquet(path, columns=[]).columns)
        # PyArrow's empty-column read does not expose schema through pandas on every version.
        try:
            import pyarrow.parquet as pq
            schema_names = set(pq.read_schema(path).names)
        except Exception:
            pass
        columns = [c for c in required if c in schema_names]
        q = pd.read_parquet(path, columns=columns)
        q["timestamp"] = pd.to_datetime(q["timestamp"], errors="coerce")
        period = q[(q["timestamp"] >= start_ts) & (q["timestamp"] < end_ts)].copy()
        valid = period.dropna(subset=["aqi"]) if "aqi" in period else pd.DataFrame()
        coverage = 100 * len(valid) / len(period) if len(period) else 0
        poor_share = 100 * (valid["aqi"] > 200).mean() if len(valid) else np.nan
        satisfactory_share = 100 * (valid["aqi"] <= 100).mean() if len(valid) else np.nan
        summaries.append({
            "station": station_name, "mean_aqi": valid["aqi"].mean() if len(valid) else np.nan,
            "median_aqi": valid["aqi"].median() if len(valid) else np.nan,
            "max_aqi": valid["aqi"].max() if len(valid) else np.nan,
            "coverage": coverage, "good_satisfactory_pct": satisfactory_share,
            "poor_plus_pct": poor_share, "observations": len(valid),
        })

        if len(valid):
            cat = valid.get("aqi_category", pd.Series(index=valid.index, dtype=str)).fillna(
                valid["aqi"].map(lambda x: aqi_style(float(x))[0])
            )
            shares = cat.value_counts(normalize=True).mul(100)
            for category_name in [x[2] for x in AQI_BANDS]:
                categories.append({"station": station_name, "category": category_name, "share": shares.get(category_name, 0)})
            h = valid.assign(hour=valid["timestamp"].dt.hour).groupby("hour", as_index=False)["aqi"].mean()
            hourly_rows.extend({"station": station_name, "hour": int(r.hour), "aqi": r.aqi} for r in h.itertuples())
            m = valid.assign(month=valid["timestamp"].dt.month).groupby("month", as_index=False)["aqi"].mean()
            monthly_rows.extend({"station": station_name, "month": int(r.month), "aqi": r.aqi} for r in m.itertuples())

        candidates = q.dropna(subset=["timestamp", "aqi"]).copy() if "aqi" in q else pd.DataFrame()
        if len(candidates):
            candidates["distance"] = (candidates["timestamp"] - snapshot_ts).abs()
            near = candidates.loc[candidates["distance"].idxmin()]
            fresh = near["distance"] <= pd.Timedelta(hours=tolerance_hours)
            aqi_value = float(near["aqi"]) if fresh else np.nan
            category_name, marker_color = aqi_style(aqi_value)
            pollutant_values = {p: near.get(p, np.nan) for p in POLLUTANTS}
            available_pollutants = {p: v for p, v in pollutant_values.items() if pd.notna(v)}
            dominant_name = POLLUTANTS[max(available_pollutants, key=available_pollutants.get)] if available_pollutants else "—"
            snapshots.append({
                "station": station_name, "aqi": aqi_value, "category": category_name if fresh else "No nearby data",
                "timestamp": near["timestamp"] if fresh else pd.NaT, "latitude": near.get("latitude", np.nan),
                "longitude": near.get("longitude", np.nan), "dominant": dominant_name,
                "color": marker_color if fresh else "#8A9691", "size": max(65, min(420, aqi_value * 1.25)) if fresh else 55,
            })

    return (pd.DataFrame(summaries), pd.DataFrame(snapshots), pd.DataFrame(categories),
            pd.DataFrame(hourly_rows), pd.DataFrame(monthly_rows))


@st.cache_data(show_spinner=False, max_entries=4)
def load_city_models(station_records: tuple[tuple[str, str], ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for station_name, folder_name in station_records:
        metrics_path = DATA_DIR / folder_name / "model_metrics.csv"
        predictions_path = DATA_DIR / folder_name / "aqi_predictions.csv"
        if not metrics_path.exists():
            continue
        metrics = pd.read_csv(metrics_path)
        metrics = metrics[metrics.get("target", pd.Series(dtype=str)).astype(str).str.lower().eq("aqi")].copy()
        if not len(metrics):
            continue
        saved_prediction_models = set()
        if predictions_path.exists():
            prediction_table = pd.read_csv(predictions_path, usecols=["model"])
            saved_prediction_models = set(prediction_table["model"].dropna().astype(str).str.lower())
        metrics["station"] = station_name
        metrics["display_model"] = metrics["model_family"].fillna(metrics["model"]).astype(str)
        metrics["row_predictions_saved"] = metrics["display_model"].str.lower().isin(saved_prediction_models)
        best_index = metrics["RMSE"].idxmin()
        metrics["best_at_station"] = metrics.index == best_index
        rows.append(metrics[["station", "display_model", "n", "MAE", "RMSE", "R2", "Bias", "row_predictions_saved", "best_at_station"]])
    detailed = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not len(detailed):
        return detailed, pd.DataFrame()
    aggregate = detailed.groupby("display_model", as_index=False).agg(
        stations=("station", "nunique"), mean_MAE=("MAE", "mean"), mean_RMSE=("RMSE", "mean"),
        mean_R2=("R2", "mean"), mean_Bias=("Bias", "mean"), best_station_count=("best_at_station", "sum"),
        stations_with_saved_rows=("row_predictions_saved", "sum"),
    ).sort_values("mean_RMSE")
    return detailed, aggregate


def hex_to_rgba(value: str) -> list[int]:
    value = value.lstrip("#")
    return [int(value[i:i + 2], 16) for i in (0, 2, 4)] + [210]


REFERENCES = [
    {"key": "Li2016", "citation": "Li et al. (2016)", "title": "A Visualization Approach to Air Pollution Data Exploration—A Case Study of Air Quality Index (PM2.5) in Beijing, China", "journal": "Atmosphere, 7(3), 35", "url": "https://www.mdpi.com/2073-4433/7/3/35", "supports": "Circular heatmaps, calendar views, hourly and monthly AQI patterns"},
    {"key": "Liu2016", "citation": "Liu et al. (2016)", "title": "Temporal Patterns in Fine Particulate Matter Time Series in Beijing: A Calendar View", "journal": "Scientific Reports, 6, 32221", "url": "https://www.nature.com/articles/srep32221", "supports": "Calendar visualization and clustering of daily pollution profiles"},
    {"key": "Qu2017", "citation": "Qu et al. (2017)", "title": "A Visual Analytics Approach for Station-Based Air Quality Data", "journal": "Sensors, 17(1), 30", "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC5298603/", "supports": "Linked map, calendar and trend views; station comparison"},
    {"key": "Lu2017", "citation": "Lu et al. (2017)", "title": "An Interactive Web Mapping Visualization of Urban Air Quality Monitoring Data of China", "journal": "Atmosphere, 8(8), 148", "url": "https://www.mdpi.com/2073-4433/8/8/148", "supports": "Multi-granularity temporal visualization and interactive maps"},
    {"key": "Wright2020", "citation": "Wright and Wernecke (2020)", "title": "Using Microsoft Power BI to Visualise Rustenburg Local Municipality's Air Quality Data", "journal": "Clean Air Journal, 30(1), 1–5", "url": "https://scielo.org.za/scielo.php?pid=S2410-972X2020000100007&script=sci_abstract", "supports": "Compliance, exceedance frequency and site-level data availability"},
    {"key": "Liu2021", "citation": "Liu et al. (2021)", "title": "AQEyes: Visual Analytics for Anomaly Detection and Examination of Air Quality Data", "journal": "arXiv:2103.12910", "url": "https://arxiv.org/abs/2103.12910", "supports": "Multiscale anomaly detection and event examination"},
    {"key": "Lee2022", "citation": "Lee et al. (2022)", "title": "An Online Interactive Dashboard to Explore Personal Exposure to Air Pollution", "journal": "Findings, 2022", "url": "https://findingspress.org/article/49875-an-online-interactive-dashboard-to-explore-personal-exposure-to-air-pollution.pdf", "supports": "Linked maps/time series, temporal filters and guideline comparisons"},
    {"key": "Lissens2022", "citation": "Carro et al. (2022)", "title": "Exploring Actionable Visualizations for Environmental Data: Air Quality Assessment of Two Belgian Locations", "journal": "Environmental Modelling & Software, 147, 105230", "url": "https://doi.org/10.1016/j.envsoft.2021.105230", "supports": "AQI health categories superimposed on concentration trends"},
    {"key": "HD2023", "citation": "Liu et al. (2023)", "title": "High-dimensional Spatiotemporal Visual Analysis of the Air Quality in China", "journal": "Scientific Reports, 13, 5462", "url": "https://www.nature.com/articles/s41598-023-31645-1", "supports": "Ridgeline distributions, maps and high-dimensional comparisons"},
    {"key": "Evolution2023", "citation": "Du et al. (2023)", "title": "Spatiotemporal Evolution Characteristics and Prediction Analysis of Urban Air Quality in China", "journal": "Scientific Reports, 13", "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10235078/", "supports": "Seasonal statistics, pollutant heatmaps and trend decomposition"},
]


def source_note(figure: str, citations: list[str]) -> None:
    matches = [r for r in REFERENCES if r["key"] in citations]
    links = "; ".join(f'<a href="{r["url"]}" target="_blank">{r["citation"]}</a>' for r in matches)
    st.markdown(
        f'<div class="data-note"><b>{figure}</b><br>Visualization design informed by {links}. '
        'The figure is computed from the present project dataset and is not reproduced from the cited publication.</div>',
        unsafe_allow_html=True,
    )


def render_about_page() -> None:
    st.markdown('<div class="hero"><div class="eyebrow">Academic project · 2026</div><h1>About this dashboard</h1><p>Project identity, institutional submission details and acknowledgements.</p></div>', unsafe_allow_html=True)
    st.write("")
    st.markdown("""
<div style="background:white;border:1px solid #e1ebe7;border-radius:22px;padding:34px;box-shadow:0 8px 28px rgba(24,62,50,.06);text-align:center">
<div style="font-size:.85rem;color:#16815d;font-weight:700;letter-spacing:.08em">PROJECT SUBMISSION</div>
<h2 style="line-height:1.35;color:#163d31">DEVELOPMENT OF PHYSICS-GUIDED AI PLATFORM FOR LOCAL AIR QUALITY INDEX PREDICTION AND MONITORING IN INDIAN CITIES</h2>
<p><em>A thesis submitted to the Centre for Distance and Online Education, Andhra University, in partial fulfillment for the award of</em></p>
<h3>MASTER OF COMPUTER APPLICATIONS</h3>
<p><b>By</b><br><b>Rama Siva Kiran Reddy</b><br>Reg. No: A24CA0239</p>
<p><b>Under the Guidance of</b><br><b>Dr. Manish Kumar</b><br>Associate Professor<br>School of Computer Science and Engineering<br>R V University, Bangalore – 560059</p>
<hr style="border:none;border-top:1px solid #e1ebe7;margin:28px 0">
<h3>CENTRE FOR DISTANCE AND ONLINE EDUCATION<br>ANDHRA UNIVERSITY<br>VISAKHAPATNAM</h3>
<h3>2026</h3>
</div>
""", unsafe_allow_html=True)
    st.write("")
    section("Gratitude", "Acknowledgement")
    st.markdown("""
With pride, immense pleasure and a profound sense of gratitude, I take this golden opportunity to express my heartiest and sincere thanks to my research guide, **Dr. Manish Kumar**, Associate Professor, School of Computer Science and Engineering, R V University, Bangalore, for his valuable guidance and encouragement throughout the progress of my thesis work.

I thank **Prof. G. M. Madhu**, Head of the Department, Department of Chemical Engineering, M S Ramaiah Institute of Technology, Bengaluru, for providing the laboratory facilities for my work.

I express my deep sense of gratitude and sincere regards to the team at the **Centre for Distance and Online Education, Andhra University, Visakhapatnam**, for their excellent stewardship towards the successful completion of my thesis work.

My sincere thanks to the **Central Pollution Control Board, New Delhi**, for providing the air-quality data of various Indian cities for the application of Physics-Guided Artificial Intelligence techniques.

I affectionately acknowledge the help and encouragement received from all my friends. Words have no power to express my thanks to my mother and my wife, who have taken every pain to support my studies.

Finally, I thank one and all who helped me in accomplishing my objective.

**Rama Siva Kiran Reddy**
""")
    st.info("This dashboard was prepared as part of the above MCA project submission. It presents historical project data and model results; it is not an official live CPCB service.")


def render_references_page() -> None:
    st.markdown('<div class="hero"><div class="eyebrow">Scholarly foundation</div><h1>References and visualization sources</h1><p>Published literature informing the analytical and visual presentation methods used in this dashboard.</p></div>', unsafe_allow_html=True)
    st.write("")
    st.markdown("The visualizations use **this project’s data**. References acknowledge published methods and presentation designs; figures have not been copied from the articles.")
    query = st.text_input("Search references", placeholder="Try: calendar, stations, compliance, anomaly…")
    refs = REFERENCES
    if query:
        needle = query.lower()
        refs = [r for r in refs if needle in " ".join(r.values()).lower()]
    for number, ref in enumerate(refs, 1):
        st.markdown(
            f'**{number}. {ref["citation"]}.** {ref["title"]}. *{ref["journal"]}*.  '
            f'[Open article or PDF ↗]({ref["url"]})  \n'
            f'<span style="color:#65756f;font-size:.88rem"><b>Supports:</b> {ref["supports"]}</span>',
            unsafe_allow_html=True,
        )
    bibliography = "\n".join(f'{i}. {r["citation"]}. {r["title"]}. {r["journal"]}. {r["url"]}' for i, r in enumerate(REFERENCES, 1))
    st.download_button("Download reference list", bibliography.encode("utf-8"), "air_quality_dashboard_references.txt", "text/plain")


@st.cache_data(show_spinner=False, max_entries=3)
def load_research_sample(records: tuple[tuple[str, str], ...], start_iso: str, end_iso: str) -> pd.DataFrame:
    """Load only analysis columns and aggregate early to protect Community Cloud memory."""
    start_ts, end_ts = pd.Timestamp(start_iso), pd.Timestamp(end_iso)
    frames = []
    wanted = ["timestamp", "aqi", *POLLUTANTS.keys()]
    for station_name, folder_name in records:
        path = DATA_DIR / folder_name / "station_hourly.parquet"
        try:
            import pyarrow.parquet as pq
            available = set(pq.read_schema(path).names)
        except Exception:
            available = set(pd.read_parquet(path, columns=[]).columns)
        cols = [c for c in wanted if c in available]
        q = pd.read_parquet(path, columns=cols)
        q["timestamp"] = pd.to_datetime(q["timestamp"], errors="coerce")
        q = q[(q["timestamp"] >= start_ts) & (q["timestamp"] < end_ts)]
        q["station"] = station_name
        frames.append(q)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def render_research_visuals(city_name: str, city_rows: pd.DataFrame) -> None:
    st.markdown(f'<div class="hero"><div class="eyebrow">Research visual atlas · {city_name}</div><h1>Manuscript-style air-quality analytics</h1><p>Compact, high-information figures designed for station comparison, temporal interpretation and reporting.</p></div>', unsafe_allow_html=True)
    all_stations = sorted(city_rows["station"].unique())
    default_stations = all_stations[:min(8, len(all_stations))]
    chosen_stations = st.multiselect("Stations", all_stations, default=default_stations, max_selections=12, help="Limited to 12 stations to keep the Community Cloud app responsive.")
    if not chosen_stations:
        st.warning("Select at least one station.")
        return
    selected_rows = city_rows[city_rows["station"].isin(chosen_stations)]
    records = tuple((str(r.station), str(r.station_folder).replace("\\", "/")) for r in selected_rows.itertuples())
    min_ts, max_ts = city_date_bounds(tuple(x[1] for x in records))
    suggested_start = max(min_ts.normalize(), max_ts.normalize() - pd.Timedelta(days=365 * 3))
    dates = st.date_input("Analysis period", (suggested_start.date(), max_ts.date()), min_value=min_ts.date(), max_value=max_ts.date(), key="research_dates")
    if not isinstance(dates, tuple) or len(dates) != 2:
        st.info("Choose a start and end date.")
        return
    start_ts, end_ts = pd.Timestamp(dates[0]), pd.Timestamp(dates[1]) + pd.Timedelta(days=1)
    with st.spinner("Preparing research visualizations…"):
        raw = load_research_sample(records, start_ts.isoformat(), end_ts.isoformat())
    if raw.empty:
        st.warning("No observations are available for this selection.")
        return
    available = [p for p in POLLUTANTS if p in raw and raw[p].notna().any()]
    dot_tab, calendar_tab, distribution_tab, association_tab = st.tabs(["Dot and bubble tables", "Temporal heatmaps", "Distributions", "Associations"])
    with dot_tab:
        section("Figure 1", "Station × pollutant bubble table")
        means = raw.groupby("station")[available].mean().reset_index().melt("station", var_name="pollutant", value_name="mean")
        means["label"] = means["pollutant"].map(POLLUTANTS)
        means["relative"] = means.groupby("pollutant")["mean"].rank(pct=True).mul(100)
        bubble = alt.Chart(means.dropna()).mark_circle(stroke="white", strokeWidth=1.5).encode(
            x=alt.X("label:N", title=None), y=alt.Y("station:N", title=None),
            size=alt.Size("relative:Q", scale=alt.Scale(range=[80, 1100]), title="Within-pollutant percentile"),
            color=alt.Color("relative:Q", scale=alt.Scale(scheme="yelloworangered"), title="Relative level"),
            tooltip=["station:N", alt.Tooltip("label:N", title="Pollutant"), alt.Tooltip("mean:Q", format=".2f", title="Mean"), alt.Tooltip("relative:Q", format=".0f", title="Percentile")],
        ).properties(height=max(330, len(chosen_stations) * 34))
        st.altair_chart(bubble, use_container_width=True)
        st.caption("Dot size and colour compare stations within each pollutant. Raw concentrations of different pollutants are not treated as directly comparable.")
        source_note("Figure 1. Station–pollutant bubble table", ["Qu2017", "HD2023"])

        section("Figure 2", "Ranked station dot plot")
        station_stats = raw.groupby("station")["aqi"].agg(mean="mean", median="median", q1=lambda x: x.quantile(.25), q3=lambda x: x.quantile(.75), observations="count").reset_index().dropna(subset=["mean"])
        base = alt.Chart(station_stats).encode(y=alt.Y("station:N", sort="-x", title=None))
        intervals = base.mark_rule(strokeWidth=3, color="#9ebbb0").encode(x=alt.X("q1:Q", title="AQI"), x2="q3:Q")
        dots = base.mark_circle(size=150, color="#147a59").encode(x="mean:Q", tooltip=["station:N", alt.Tooltip("mean:Q", format=".1f"), alt.Tooltip("median:Q", format=".1f"), "observations:Q"])
        st.altair_chart((intervals + dots).properties(height=max(320, len(chosen_stations) * 34)), use_container_width=True)
        source_note("Figure 2. Mean AQI with interquartile range", ["Wright2020", "Qu2017"])

    with calendar_tab:
        pollutant = st.selectbox("Temporal variable", ["aqi", *available], format_func=lambda x: "AQI" if x == "aqi" else POLLUTANTS[x], key="temporal_variable")
        daily = raw.assign(date=raw["timestamp"].dt.floor("D")).groupby("date", as_index=False)[pollutant].mean().dropna()
        daily["weekday"] = daily["date"].dt.weekday
        daily["week"] = daily["date"].dt.strftime("%U").astype(int)
        daily["year"] = daily["date"].dt.year
        section("Figure 3", "Calendar heatmap")
        calendar = alt.Chart(daily).mark_rect(cornerRadius=2).encode(
            x=alt.X("week:O", title="Week of year", axis=alt.Axis(labels=False, ticks=False)),
            y=alt.Y("weekday:O", title=None, sort=[0,1,2,3,4,5,6], axis=alt.Axis(labelExpr="['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][datum.value]")),
            row=alt.Row("year:N", title=None), color=alt.Color(f"{pollutant}:Q", scale=alt.Scale(scheme="yelloworangered")),
            tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip(f"{pollutant}:Q", format=".1f")],
        ).properties(height=95)
        st.altair_chart(calendar, use_container_width=True)
        source_note("Figure 3. Daily calendar heatmap", ["Li2016", "Liu2016", "Qu2017"])

        section("Figure 4", "Hour × month heatmap")
        hm = raw.assign(hour=raw["timestamp"].dt.hour, month=raw["timestamp"].dt.month).groupby(["month", "hour"], as_index=False)[pollutant].mean()
        heat = alt.Chart(hm).mark_rect().encode(x=alt.X("hour:O", title="Hour"), y=alt.Y("month:O", title="Month"), color=alt.Color(f"{pollutant}:Q", scale=alt.Scale(scheme="yelloworangered")), tooltip=["month:O", "hour:O", alt.Tooltip(f"{pollutant}:Q", format=".1f")]).properties(height=330)
        st.altair_chart(heat, use_container_width=True)
        source_note("Figure 4. Diurnal and monthly pattern matrix", ["Li2016", "Evolution2023"])

    with distribution_tab:
        variable = st.selectbox("Distribution variable", ["aqi", *available], format_func=lambda x: "AQI" if x == "aqi" else POLLUTANTS[x], key="distribution_variable")
        sample = raw[["station", "timestamp", variable]].dropna()
        sample = downsample(sample, 12000)
        sample["season"] = sample["timestamp"].dt.month.map({12:"Winter",1:"Winter",2:"Winter",3:"Pre-monsoon",4:"Pre-monsoon",5:"Pre-monsoon",6:"Monsoon",7:"Monsoon",8:"Monsoon",9:"Monsoon",10:"Post-monsoon",11:"Post-monsoon"})
        section("Figure 5", "Seasonal box-and-point distribution")
        box = alt.Chart(sample).mark_boxplot(size=34, extent="min-max").encode(x=alt.X("season:N", sort=["Winter","Pre-monsoon","Monsoon","Post-monsoon"], title=None), y=alt.Y(f"{variable}:Q", title="AQI" if variable == "aqi" else POLLUTANTS[variable]), color=alt.Color("season:N", legend=None, scale=alt.Scale(range=["#5871a8","#e4a43b","#32a879","#b76450"])), tooltip=["season:N"])
        st.altair_chart(box.properties(height=390), use_container_width=True)
        source_note("Figure 5. Seasonal distribution", ["Evolution2023", "HD2023"])

    with association_tab:
        numeric = ["aqi", *available]
        corr = raw[numeric].corr(method="spearman").stack().rename("correlation").reset_index()
        corr.columns = ["variable_1", "variable_2", "correlation"]
        labels = {"aqi":"AQI", **POLLUTANTS}
        corr["variable_1"] = corr["variable_1"].map(labels)
        corr["variable_2"] = corr["variable_2"].map(labels)
        corr["strength"] = corr["correlation"].abs()
        section("Figure 6", "Correlation bubble matrix")
        matrix = alt.Chart(corr).mark_circle(stroke="#ffffff", strokeWidth=1).encode(
            x=alt.X("variable_1:N", title=None), y=alt.Y("variable_2:N", title=None),
            size=alt.Size("strength:Q", scale=alt.Scale(range=[20, 1100]), legend=None),
            color=alt.Color("correlation:Q", scale=alt.Scale(domain=[-1,0,1], range=["#315a9a","#f3f3ef","#b83745"]), title="Spearman ρ"),
            tooltip=[alt.Tooltip("variable_1:N", title="Variable 1"), alt.Tooltip("variable_2:N", title="Variable 2"), alt.Tooltip("correlation:Q", format=".3f")],
        ).properties(height=440)
        st.altair_chart(matrix, use_container_width=True)
        source_note("Figure 6. Spearman correlation dot matrix", ["HD2023", "Evolution2023"])


def render_city_overview(city_name: str, city_rows: pd.DataFrame, full_index: pd.DataFrame) -> None:
    records = tuple((str(r.station), str(r.station_folder).replace("\\", "/")) for r in city_rows.itertuples())
    min_ts, max_ts = city_date_bounds(tuple(x[1] for x in records))
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### City comparison period")
    selected_dates = st.sidebar.date_input(
        "COMPARISON RANGE", (min_ts.date(), max_ts.date()), min_value=min_ts.date(), max_value=max_ts.date(), key="city_dates"
    )
    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        range_start = pd.Timestamp(selected_dates[0])
        range_end = pd.Timestamp(selected_dates[1]) + pd.Timedelta(days=1)
    else:
        range_start, range_end = min_ts.normalize(), max_ts.normalize() + pd.Timedelta(days=1)
    snapshot_date = st.sidebar.date_input("MAP DATE", max_ts.date(), min_value=min_ts.date(), max_value=max_ts.date())
    snapshot_hour = st.sidebar.slider("MAP HOUR", 0, 23, int(max_ts.hour))
    tolerance = st.sidebar.select_slider("NEAREST-DATA TOLERANCE", options=[1, 3, 6, 12, 24], value=6, format_func=lambda x: f"±{x} hours")
    snapshot_ts = pd.Timestamp(snapshot_date) + pd.Timedelta(hours=snapshot_hour)

    with st.spinner(f"Comparing {len(records)} {city_name} stations…"):
        summary, snapshot, category_share, hourly, monthly = build_city_analysis(
            records, range_start.isoformat(), range_end.isoformat(), snapshot_ts.isoformat(), tolerance
        )

    valid_snapshot = snapshot.dropna(subset=["aqi"])
    city_aqi = valid_snapshot["aqi"].median() if len(valid_snapshot) else np.nan
    city_category, city_color = aqi_style(city_aqi)
    best_now = valid_snapshot.sort_values("aqi").iloc[0]["station"] if len(valid_snapshot) else "—"
    worst_now = valid_snapshot.sort_values("aqi", ascending=False).iloc[0]["station"] if len(valid_snapshot) else "—"
    reporting = len(valid_snapshot)

    st.markdown(
        f'<div class="hero"><div class="eyebrow">City intelligence · {city_name}</div><h1>All-station air quality overview</h1>'
        f'<p>Compare {len(records)} stations fairly using a shared period and a time-aligned map snapshot.</p></div>',
        unsafe_allow_html=True,
    )
    st.write("")
    render_study_area(full_index)
    st.markdown("---")
    section("Selected-city dashboard", f"Detailed analysis for {city_name}")
    cards = st.columns(5)
    with cards[0]: metric_card("◎", "City median AQI", f"{city_aqi:.0f}" if pd.notna(city_aqi) else "—", city_category)
    with cards[1]: metric_card("↘", "Cleanest at snapshot", best_now, "Lowest available AQI")
    with cards[2]: metric_card("↗", "Highest at snapshot", worst_now, "Highest available AQI")
    with cards[3]: metric_card("●", "Stations reporting", f"{reporting}/{len(records)}", f"Within ±{tolerance} hours")
    with cards[4]: metric_card("◷", "Snapshot", snapshot_ts.strftime("%d %b · %H:%M"), "Shared comparison time")
    st.markdown(f'<div class="health-card" style="background:{city_color}"><b>{city_category} city snapshot</b> — Based on the median of stations reporting near the selected hour.</div>', unsafe_allow_html=True)

    map_tab, rank_tab, patterns_tab, burden_tab, model_tab, reliability_tab = st.tabs(
        ["Station map", "Station rankings", "Time patterns", "Pollution burden", "City model lab", "Reliability"]
    )
    with map_tab:
        section("Spatial snapshot", "AQI status across all monitoring stations")
        map_data = snapshot.dropna(subset=["latitude", "longitude"]).copy()
        if len(map_data):
            map_data["fill_color"] = map_data["color"].map(hex_to_rgba)
            map_data["aqi_display"] = map_data["aqi"].map(lambda x: f"{x:.0f}" if pd.notna(x) else "Unavailable")
            map_data["time_display"] = map_data["timestamp"].map(
                lambda x: x.strftime("%d %b %Y · %H:%M") if pd.notna(x) else "No nearby observation"
            )
            layer = pdk.Layer(
                "ScatterplotLayer", map_data, get_position="[longitude, latitude]", get_fill_color="fill_color",
                get_radius="size * 2.4", radius_min_pixels=9, radius_max_pixels=28,
                pickable=True, auto_highlight=True, stroked=True, get_line_color=[255, 255, 255, 230], line_width_min_pixels=2,
            )
            view_state = pdk.ViewState(
                latitude=float(map_data["latitude"].mean()), longitude=float(map_data["longitude"].mean()), zoom=10.3, pitch=0
            )
            tooltip = {
                "html": "<div style='font-family:Arial;padding:4px 6px'>"
                        "<b style='font-size:15px'>{station}</b><br/>"
                        "<span style='font-size:22px;font-weight:700'>AQI {aqi_display}</span><br/>"
                        "<b>Status:</b> {category}<br/><b>Dominant:</b> {dominant}<br/>"
                        "<b>Observation:</b> {time_display}</div>",
                "style": {"backgroundColor": "#10261f", "color": "white", "borderRadius": "10px"},
            }
            deck = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip, map_style=None)
            st.pydeck_chart(deck, use_container_width=True)
        else:
            st.warning("No station coordinates are available.")
        st.caption("Hover over or click a marker to see station, AQI, category, dominant concentration and matched observation time. Grey markers have no nearby AQI value.")
        display = snapshot[["station", "aqi", "category", "dominant", "timestamp"]].copy()
        display["aqi"] = display["aqi"].round(0)
        display = display.sort_values("aqi", ascending=False, na_position="last")
        st.dataframe(display, use_container_width=True, hide_index=True, column_config={
            "station": "Station", "aqi": st.column_config.NumberColumn("AQI", format="%.0f"),
            "category": "Status", "dominant": "Dominant concentration", "timestamp": "Matched observation",
        })

    with rank_tab:
        section("Fair comparison", "Station ranking for the selected period")
        min_coverage = st.slider("Minimum AQI coverage required for ranking", 0, 90, 25, 5)
        metric_choice = st.selectbox("Rank stations by", ["Mean AQI", "Median AQI", "Good + satisfactory hours", "Poor-or-worse hours"])
        metric_map = {"Mean AQI": ("mean_aqi", True), "Median AQI": ("median_aqi", True),
                      "Good + satisfactory hours": ("good_satisfactory_pct", False), "Poor-or-worse hours": ("poor_plus_pct", True)}
        metric_col, ascending = metric_map[metric_choice]
        eligible = summary[summary["coverage"] >= min_coverage].sort_values(metric_col, ascending=ascending).copy()
        excluded_count = len(summary) - len(eligible)
        if len(eligible):
            eligible["rank"] = np.arange(1, len(eligible) + 1)
            rank_chart = alt.Chart(eligible).mark_bar(cornerRadiusEnd=7).encode(
                y=alt.Y("station:N", sort=alt.EncodingSortField(field=metric_col, order="ascending" if ascending else "descending"), title=None),
                x=alt.X(f"{metric_col}:Q", title=metric_choice),
                color=alt.Color(f"{metric_col}:Q", scale=alt.Scale(range=["#65bd9d", "#e3a33b", "#d45858"]), legend=None),
                tooltip=["station:N", alt.Tooltip(f"{metric_col}:Q", format=".1f"), alt.Tooltip("coverage:Q", format=".1f")],
            ).properties(height=max(300, len(eligible) * 29))
            st.altair_chart(rank_chart, use_container_width=True)
            winner = eligible.iloc[0]
            st.success(f"Best on **{metric_choice.lower()}**: **{winner['station']}**. This ranking includes only stations with at least {min_coverage}% AQI coverage.")
        else:
            st.warning("No stations meet the selected coverage requirement.")
        if excluded_count:
            st.caption(f"{excluded_count} station(s) excluded because their AQI coverage was below {min_coverage}%.")

    with patterns_tab:
        section("Recurring behaviour", "Station-by-hour AQI heatmap")
        if len(hourly):
            hour_heatmap = alt.Chart(hourly).mark_rect().encode(
                x=alt.X("hour:O", title="Hour of day"), y=alt.Y("station:N", title=None),
                color=alt.Color("aqi:Q", scale=alt.Scale(scheme="yelloworangered"), title="Mean AQI"),
                tooltip=["station:N", "hour:O", alt.Tooltip("aqi:Q", format=".1f")],
            ).properties(height=max(300, len(records) * 28))
            st.altair_chart(hour_heatmap, use_container_width=True)
        section("Seasonal behaviour", "Station-by-month AQI heatmap")
        if len(monthly):
            month_heatmap = alt.Chart(monthly).mark_rect().encode(
                x=alt.X("month:O", title="Month", axis=alt.Axis(labelExpr="['','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][datum.value]")),
                y=alt.Y("station:N", title=None), color=alt.Color("aqi:Q", scale=alt.Scale(scheme="yelloworangered"), title="Mean AQI"),
                tooltip=["station:N", "month:O", alt.Tooltip("aqi:Q", format=".1f")],
            ).properties(height=max(300, len(records) * 28))
            st.altair_chart(month_heatmap, use_container_width=True)

    with burden_tab:
        section("Health exposure", "AQI category share at each station")
        if len(category_share):
            order = [x[2] for x in AQI_BANDS]
            colors = [x[3] for x in AQI_BANDS]
            burden = alt.Chart(category_share).mark_bar().encode(
                y=alt.Y("station:N", title=None), x=alt.X("share:Q", stack="normalize", title="Share of observed hours"),
                color=alt.Color("category:N", scale=alt.Scale(domain=order, range=colors), sort=order, title="AQI category"),
                order=alt.Order("category:N", sort="ascending"), tooltip=["station:N", "category:N", alt.Tooltip("share:Q", format=".1f")],
            ).properties(height=max(300, len(records) * 29))
            st.altair_chart(burden, use_container_width=True)
        burden_table = summary[["station", "good_satisfactory_pct", "poor_plus_pct", "max_aqi"]].sort_values("poor_plus_pct")
        st.dataframe(burden_table, use_container_width=True, hide_index=True, column_config={
            "station": "Station", "good_satisfactory_pct": st.column_config.NumberColumn("Good + satisfactory (%)", format="%.1f"),
            "poor_plus_pct": st.column_config.NumberColumn("Poor or worse (%)", format="%.1f"),
            "max_aqi": st.column_config.NumberColumn("Maximum AQI", format="%.0f"),
        })

    with model_tab:
        section("City-wide model evidence", "AQI model performance at every station")
        detailed_models, aggregate_models = load_city_models(records)
        if len(detailed_models):
            best_city_model = aggregate_models.iloc[0]
            mc1, mc2, mc3 = st.columns(3)
            with mc1: metric_card("✦", "Lowest mean RMSE", str(best_city_model["display_model"]), f'{best_city_model["mean_RMSE"]:.2f} across stations')
            with mc2: metric_card("▦", "Models compared", f'{detailed_models["display_model"].nunique()}', "AQI validation models")
            with mc3: metric_card("✓", "Station-model results", f"{len(detailed_models):,}", f"Across {detailed_models['station'].nunique()} stations")

            section("Model comparison", "Mean validation performance across the city")
            aggregate_display = aggregate_models.rename(columns={
                "display_model": "Model", "stations": "Stations", "mean_MAE": "Mean MAE", "mean_RMSE": "Mean RMSE",
                "mean_R2": "Mean R²", "mean_Bias": "Mean bias", "best_station_count": "Best at stations",
                "stations_with_saved_rows": "Stations with row predictions",
            })
            st.dataframe(aggregate_display, use_container_width=True, hide_index=True, column_config={
                "Mean MAE": st.column_config.NumberColumn(format="%.2f"), "Mean RMSE": st.column_config.NumberColumn(format="%.2f"),
                "Mean R²": st.column_config.NumberColumn(format="%.3f"), "Mean bias": st.column_config.NumberColumn(format="%.2f"),
            })

            section("Station × model", "RMSE heatmap")
            rmse_heatmap = alt.Chart(detailed_models).mark_rect().encode(
                x=alt.X("display_model:N", title="Model"), y=alt.Y("station:N", title=None),
                color=alt.Color("RMSE:Q", scale=alt.Scale(scheme="yelloworangered"), title="RMSE"),
                tooltip=["station:N", alt.Tooltip("display_model:N", title="Model"), alt.Tooltip("RMSE:Q", format=".2f"),
                         alt.Tooltip("MAE:Q", format=".2f"), alt.Tooltip("R2:Q", format=".3f")],
            ).properties(height=max(320, len(records) * 28))
            st.altair_chart(rmse_heatmap, use_container_width=True)

            section("Complete evidence table", "Every station and AQI model")
            evidence = detailed_models.rename(columns={
                "station": "Station", "display_model": "Model", "n": "Validation rows", "row_predictions_saved": "Row predictions saved",
                "best_at_station": "Best at station",
            }).sort_values(["Station", "RMSE"])
            st.dataframe(evidence, use_container_width=True, hide_index=True, column_config={
                "MAE": st.column_config.NumberColumn(format="%.2f"), "RMSE": st.column_config.NumberColumn(format="%.2f"),
                "R2": st.column_config.NumberColumn("R²", format="%.3f"), "Bias": st.column_config.NumberColumn(format="%.2f"),
                "Row predictions saved": st.column_config.CheckboxColumn(), "Best at station": st.column_config.CheckboxColumn(),
            })
            st.download_button("Download city model evidence as CSV", evidence.to_csv(index=False).encode("utf-8"), f"{city_name}_model_evidence.csv".replace(" ", "_"), "text/csv")
            st.info("All seven models have aggregate validation metrics. Row-level observed/predicted values are saved only for models present in each station's aqi_predictions.csv—currently usually GRU, PINN and PIIANN.")
        else:
            st.warning("No station-level AQI model metrics are available for this city.")

    with reliability_tab:
        section("Evidence quality", "AQI coverage by station")
        coverage_chart = alt.Chart(summary).mark_bar(cornerRadiusEnd=7).encode(
            y=alt.Y("station:N", sort="-x", title=None), x=alt.X("coverage:Q", title="AQI coverage (%)", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("coverage:Q", scale=alt.Scale(domain=[0, 100], range=["#d9e4df", "#15a776"]), legend=None),
            tooltip=["station:N", alt.Tooltip("coverage:Q", format=".1f"), alt.Tooltip("observations:Q", format=",")],
        ).properties(height=max(300, len(records) * 29))
        st.altair_chart(coverage_chart, use_container_width=True)
        st.warning("A cleaner-looking station is not automatically better if it has sparse data. Rankings should always be reported with coverage and a common time period.")
        export = summary.merge(snapshot[["station", "aqi", "category", "timestamp"]], on="station", how="left", suffixes=("_period", "_snapshot"))
        st.download_button("Download city comparison as CSV", export.to_csv(index=False).encode("utf-8"), f"{city_name}_station_comparison.csv".replace(" ", "_"), "text/csv")


index = load_index()
st.sidebar.markdown("## PIIANN AQI Dashboard")
st.sidebar.caption("Physics-guided air-quality intelligence")
app_page = st.sidebar.radio(
    "NAVIGATION",
    ["City dashboard", "Station explorer", "Research visuals", "References", "About & acknowledgement"],
    index=0,
)
if app_page == "References":
    render_references_page()
    st.stop()
if app_page == "About & acknowledgement":
    render_about_page()
    st.stop()

preferred_city_order = ["Delhi", "Mumbai", "Kolkata", "Chennai", "Hyderabad", "Bengaluru", "Jaipur"]
available_cities = list(index["city"].dropna().unique())
ordered_cities = [c for c in preferred_city_order if c in available_cities] + sorted(c for c in available_cities if c not in preferred_city_order)
city = st.sidebar.selectbox("CITY", ordered_cities)
city_index = index[index["city"] == city]
if app_page == "Research visuals":
    render_research_visuals(city, city_index)
    st.markdown("---")
    st.caption("Research visualizations are calculated from project data. See References for the published design sources.")
    st.stop()
if app_page == "City dashboard":
    render_city_overview(city, city_index, index)
    st.markdown("---")
    st.caption("PIIANN AQI Dashboard · City comparisons use a common period, shared snapshot time and explicit data-coverage safeguards. See References for visualization sources.")
    st.stop()
station = st.sidebar.selectbox("MONITORING STATION", sorted(city_index["station"].unique()))
row = city_index[city_index["station"] == station].iloc[0]
folder = str(row["station_folder"]).replace("\\", "/")
df, config = load_station(folder)

st.sidebar.markdown("---")
st.sidebar.markdown("#### Time window")
min_date, max_date = df["timestamp"].min().date(), df["timestamp"].max().date()
date_range = st.sidebar.date_input("DATE RANGE", (min_date, max_date), min_value=min_date, max_value=max_date)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
else:
    start, end = pd.Timestamp(min_date), pd.Timestamp(max_date) + pd.Timedelta(days=1)
filtered = df[(df["timestamp"] >= start) & (df["timestamp"] < end)].copy()

st.sidebar.markdown("---")
st.sidebar.caption("Historical/model-results dashboard · Not a live CPCB feed")

st.markdown(
    f'<div class="hero"><div class="eyebrow">Air quality intelligence · {city}</div>'
    f'<h1>{station}</h1><p>Explore hourly air quality, pollutant behaviour and transparent model performance '
    f'from {min_date:%d %b %Y} to {max_date:%d %b %Y}.</p></div>',
    unsafe_allow_html=True,
)
st.write("")

aqi, aqi_time = latest_value(filtered, "aqi")
category, category_color = aqi_style(aqi)
dominant = "—"
pollutant_latest = {}
for key, label in POLLUTANTS.items():
    value, _ = latest_value(filtered, key)
    pollutant_latest[key] = value
if any(pd.notna(v) for v in pollutant_latest.values()):
    dominant = POLLUTANTS[max((k for k, v in pollutant_latest.items() if pd.notna(v)), key=lambda k: pollutant_latest[k])]

cols = st.columns(5)
with cols[0]:
    metric_card("◉", "Latest available AQI", f"{aqi:.0f}" if pd.notna(aqi) else "—", category)
with cols[1]:
    metric_card("⌁", "Dominant concentration", dominant, "Latest available hour")
with cols[2]:
    metric_card("◷", "Average AQI", f"{filtered['aqi'].mean():.1f}" if "aqi" in filtered else "—", "Selected period")
with cols[3]:
    poor_hours = int((filtered.get("aqi", pd.Series(dtype=float)) > 200).sum())
    metric_card("⚠", "Poor+ hours", f"{poor_hours:,}", "AQI above 200")
with cols[4]:
    coverage = filtered["aqi"].notna().mean() * 100 if len(filtered) and "aqi" in filtered else 0
    metric_card("✓", "AQI coverage", f"{coverage:.1f}%", f"{len(filtered):,} timeline hours")

st.markdown(
    f'<div class="health-card" style="background:{category_color}"><b>{category} air quality</b> &nbsp;—&nbsp; '
    f'{HEALTH.get(category, "No guidance available.")}</div>',
    unsafe_allow_html=True,
)

overview_tab, trends_tab, pollutants_tab, models_tab, quality_tab, stats_tab, patterns_tab, advanced_tab = st.tabs(
    ["Overview", "AQI trends", "Pollutants", "Model lab", "Data quality", "Statistical atlas", "Pattern atlas", "Advanced analytics"]
)

with overview_tab:
    left, right = st.columns([1.6, 1])
    with left:
        section("Selected period", "Hourly AQI trajectory")
        chart_df = downsample(filtered[["timestamp", "aqi"]].dropna())
        line = (
            alt.Chart(chart_df)
            .mark_area(line={"color": "#16815d", "strokeWidth": 2}, color=alt.Gradient(
                gradient="linear", stops=[alt.GradientStop(color="#35b98b", offset=0), alt.GradientStop(color="#ffffff", offset=1)], x1=1, x2=1, y1=0, y2=1
            ))
            .encode(
                x=alt.X("timestamp:T", title=None),
                y=alt.Y("aqi:Q", title="AQI", scale=alt.Scale(zero=False)),
                tooltip=[alt.Tooltip("timestamp:T", title="Time"), alt.Tooltip("aqi:Q", format=".0f", title="AQI")],
            ).properties(height=350).interactive()
        )
        st.altair_chart(line, use_container_width=True)
    with right:
        section("Location", "Monitoring station")
        coords = filtered[[c for c in ["latitude", "longitude"] if c in filtered]].dropna().tail(1)
        if set(coords.columns) == {"latitude", "longitude"} and len(coords):
            st.map(coords, latitude="latitude", longitude="longitude", zoom=11, size=120, color="#15B87A")
        else:
            st.info("Station coordinates are unavailable.")
        stamp = aqi_time.strftime("%d %b %Y · %H:%M") if aqi_time is not None else "Unavailable"
        st.markdown(f'<div class="data-note"><b>Latest selected observation</b><br>{stamp}<br><br>Values shown are historical results, not a live sensor feed.</div>', unsafe_allow_html=True)

    section("Exposure profile", "AQI category distribution")
    cats = filtered.get("aqi_category", pd.Series(dtype=str)).value_counts().rename_axis("category").reset_index(name="hours")
    order = [x[2] for x in AQI_BANDS]
    colors = [x[3] for x in AQI_BANDS]
    cat_chart = alt.Chart(cats).mark_bar(cornerRadiusTopLeft=7, cornerRadiusTopRight=7).encode(
        x=alt.X("category:N", sort=order, title=None), y=alt.Y("hours:Q", title="Hours"),
        color=alt.Color("category:N", scale=alt.Scale(domain=order, range=colors), legend=None),
        tooltip=["category:N", alt.Tooltip("hours:Q", format=",")],
    ).properties(height=270)
    st.altair_chart(cat_chart, use_container_width=True)

with trends_tab:
    section("Temporal intelligence", "Explore recurring AQI patterns")
    granularity = st.segmented_control("Aggregation", ["Hourly", "Daily", "Monthly"], default="Daily")
    rule = {"Hourly": "h", "Daily": "D", "Monthly": "MS"}[granularity]
    agg = filtered.set_index("timestamp")["aqi"].resample(rule).agg(["mean", "min", "max"]).dropna().reset_index()
    base = alt.Chart(downsample(agg)).encode(x=alt.X("timestamp:T", title=None))
    band = base.mark_area(opacity=.15, color="#19a978").encode(y="min:Q", y2="max:Q")
    mean_line = base.mark_line(color="#11654b", strokeWidth=2.5).encode(
        y=alt.Y("mean:Q", title="AQI"), tooltip=["timestamp:T", alt.Tooltip("mean:Q", format=".1f")]
    )
    st.altair_chart((band + mean_line).properties(height=410).interactive(), use_container_width=True)
    c1, c2 = st.columns(2)
    hourly = filtered.assign(hour=filtered["timestamp"].dt.hour).groupby("hour", as_index=False)["aqi"].mean()
    monthly = filtered.assign(month=filtered["timestamp"].dt.month_name().str[:3]).groupby("month", as_index=False)["aqi"].mean()
    with c1:
        section("Daily rhythm", "Mean AQI by hour")
        st.altair_chart(alt.Chart(hourly).mark_line(point=True, color="#15a776").encode(x=alt.X("hour:Q", title="Hour"), y=alt.Y("aqi:Q", title="Mean AQI"), tooltip=["hour", alt.Tooltip("aqi", format=".1f")]).properties(height=270), use_container_width=True)
    with c2:
        section("Seasonality", "Mean AQI by month")
        months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        st.altair_chart(alt.Chart(monthly).mark_bar(color="#2d8f70", cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(x=alt.X("month:N", sort=months, title=None), y=alt.Y("aqi:Q", title="Mean AQI"), tooltip=["month", alt.Tooltip("aqi", format=".1f")]).properties(height=270), use_container_width=True)

with pollutants_tab:
    section("Pollutant explorer", "Compare hourly concentration patterns")
    available = [k for k in POLLUTANTS if k in filtered and filtered[k].notna().any()]
    chosen = st.multiselect("Pollutants", available, default=available[:2], format_func=lambda x: POLLUTANTS[x], max_selections=4)
    if chosen:
        long = downsample(filtered[["timestamp", *chosen]].dropna(how="all")).melt("timestamp", var_name="pollutant", value_name="concentration").dropna()
        long["pollutant"] = long["pollutant"].map(POLLUTANTS)
        chart = alt.Chart(long).mark_line(strokeWidth=1.6).encode(
            x=alt.X("timestamp:T", title=None), y=alt.Y("concentration:Q", title="Concentration"),
            color=alt.Color("pollutant:N", scale=alt.Scale(range=["#15a776", "#e5a72e", "#d35f5f", "#486fa8"]), title=None),
            tooltip=["timestamp:T", "pollutant:N", alt.Tooltip("concentration:Q", format=".2f")]
        ).properties(height=400).interactive()
        st.altair_chart(chart, use_container_width=True)
    st.caption("Concentrations are displayed in the units supplied by the source bundle. CO may use a different scale from particulate pollutants.")

with models_tab:
    section("Physics-informed ML", "Transparent model validation")
    metrics = load_optional_csv(folder, "model_metrics.csv")
    aqi_metrics = metrics[metrics.get("target", pd.Series(dtype=str)).astype(str).str.lower() == "aqi"].copy() if len(metrics) else metrics
    if len(aqi_metrics):
        aqi_metrics["display_model"] = aqi_metrics["model_family"].fillna(aqi_metrics["model"])
        best = aqi_metrics.sort_values("RMSE").iloc[0]
        m1, m2, m3 = st.columns(3)
        with m1: metric_card("✦", "Best validation model", str(best["display_model"]), "Lowest RMSE")
        with m2: metric_card("↓", "Validation RMSE", f'{best["RMSE"]:.2f}', "Lower is better")
        with m3: metric_card("↗", "Validation R²", f'{best["R2"]:.3f}', "Higher is better")
        metric_chart = alt.Chart(aqi_metrics).mark_bar(cornerRadiusEnd=6).encode(
            y=alt.Y("display_model:N", sort="x", title=None), x=alt.X("RMSE:Q", title="RMSE"),
            color=alt.condition(alt.datum.display_model == str(best["display_model"]), alt.value("#15a776"), alt.value("#b9cbc4")),
            tooltip=["display_model:N", alt.Tooltip("RMSE:Q", format=".2f"), alt.Tooltip("MAE:Q", format=".2f"), alt.Tooltip("R2:Q", format=".3f")]
        ).properties(height=320)
        st.altair_chart(metric_chart, use_container_width=True)
    preds = load_optional_csv(folder, "aqi_predictions.csv")
    if len(aqi_metrics) and len(preds):
        saved_model_names = set(preds["model"].dropna().astype(str))
        diagnostic_models = [name for name in ["GRU", "PINN", "PIIANN"] if name in saved_model_names]
        model = st.selectbox("Model diagnostics", diagnostic_models, help="Only models with saved point-level predictions are shown here.")
        selected_metric = aqi_metrics[aqi_metrics["display_model"].astype(str).eq(model)].iloc[0]
        d1, d2, d3, d4, d5 = st.columns(5)
        with d1: metric_card("N", "Validation rows", f'{int(selected_metric["n"]):,}', model)
        with d2: metric_card("E", "MAE", f'{selected_metric["MAE"]:.2f}', "Lower is better")
        with d3: metric_card("R", "RMSE", f'{selected_metric["RMSE"]:.2f}', "Lower is better")
        with d4: metric_card("R²", "Coefficient", f'{selected_metric["R2"]:.3f}', "Higher is better")
        with d5: metric_card("B", "Bias", f'{selected_metric["Bias"]:.2f}', "Closer to zero")
    if len(preds) and "model" in locals():
        q = preds[preds["model"] == model].copy()
        if len(q):
            section("Point-level validation", f"Observed versus predicted · {model}")
            scatter = alt.Chart(downsample(q, 5000)).mark_circle(opacity=.42, color="#16815d").encode(
            x=alt.X("observed:Q", title="Observed AQI"), y=alt.Y("predicted:Q", title="Predicted AQI"),
            tooltip=[alt.Tooltip("observed:Q", format=".1f"), alt.Tooltip("predicted:Q", format=".1f")]
            ).properties(height=380).interactive()
            diagonal = alt.Chart(pd.DataFrame({"x": [q.observed.min(), q.observed.max()]})).mark_line(color="#e28b38", strokeDash=[5,5]).encode(x="x:Q", y="x:Q")
            st.altair_chart(scatter + diagonal, use_container_width=True)
            q["residual"] = q["predicted"] - q["observed"]
            q["observed_category"] = q["observed"].map(lambda x: aqi_style(float(x))[0])
            q["predicted_category"] = q["predicted"].map(lambda x: aqi_style(float(x))[0])
            left_diag, right_diag = st.columns(2)
            with left_diag:
                section("Error structure", "Residual distribution")
                residual_chart = alt.Chart(q).mark_bar(color="#2d8f70").encode(x=alt.X("residual:Q", bin=alt.Bin(maxbins=45), title="Predicted − observed AQI"), y=alt.Y("count():Q", title="Rows"), tooltip=["count():Q"])
                st.altair_chart(residual_chart.properties(height=300), use_container_width=True)
            with right_diag:
                section("Classification view", "AQI-category confusion matrix")
                confusion = q.groupby(["observed_category", "predicted_category"]).size().rename("count").reset_index()
                confusion_chart = alt.Chart(confusion).mark_rect().encode(
                    x=alt.X("predicted_category:N", sort=[x[2] for x in AQI_BANDS], title="Predicted"),
                    y=alt.Y("observed_category:N", sort=[x[2] for x in AQI_BANDS], title="Observed"),
                    color=alt.Color("count:Q", scale=alt.Scale(scheme="greens")), tooltip=["observed_category:N", "predicted_category:N", "count:Q"])
                labels = alt.Chart(confusion).mark_text().encode(x=alt.X("predicted_category:N", sort=[x[2] for x in AQI_BANDS]), y=alt.Y("observed_category:N", sort=[x[2] for x in AQI_BANDS]), text="count:Q")
                st.altair_chart((confusion_chart + labels).properties(height=300), use_container_width=True)
    st.info("These are held-out validation predictions from the saved workflow, not a live or future AQI forecast.")

with quality_tab:
    section("Data transparency", "Coverage by variable")
    candidates = ["aqi", *POLLUTANTS.keys(), "temperature", "rh", "ws", "rain", "pressure"]
    coverage_rows = [{"variable": POLLUTANTS.get(c, c.replace("_", " ").title()), "coverage": filtered[c].notna().mean() * 100} for c in candidates if c in filtered]
    coverage_df = pd.DataFrame(coverage_rows)
    quality_chart = alt.Chart(coverage_df).mark_bar(cornerRadiusEnd=7).encode(
        y=alt.Y("variable:N", sort="-x", title=None), x=alt.X("coverage:Q", title="Coverage (%)", scale=alt.Scale(domain=[0,100])),
        color=alt.Color("coverage:Q", scale=alt.Scale(domain=[0,100], range=["#d8e4df", "#15a776"]), legend=None),
        tooltip=["variable:N", alt.Tooltip("coverage:Q", format=".1f")]
    ).properties(height=400)
    st.altair_chart(quality_chart, use_container_width=True)
    st.markdown(f'<div class="data-note"><b>Selected timeline:</b> {len(filtered):,} hourly rows. Missing values remain missing and are not presented as observations. Coverage varies by station and variable.</div>', unsafe_allow_html=True)
    export_cols = [c for c in ["timestamp", "aqi", "aqi_category", *POLLUTANTS.keys(), "temperature", "rh", "ws", "wd", "rain"] if c in filtered]
    st.download_button("Download selected data as CSV", filtered[export_cols].to_csv(index=False).encode("utf-8"), f"{city}_{station}_selected_data.csv".replace(" ", "_"), "text/csv")

with stats_tab:
    section("Descriptive evidence", "Manuscript-style statistical atlas")
    numeric_candidates = [c for c in ["aqi", *POLLUTANTS.keys(), "temperature", "rh", "ws", "rain", "pressure"] if c in filtered]
    stats = descriptive_statistics(filtered, numeric_candidates)
    s1, s2, s3, s4 = st.columns(4)
    with s1: metric_card("Σ", "Variables summarized", f"{len(stats)}", "AQI, pollutants and weather")
    with s2: metric_card("N", "Timeline rows", f"{len(filtered):,}", "Selected period")
    with s3: metric_card("∅", "Total missing cells", f'{int(filtered[numeric_candidates].isna().sum().sum()):,}', "Selected variables")
    with s4: metric_card("≈", "Median AQI", f'{filtered["aqi"].median():.1f}' if filtered["aqi"].notna().any() else "—", "Robust centre")
    st.dataframe(stats, use_container_width=True, hide_index=True, column_config={c: st.column_config.NumberColumn(format="%.2f") for c in stats.columns if c not in ["Variable", "N", "Missing", "Unique"]})
    st.download_button("Download descriptive statistics", stats.to_csv(index=False).encode("utf-8"), f"{city}_{station}_descriptive_statistics.csv".replace(" ", "_"), "text/csv")

    section("Distribution summary", "Box plots across measured variables")
    normalized = filtered[numeric_candidates].copy()
    normalized = (normalized - normalized.median()) / normalized.apply(lambda s: s.quantile(.75) - s.quantile(.25)).replace(0, np.nan)
    normalized = downsample(normalized, 10000).melt(var_name="variable", value_name="robust_scaled").dropna()
    normalized["variable"] = normalized["variable"].map(lambda x: "AQI" if x == "aqi" else POLLUTANTS.get(x, x.replace("_", " ").title()))
    st.altair_chart(alt.Chart(normalized).mark_boxplot(extent=1.5).encode(x=alt.X("variable:N", title=None), y=alt.Y("robust_scaled:Q", title="Robust-scaled value"), color=alt.Color("variable:N", legend=None)).properties(height=390), use_container_width=True)
    source_note("Statistical atlas. Descriptive summaries and distribution comparison", ["Wright2020", "Evolution2023"])

with patterns_tab:
    section("Temporal signatures", "Dense pattern atlas")
    variable = st.selectbox("Pattern variable", [c for c in ["aqi", *POLLUTANTS.keys()] if c in filtered and filtered[c].notna().any()], format_func=lambda x: "AQI" if x == "aqi" else POLLUTANTS[x], key="pattern_variable")
    qpattern = filtered[["timestamp", variable]].dropna().copy()
    qpattern["hour"] = qpattern["timestamp"].dt.hour
    qpattern["weekday"] = qpattern["timestamp"].dt.day_name().str[:3]
    qpattern["month"] = qpattern["timestamp"].dt.month
    qpattern["year"] = qpattern["timestamp"].dt.year
    qpattern["season"] = qpattern["timestamp"].dt.month.map({12:"Winter",1:"Winter",2:"Winter",3:"Pre-monsoon",4:"Pre-monsoon",5:"Pre-monsoon",6:"Monsoon",7:"Monsoon",8:"Monsoon",9:"Monsoon",10:"Post-monsoon",11:"Post-monsoon"})
    p1, p2 = st.columns(2)
    with p1:
        section("Pattern 1", "Hour × weekday heatmap")
        hw = qpattern.groupby(["weekday", "hour"], as_index=False)[variable].mean()
        st.altair_chart(alt.Chart(hw).mark_rect().encode(x=alt.X("hour:O", title="Hour"), y=alt.Y("weekday:N", sort=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], title=None), color=alt.Color(f"{variable}:Q", scale=alt.Scale(scheme="yelloworangered")), tooltip=["weekday:N", "hour:O", alt.Tooltip(f"{variable}:Q", format=".1f")]).properties(height=330), use_container_width=True)
    with p2:
        section("Pattern 2", "Month × year heatmap")
        my = qpattern.groupby(["year", "month"], as_index=False)[variable].mean()
        st.altair_chart(alt.Chart(my).mark_rect().encode(x=alt.X("month:O", title="Month"), y=alt.Y("year:O", title="Year"), color=alt.Color(f"{variable}:Q", scale=alt.Scale(scheme="yelloworangered")), tooltip=["year:O", "month:O", alt.Tooltip(f"{variable}:Q", format=".1f")]).properties(height=330), use_container_width=True)
    p3, p4 = st.columns(2)
    with p3:
        section("Pattern 3", "Seasonal violin-density view")
        density = alt.Chart(downsample(qpattern, 10000)).transform_density(variable, as_=[variable, "density"], groupby=["season"]).mark_area(orient="horizontal", opacity=.65).encode(y=alt.Y(f"{variable}:Q", title="AQI" if variable == "aqi" else POLLUTANTS[variable]), x=alt.X("density:Q", stack="center", title=None), color=alt.Color("season:N", legend=None), column=alt.Column("season:N", sort=["Winter","Pre-monsoon","Monsoon","Post-monsoon"], title=None)).properties(width=120, height=300)
        st.altair_chart(density, use_container_width=True)
    with p4:
        section("Pattern 4", "Exceedance-frequency curve")
        exceed = qpattern[[variable]].sort_values(variable, ascending=False).reset_index(drop=True)
        exceed["Percent of observations"] = 100 * (exceed.index + 1) / len(exceed)
        st.altair_chart(alt.Chart(downsample(exceed, 5000)).mark_line(color="#b74747", strokeWidth=2).encode(x=alt.X("Percent of observations:Q"), y=alt.Y(f"{variable}:Q"), tooltip=[alt.Tooltip("Percent of observations:Q", format=".1f"), alt.Tooltip(f"{variable}:Q", format=".1f")]).properties(height=330), use_container_width=True)
    section("Pattern 5", "Empirical cumulative distribution")
    ecdf = qpattern[[variable]].sort_values(variable).reset_index(drop=True)
    ecdf["Cumulative percent"] = 100 * (ecdf.index + 1) / len(ecdf)
    st.altair_chart(alt.Chart(downsample(ecdf, 6000)).mark_line(color="#176f54", strokeWidth=2.5).encode(x=alt.X(f"{variable}:Q"), y=alt.Y("Cumulative percent:Q"), tooltip=[alt.Tooltip(f"{variable}:Q", format=".1f"), alt.Tooltip("Cumulative percent:Q", format=".1f")]).properties(height=360), use_container_width=True)
    source_note("Temporal pattern atlas", ["Li2016", "Liu2016", "HD2023"])

with advanced_tab:
    section("Multivariate analysis", "Principal component analysis")
    pca_columns = [c for c in ["aqi", *POLLUTANTS.keys(), "temperature", "rh", "ws", "pressure"] if c in filtered and filtered[c].notna().sum() >= 20]
    scores, loadings, ratios = pca_scores(filtered, pca_columns)
    if len(scores):
        a1, a2, a3 = st.columns(3)
        with a1: metric_card("PC1", "Variance explained", f"{ratios[0]*100:.1f}%", "First component")
        with a2: metric_card("PC2", "Variance explained", f"{ratios[1]*100:.1f}%", "Second component")
        with a3: metric_card("N", "Complete PCA rows", f"{len(scores):,}", f"{len(loadings)} variables")
        left_pca, right_pca = st.columns([1.35, 1])
        with left_pca:
            st.altair_chart(alt.Chart(scores).mark_circle(opacity=.35, size=35).encode(x=alt.X("PC1:Q", title=f"PC1 ({ratios[0]*100:.1f}%)"), y=alt.Y("PC2:Q", title=f"PC2 ({ratios[1]*100:.1f}%)"), color=alt.Color("season:N"), tooltip=["timestamp:T", "season:N", alt.Tooltip("PC1:Q", format=".2f"), alt.Tooltip("PC2:Q", format=".2f")]).properties(height=390).interactive(), use_container_width=True)
        with right_pca:
            load_long = loadings.melt("Variable", var_name="Component", value_name="Loading")
            st.altair_chart(alt.Chart(load_long).mark_bar().encode(y=alt.Y("Variable:N", title=None), x=alt.X("Loading:Q"), color=alt.Color("Component:N"), row=alt.Row("Component:N", title=None), tooltip=["Variable:N", "Component:N", alt.Tooltip("Loading:Q", format=".3f")]).properties(height=160), use_container_width=True)
    else:
        st.info("PCA requires at least two sufficiently complete numeric variables.")

    section("Event intelligence", "Pollution episodes and anomalies")
    episodes = episode_table(filtered)
    e1, e2, e3 = st.columns(3)
    with e1: metric_card("#", "Poor+ episodes", f"{len(episodes)}", "Daily mean AQI > 200")
    with e2: metric_card("↔", "Longest episode", f'{int(episodes["Duration_days"].max())} days' if len(episodes) else "0 days", "Consecutive days")
    with e3: metric_card("↑", "Highest episode peak", f'{episodes["Peak_AQI"].max():.0f}' if len(episodes) else "—", "Daily mean AQI")
    if len(episodes):
        st.dataframe(episodes.head(25), use_container_width=True, hide_index=True)
    anomaly = filtered[["timestamp", "aqi"]].dropna().copy()
    median = anomaly["aqi"].median()
    mad = (anomaly["aqi"] - median).abs().median()
    anomaly["robust_z"] = .6745 * (anomaly["aqi"] - median) / mad if mad else 0
    anomaly["Anomaly"] = anomaly["robust_z"].abs() > 3.5
    base_anomaly = alt.Chart(downsample(anomaly, 8000)).encode(x=alt.X("timestamp:T", title=None), y=alt.Y("aqi:Q", title="AQI"))
    st.altair_chart((base_anomaly.mark_line(color="#96afa5") + base_anomaly.transform_filter(alt.datum.Anomaly).mark_circle(color="#c94343", size=55).encode(tooltip=["timestamp:T", alt.Tooltip("aqi:Q", format=".1f"), alt.Tooltip("robust_z:Q", format=".2f")])).properties(height=360), use_container_width=True)
    source_note("Advanced analytics. PCA, pollution episodes and robust anomaly screening", ["Liu2021", "HD2023"])

st.markdown("---")
st.caption("PIIANN AQI Dashboard · Prepared for the MCA project submission of Rama Siva Kiran Reddy (A24CA0239), Centre for Distance and Online Education, Andhra University · Historical CPCB workbench results · Health guidance is informational, not medical advice.")
