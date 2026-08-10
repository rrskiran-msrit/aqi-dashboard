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


def render_city_overview(city_name: str, city_rows: pd.DataFrame) -> None:
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
st.sidebar.markdown("## 🌿 AirScope")
st.sidebar.caption("Community air-quality intelligence")
preferred_city_order = ["Hyderabad", "Bengaluru", "Jaipur"]
available_cities = list(index["city"].dropna().unique())
ordered_cities = [c for c in preferred_city_order if c in available_cities] + sorted(c for c in available_cities if c not in preferred_city_order)
city = st.sidebar.selectbox("CITY", ordered_cities)
city_index = index[index["city"] == city]
analysis_level = st.sidebar.radio("VIEW", ["City overview", "Station explorer"], horizontal=True)
if analysis_level == "City overview":
    render_city_overview(city, city_index)
    st.markdown("---")
    st.caption("AirScope · City comparisons use a common period, shared snapshot time and explicit data-coverage safeguards.")
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

overview_tab, trends_tab, pollutants_tab, models_tab, quality_tab = st.tabs(
    ["Overview", "AQI trends", "Pollutants", "Model lab", "Data quality"]
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
    if len(preds):
        model_names = sorted(preds["model"].dropna().unique())
        model = st.selectbox("Observed versus predicted model", model_names)
        q = preds[preds["model"] == model].copy()
        scatter = alt.Chart(downsample(q, 5000)).mark_circle(opacity=.42, color="#16815d").encode(
            x=alt.X("observed:Q", title="Observed AQI"), y=alt.Y("predicted:Q", title="Predicted AQI"),
            tooltip=[alt.Tooltip("observed:Q", format=".1f"), alt.Tooltip("predicted:Q", format=".1f")]
        ).properties(height=380).interactive()
        diagonal = alt.Chart(pd.DataFrame({"x": [q.observed.min(), q.observed.max()]})).mark_line(color="#e28b38", strokeDash=[5,5]).encode(x="x:Q", y="x:Q")
        st.altair_chart(scatter + diagonal, use_container_width=True)
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

st.markdown("---")
st.caption("AirScope · Data-only community dashboard · Historical CPCB workbench results · Health guidance is informational, not medical advice.")
