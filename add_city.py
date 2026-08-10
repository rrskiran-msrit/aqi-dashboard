r"""Import a compact city result bundle into the dashboard.

Usage:
    python add_city.py "C:\path\to\city results"
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
KEEP_FILES = (
    "station_hourly.parquet",
    "station_config.json",
    "model_metrics.csv",
    "aqi_predictions.csv",
    "pollutant_predictions.csv",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Import a modeled city into AirScope")
    parser.add_argument("result_folder", type=Path, help="Folder containing streamlit_bundle/")
    args = parser.parse_args()
    source_bundle = args.result_folder.resolve() / "streamlit_bundle"
    source_index = source_bundle / "station_index.csv"
    if not source_index.exists():
        raise SystemExit(f"Could not find {source_index}")

    incoming = pd.read_csv(source_index)
    required = {"city", "station", "station_folder"}
    if not required.issubset(incoming.columns):
        raise SystemExit(f"station_index.csv is missing: {sorted(required - set(incoming.columns))}")

    for row in incoming.itertuples(index=False):
        relative = Path(*str(row.station_folder).replace("\\", "/").split("/"))
        source_station = source_bundle / relative
        target_station = DATA_DIR / relative
        hourly = source_station / "station_hourly.parquet"
        config = source_station / "station_config.json"
        if not hourly.exists() or not config.exists():
            raise SystemExit(f"Required station files are missing in {source_station}")
        target_station.mkdir(parents=True, exist_ok=True)
        for filename in KEEP_FILES:
            source = source_station / filename
            if source.exists():
                shutil.copy2(source, target_station / filename)

    dashboard_index = DATA_DIR / "station_index.csv"
    existing = pd.read_csv(dashboard_index) if dashboard_index.exists() else incoming.iloc[0:0]
    combined = pd.concat([existing, incoming], ignore_index=True, sort=False)
    combined = combined.drop_duplicates(subset=["city", "station"], keep="last").sort_values(["city", "station"])
    combined.to_csv(dashboard_index, index=False)
    cities = ", ".join(sorted(incoming["city"].astype(str).unique()))
    print(f"Imported {len(incoming)} station(s) for {cities}.")
    print(f"Updated {dashboard_index}")


if __name__ == "__main__":
    main()
