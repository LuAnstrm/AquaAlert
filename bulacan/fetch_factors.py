"""Download rain forecasts, weather, cyclone, climate and terrain factors."""

import io
import time
from datetime import date, timedelta

import geopandas as gpd
import numpy as np
import pandas as pd
import truststore

truststore.inject_into_ssl()
import rasterio
import requests  # noqa: E402
from rasterio.features import rasterize
from rasterio.merge import merge
from rasterio.transform import from_origin

from build_hazard import read_barangays
from common import RAIN_POINTS, ROOT
from fetch_history import (ARCHIVE_URL, ERA5_LAG_DAYS, FORECAST_ARCHIVE_START,
                           HISTORICAL_FORECAST_URL, START_DATE, TIMEZONE, cached_get)

FACTOR_DIR = ROOT / "data/external/factors"
PROCESSED = ROOT / "data/processed"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
FORECAST_START = "2023-12-01"
IBTRACS_URL = ("https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/"
               "v04r01/access/csv/ibtracs.WP.list.v04r01.csv")
ONI_URL = "https://psl.noaa.gov/data/correlation/oni.data"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
DEM_URL = ("https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_{t}_00_DEM/"
           "Copernicus_DSM_COG_10_{t}_00_DEM.tif")
WORLDCOVER_URL = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
                  "ESA_WorldCover_10m_2021_v200_{t}_Map.tif")
DEM_TILES = ["N14_00_E120", "N15_00_E120", "N14_00_E121", "N15_00_E121"]
WORLDCOVER_TILES = ["N12E120", "N15E120"]
BAY_POINT = (14.60, 120.75)
LOCAL_POINT = RAIN_POINTS["calumpit"]
CALUMPIT = (14.91, 120.77)
# Manila Bay opens to the southwest, so wind from there can push water toward Bulacan.
ONSHORE_FROM_DEGREES = 225
USER_AGENT = "AquaAlert-Bulacan-research"
GRID_DEGREES = 0.0005
WATER_CLASS, BUILT_CLASS, WETLAND_CLASS, MANGROVE_CLASS = 80, 50, 90, 95


def daily_weather(url, start, end, model, include_deep_soil):
    hourly_vars = ["precipitation"]
    if include_deep_soil:
        hourly_vars.append("soil_moisture_28_to_100cm")
    local = cached_get(url, {"latitude": LOCAL_POINT[0], "longitude": LOCAL_POINT[1], "start_date": start,
                             "end_date": end, "hourly": ",".join(hourly_vars), "timezone": TIMEZONE,
                             **({"models": model} if model else {})})["hourly"]
    bay = cached_get(url, {"latitude": BAY_POINT[0], "longitude": BAY_POINT[1], "start_date": start,
                           "end_date": end, "hourly": "wind_speed_10m,wind_direction_10m,pressure_msl",
                           "timezone": TIMEZONE, **({"models": model} if model else {})})["hourly"]
    local_frame = pd.DataFrame(local).assign(time=lambda d: pd.to_datetime(d["time"])).set_index("time")
    bay_frame = pd.DataFrame(bay).assign(time=lambda d: pd.to_datetime(d["time"])).set_index("time")

    angle = np.radians(bay_frame["wind_direction_10m"] - ONSHORE_FROM_DEGREES)
    onshore = bay_frame["wind_speed_10m"] * np.cos(angle)
    out = pd.DataFrame({
        "rain_peak_hour_mm": local_frame["precipitation"].resample("D").max(),
        "rain_hours": (local_frame["precipitation"] >= 1.0).resample("D").sum(min_count=1),
        "onshore_wind_max_kmh": onshore.resample("D").max(),
        "pressure_min_hpa": bay_frame["pressure_msl"].resample("D").min(),
    })
    if include_deep_soil:
        out["soil_deep"] = local_frame["soil_moisture_28_to_100cm"].resample("D").mean()
    out.index.name = "date"
    return out


def rain_forecasts(end: str) -> pd.DataFrame:
    """Save forecasts issued 1 and 2 days before each valid day, by local day."""
    frames = []
    for name, (lat, lon) in RAIN_POINTS.items():
        print(f"  forecast {name}")
        pieces = []
        # This archive rejects very long requests, so ask for 90 days at a time.
        for piece_start in pd.date_range(FORECAST_START, end, freq="90D"):
            piece_end = min(piece_start + pd.Timedelta(days=89), pd.Timestamp(end))
            payload = cached_get(PREVIOUS_RUNS_URL, {
                "latitude": lat, "longitude": lon, "start_date": str(piece_start.date()),
                "end_date": str(piece_end.date()),
                "hourly": "precipitation_previous_day1,precipitation_previous_day2", "timezone": TIMEZONE,
            })["hourly"]
            pieces.append(pd.DataFrame(payload))
        hourly = pd.concat(pieces).assign(time=lambda d: pd.to_datetime(d["time"])).set_index("time")
        daily = hourly.resample("D").sum(min_count=20)
        frames.append(daily.rename(columns={
            "precipitation_previous_day1": f"fc1_{name}", "precipitation_previous_day2": f"fc2_{name}"}))
    out = pd.concat(frames, axis=1).sort_index()
    out.index.name = "date"
    return out


def cyclone_distance(start: str, end: str) -> pd.DataFrame:
    path = FACTOR_DIR / "ibtracs.WP.list.v04r01.csv"
    if not path.exists():
        response = requests.get(IBTRACS_URL, timeout=280)
        response.raise_for_status()
        path.write_bytes(response.content)
    tracks = pd.read_csv(path, usecols=["ISO_TIME", "LAT", "LON"], skiprows=[1], low_memory=False)
    tracks["time"] = pd.to_datetime(tracks["ISO_TIME"]) + pd.Timedelta(hours=8)
    tracks = tracks[(tracks["time"] >= start) & (tracks["time"] <= pd.Timestamp(end) + pd.Timedelta(days=1))]
    lat = np.radians(tracks["LAT"].astype(float))
    lon = np.radians(tracks["LON"].astype(float))
    lat0, lon0 = np.radians(CALUMPIT[0]), np.radians(CALUMPIT[1])
    haversine = (np.sin((lat - lat0) / 2) ** 2
                 + np.cos(lat) * np.cos(lat0) * np.sin((lon - lon0) / 2) ** 2)
    tracks["distance_km"] = 6371 * 2 * np.arcsin(np.sqrt(haversine))
    daily = tracks.groupby(tracks["time"].dt.normalize())["distance_km"].min()
    index = pd.date_range(start, end, name="date")
    print(f"  cyclone tracks end {tracks['time'].max()}")
    return pd.DataFrame({"tc_distance_km": daily.reindex(index)}).assign(
        tracks_available_until=str(tracks["time"].max().date()))


def enso_index() -> pd.DataFrame:
    response = requests.get(ONI_URL, timeout=60, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    rows = []
    for line in response.text.splitlines():
        parts = line.split()
        if len(parts) == 13 and parts[0].isdigit():
            year = int(parts[0])
            for month, value in enumerate(parts[1:], start=1):
                if float(value) > -90:
                    rows.append({"month": pd.Timestamp(year, month, 1), "oni": float(value)})
    return pd.DataFrame(rows).set_index("month")


def merged_tiles(urls, bounds, resolution, resampling="nearest"):
    sources = [rasterio.open(url) for url in urls]
    try:
        data, transform = merge(sources, bounds=bounds, res=resolution, resampling=rasterio.enums.Resampling[resampling])
    finally:
        for source in sources:
            source.close()
    return data[0], transform


def barangay_static() -> None:
    """Summarise elevation and land cover. Copernicus DEM includes buildings and trees."""
    barangays = read_barangays()
    left, bottom, right, top = barangays.total_bounds
    bounds = (left, bottom, right, top)

    dem_paths, cover_paths = [], []
    for tile in DEM_TILES:
        path = FACTOR_DIR / f"dem_{tile}.tif"
        if not path.exists():
            response = requests.get(DEM_URL.format(t=tile), timeout=280)
            response.raise_for_status()
            path.write_bytes(response.content)
        dem_paths.append(path)
    for tile in WORLDCOVER_TILES:
        path = FACTOR_DIR / f"worldcover_{tile}.tif"
        if not path.exists():
            response = requests.get(WORLDCOVER_URL.format(t=tile), timeout=280)
            response.raise_for_status()
            path.write_bytes(response.content)
        cover_paths.append(path)

    dem, transform = merged_tiles(dem_paths, bounds, GRID_DEGREES)
    cover, cover_transform = merged_tiles(cover_paths, bounds, GRID_DEGREES)
    height, width = dem.shape
    if cover.shape != dem.shape:
        raise ValueError(f"Grids differ: DEM {dem.shape}, land cover {cover.shape}")
    ids = rasterize(((geom, i + 1) for i, geom in enumerate(barangays.geometry)),
                    out_shape=(height, width), transform=transform, fill=0, dtype="int32")

    rows = []
    for index, row in enumerate(barangays.itertuples(), start=1):
        mask = ids == index
        if not mask.any():
            continue
        heights = dem[mask]
        covers = cover[mask]
        rows.append({
            "adm4_pcode": row.adm4_pcode, "adm4_name": row.adm4_name, "adm3_name": row.adm3_name,
            "elev_mean_m": round(float(heights.mean()), 2),
            "elev_p10_m": round(float(np.percentile(heights, 10)), 2),
            "share_below_2m": round(float((heights < 2).mean()), 3),
            "share_water": round(float((covers == WATER_CLASS).mean()), 3),
            "share_built": round(float((covers == BUILT_CLASS).mean()), 3),
            "share_wetland_mangrove": round(float(np.isin(covers, [WETLAND_CLASS, MANGROVE_CLASS]).mean()), 3),
        })
    static = pd.DataFrame(rows)

    # Waterways are optional because public Overpass servers can be busy.
    query = (f"[out:json][timeout:120];(way[\"waterway\"~\"river|canal|drain|stream\"]"
             f"({bottom},{left},{top},{right}););out geom;")
    try:
        response = requests.post(OVERPASS_URL, data={"data": query}, timeout=180,
                                 headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
        ways = []
        for element in response.json()["elements"]:
            points = [(p["lon"], p["lat"]) for p in element.get("geometry", [])]
            if len(points) >= 2:
                ways.append(points)
        water = gpd.GeoSeries.from_wkt(
            [("LINESTRING(" + ",".join(f"{x} {y}" for x, y in way) + ")") for way in ways],
            crs="EPSG:4326").to_crs("EPSG:3123")
        projected = barangays.set_index("adm4_pcode").to_crs("EPSG:3123")
        lines = gpd.GeoDataFrame(geometry=water)
        joined = gpd.overlay(lines, projected.reset_index()[["adm4_pcode", "geometry"]], how="intersection",
                             keep_geom_type=True)
        lengths = joined.assign(km=joined.length / 1000).groupby("adm4_pcode")["km"].sum()
        areas = projected.area / 1e6
        static["waterway_km_per_km2"] = (static["adm4_pcode"].map(lengths).fillna(0)
                                         / static["adm4_pcode"].map(areas)).round(2)
        print(f"  OSM waterways: {len(ways)} lines")
    except requests.RequestException as error:
        print(f"  OSM waterways skipped: {error}")

    static.to_csv(PROCESSED / "bulacan_barangay_static.csv", index=False)
    print(f"  Static table: {len(static)} barangays")


def main() -> None:
    FACTOR_DIR.mkdir(parents=True, exist_ok=True)
    end = str(date.today() - timedelta(days=ERA5_LAG_DAYS))

    print("ERA5 weather factors")
    daily_weather(ARCHIVE_URL, START_DATE, end, "era5", True).to_csv(PROCESSED / "bulacan_weather_factors.csv")
    print("Live-like weather factors")
    daily_weather(HISTORICAL_FORECAST_URL, FORECAST_ARCHIVE_START, end, None, False).to_csv(
        PROCESSED / "bulacan_weather_factors_live.csv")
    print("Rain forecasts issued 1-2 days ahead")
    forecasts = rain_forecasts(end)
    forecasts.to_csv(PROCESSED / "bulacan_rain_forecasts.csv")
    print(f"  coverage: {forecasts.dropna().index.min()} to {forecasts.dropna().index.max()}")
    print("Cyclones")
    cyclone_distance(START_DATE, end).to_csv(PROCESSED / "bulacan_cyclones.csv")
    print("El Nino")
    enso_index().to_csv(PROCESSED / "enso_oni.csv")
    print("Barangay elevation, land cover and waterways")
    barangay_static()


if __name__ == "__main__":
    main()
