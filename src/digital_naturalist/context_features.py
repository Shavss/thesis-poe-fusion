"""
Context feature engineering helpers for the environmental (XGBoost) expert.

This module supports TWO pipelines:

1) status_quo  -> replicates the original notebook's `engineer_features(...)` feature set
2) experimental -> the newer modular pipeline

Usage (in notebooks):
    from digital_naturalist.context_features import engineer_context_features, prepare_X

    df = engineer_context_features(df, feature_set="status_quo")  # or "experimental"
    feature_names = ...  # ideally load a fixed CSV for reproducibility
    X = prepare_X(df, feature_names)

Notes
- Pure functions: operate on pandas DataFrames, no disk I/O.
- Deterministic transforms (given the input DataFrame).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "AmsterdamConfig",
    "engineer_context_features",
    # status quo
    "engineer_features_status_quo",
    "build_feature_list_status_quo",
    # experimental
    "engineer_features_experimental",
    "build_feature_list_experimental",
    # utilities
    "prepare_X",
    "top_k_accuracy",
]


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class AmsterdamConfig:
    tz: str = "Europe/Amsterdam"
    city_lat: float = 52.3728
    city_lon: float = 4.8936
    grid_decimals: int = 2  # 0.01 deg ~ 1.1km


# -----------------------------
# Small utilities
# -----------------------------
def _require_cols(df: pd.DataFrame, cols: Sequence[str], where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{where}: missing required columns: {missing}")


def _get_lat_lon(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    if "final_latitude" in df.columns:
        lat = pd.to_numeric(df["final_latitude"], errors="coerce")
    elif "latitude" in df.columns:
        lat = pd.to_numeric(df["latitude"], errors="coerce")
    else:
        raise ValueError("No latitude column found (expected final_latitude or latitude).")

    if "final_longitude" in df.columns:
        lon = pd.to_numeric(df["final_longitude"], errors="coerce")
    elif "longitude" in df.columns:
        lon = pd.to_numeric(df["longitude"], errors="coerce")
    else:
        raise ValueError("No longitude column found (expected final_longitude or longitude).")

    return lat.astype(float), lon.astype(float)


# -----------------------------
# STATUS QUO PIPELINE (matches original notebook engineer_features)
# -----------------------------
def engineer_features_status_quo(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Replicates the original notebook `engineer_features(...)` feature set.

    Assumptions:
    - parquet already contains weather + landcover columns
      e.g., temp_c, rhum, wspd_ms, prcp_mm, cloud_cover, swrad, vpd_kpa, pres_hpa (optional),
      and worldcover fractions wc{radius}_<class> for radius in {10,50,100,250}.
    """
    out = df.copy()

    # time base
    if "obs_dt_utc" in out.columns:
        dt_utc = pd.to_datetime(out["obs_dt_utc"], utc=True, errors="coerce")
    elif "observed_at" in out.columns:
        dt_utc = pd.to_datetime(out["observed_at"], utc=True, errors="coerce")
    else:
        dt_utc = pd.to_datetime(pd.NaT)

    # local hour / month / doy
    if "hour_local" not in out.columns:
        try:
            dt_local = dt_utc.dt.tz_convert("Europe/Amsterdam")
            out["hour_local"] = dt_local.dt.hour
        except Exception:
            out["hour_local"] = np.nan

    if "obs_month" not in out.columns:
        try:
            dt_local = dt_utc.dt.tz_convert("Europe/Amsterdam")
            out["obs_month"] = dt_local.dt.month
        except Exception:
            out["obs_month"] = np.nan

    if "doy" not in out.columns:
        try:
            dt_local = dt_utc.dt.tz_convert("Europe/Amsterdam")
            out["doy"] = dt_local.dt.dayofyear
        except Exception:
            out["doy"] = np.nan

    # TEMPORAL FEATURES
    if verbose:
        print("  ✓ Temporal features...")
    out["hour_sin"] = np.sin(2 * np.pi * out["hour_local"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour_local"] / 24)

    week_of_year = dt_utc.dt.isocalendar().week if hasattr(dt_utc, "dt") else pd.Series([np.nan] * len(out))
    out["week_of_year"] = week_of_year.astype("Int64")
    out["week_sin"] = np.sin(2 * np.pi * out["week_of_year"].astype(float) / 52)
    out["week_cos"] = np.cos(2 * np.pi * out["week_of_year"].astype(float) / 52)

    # Approx. day-length model (as in your notebook)
    day_length = 12 + 6 * np.sin(2 * np.pi * (out["doy"].astype(float) - 80) / 365)
    sunrise_hour = 12 - day_length / 2
    sunset_hour = 12 + day_length / 2
    out["hours_since_sunrise"] = out["hour_local"] - sunrise_hour
    out["hours_until_sunset"] = sunset_hour - out["hour_local"]
    out["is_golden_hour"] = ((out["hours_since_sunrise"] < 2) | (out["hours_until_sunset"] < 2)).astype(int)

    out["is_spring"] = out["obs_month"].isin([3, 4, 5]).astype(int)
    out["is_summer"] = out["obs_month"].isin([6, 7, 8]).astype(int)
    out["is_fall"] = out["obs_month"].isin([9, 10]).astype(int)

    # WEATHER FEATURES
    if verbose:
        print("  ✓ Weather features...")
    if "temp_c" in out.columns:
        out["is_optimal_temp"] = ((out["temp_c"] >= 15) & (out["temp_c"] <= 28)).astype(int)
        out["temp_squared"] = out["temp_c"] ** 2
    else:
        out["is_optimal_temp"] = 0
        out["temp_squared"] = 0.0

    if "rhum" in out.columns:
        out["is_humid"] = (out["rhum"] > 70).astype(int)
        out["is_dry"] = (out["rhum"] < 40).astype(int)
    else:
        out["is_humid"] = 0
        out["is_dry"] = 0

    if "wspd_ms" in out.columns:
        out["is_calm"] = (out["wspd_ms"] < 3).astype(int)
        out["is_windy"] = (out["wspd_ms"] > 7).astype(int)
    else:
        out["is_calm"] = 0
        out["is_windy"] = 0

    if "prcp_mm" in out.columns:
        out["has_rain"] = (out["prcp_mm"] > 0.5).astype(int)
    else:
        out["has_rain"] = 0

    if "cloud_cover" in out.columns:
        out["is_sunny"] = (out["cloud_cover"] < 30).astype(int)
        out["is_overcast"] = (out["cloud_cover"] > 70).astype(int)
    else:
        out["is_sunny"] = 0
        out["is_overcast"] = 0

    if "swrad" in out.columns:
        out["swrad_per_hour"] = out["swrad"] / np.maximum(day_length, 1)
    else:
        out["swrad_per_hour"] = 0.0

    # HABITAT COMPOSITION
    if verbose:
        print("  ✓ Habitat composition...")
    for radius in [10, 50, 100, 250]:
        out[f"vegetation_total_{radius}"] = (
            out.get(f"wc{radius}_tree", 0.0) + out.get(f"wc{radius}_shrub", 0.0) + out.get(f"wc{radius}_grass", 0.0)
        )
        out[f"natural_total_{radius}"] = (
            out.get(f"wc{radius}_tree", 0.0) + out.get(f"wc{radius}_shrub", 0.0)
            + out.get(f"wc{radius}_grass", 0.0) + out.get(f"wc{radius}_herb_wetland", 0.0)
        )
        out[f"impervious_{radius}"] = out.get(f"wc{radius}_builtup", 0.0) + out.get(f"wc{radius}_bare", 0.0)

    # HABITAT DIVERSITY
    if verbose:
        print("  ✓ Habitat diversity...")
    for radius in [10, 50, 100, 250]:
        habitat_cols = [
            f"wc{radius}_tree", f"wc{radius}_shrub", f"wc{radius}_grass",
            f"wc{radius}_cropland", f"wc{radius}_builtup", f"wc{radius}_water",
        ]
        M = out.reindex(columns=habitat_cols, fill_value=0.0).to_numpy(dtype=float) + 1e-6
        M = M / M.sum(axis=1, keepdims=True)
        shannon = -np.sum(M * np.log(M), axis=1)

        out[f"habitat_diversity_{radius}"] = shannon
        out[f"habitat_richness_{radius}"] = (out.reindex(columns=habitat_cols, fill_value=0.0) > 0.05).sum(axis=1)
        out[f"habitat_dominance_{radius}"] = out.reindex(columns=habitat_cols, fill_value=0.0).max(axis=1)

    # CROSS-SCALE GRADIENTS
    if verbose:
        print("  ✓ Cross-scale gradients...")
    out["vegetation_gradient_10_50"] = out["vegetation_total_10"] - out["vegetation_total_50"]
    out["vegetation_gradient_50_250"] = out["vegetation_total_50"] - out["vegetation_total_250"]
    out["urban_gradient_10_50"] = out.get("wc10_builtup", 0.0) - out.get("wc50_builtup", 0.0)
    out["urban_gradient_50_250"] = out.get("wc50_builtup", 0.0) - out.get("wc250_builtup", 0.0)
    out["water_gradient_10_100"] = out.get("wc10_water", 0.0) - out.get("wc100_water", 0.0)
    out["tree_gradient_10_100"] = out.get("wc10_tree", 0.0) - out.get("wc100_tree", 0.0)

    # WEATHER × HABITAT INTERACTIONS
    if verbose:
        print("  ✓ Weather-habitat interactions...")
    out["temp_x_vegetation_50"] = out.get("temp_c", 0.0) * out["vegetation_total_50"]
    out["temp_x_builtup_50"] = out.get("temp_c", 0.0) * out.get("wc50_builtup", 0.0)
    out["temp_x_water_50"] = out.get("temp_c", 0.0) * out.get("wc50_water", 0.0)
    out["humidity_x_vegetation_50"] = out.get("rhum", 0.0) * out["vegetation_total_50"]
    out["humidity_x_wetland_50"] = out.get("rhum", 0.0) * out.get("wc50_herb_wetland", 0.0)
    out["wind_x_vegetation_100"] = out.get("wspd_ms", 0.0) * out["vegetation_total_100"]
    out["wind_x_tree_shelter"] = out.get("wspd_ms", 0.0) * out.get("wc100_tree", 0.0)
    out["solar_x_vegetation"] = out.get("swrad", 0.0) * out["vegetation_total_50"]
    out["solar_x_builtup"] = out.get("swrad", 0.0) * out.get("wc50_builtup", 0.0)
    out["vpd_x_vegetation"] = out.get("vpd_kpa", 0.0) * out["vegetation_total_50"]
    out["vpd_x_water_proximity"] = out.get("vpd_kpa", 0.0) * (1 - out.get("wc50_water", 0.0))

    # URBAN CONTEXT
    if verbose:
        print("  ✓ Urban context...")
    out["urban_heat_index"] = (
        out.get("wc50_builtup", 0.0) * 2
        + out.get("wc250_builtup", 0.0)
        - 0.5 * out["vegetation_total_50"]
    )
    out["floral_resources"] = (
        out.get("wc10_grass", 0.0) * 0.5
        + out.get("wc50_grass", 0.0) * 1.0
        + out.get("wc50_shrub", 0.0) * 1.5
        + out.get("wc50_cropland", 0.0) * 0.8
    )
    out["cavity_nesting_habitat"] = out.get("wc50_tree", 0.0) + out.get("wc50_builtup", 0.0) * 0.2
    out["ground_nesting_habitat"] = out.get("wc10_grass", 0.0) + out.get("wc10_bare", 0.0) * 0.5
    out["habitat_edges_50"] = out["habitat_richness_50"] * out["habitat_diversity_50"]

    # TEMPORAL × HABITAT INTERACTIONS
    if verbose:
        print("  ✓ Temporal-habitat interactions...")
    out["spring_x_vegetation"] = out["is_spring"] * out["vegetation_total_50"]
    out["summer_x_water"] = out["is_summer"] * out.get("wc50_water", 0.0)
    out["morning_x_flowers"] = (out["hour_local"] < 12).astype(int) * out["floral_resources"]
    out["afternoon_x_flowers"] = (out["hour_local"] >= 12).astype(int) * out["floral_resources"]

    return out


def build_feature_list_status_quo(df: pd.DataFrame) -> List[str]:
    """
    Conservative feature list builder for the status quo pipeline.
    In practice, prefer loading your fixed feature list CSV for reproducibility.
    """
    candidates = [
        "hour_local", "obs_month", "doy", "week_of_year",
        "hour_sin", "hour_cos", "week_sin", "week_cos",
        "hours_since_sunrise", "hours_until_sunset", "is_golden_hour",
        "is_spring", "is_summer", "is_fall",
        "temp_c", "rhum", "wspd_ms", "prcp_mm", "cloud_cover", "swrad", "vpd_kpa", "pres_hpa",
        "is_optimal_temp", "temp_squared", "is_humid", "is_dry", "is_calm", "is_windy", "has_rain",
        "is_sunny", "is_overcast", "swrad_per_hour",
        "vegetation_total_10", "vegetation_total_50", "vegetation_total_100", "vegetation_total_250",
        "natural_total_10", "natural_total_50", "natural_total_100", "natural_total_250",
        "impervious_10", "impervious_50", "impervious_100", "impervious_250",
        "habitat_diversity_10", "habitat_diversity_50", "habitat_diversity_100", "habitat_diversity_250",
        "habitat_richness_10", "habitat_richness_50", "habitat_richness_100", "habitat_richness_250",
        "habitat_dominance_10", "habitat_dominance_50", "habitat_dominance_100", "habitat_dominance_250",
        "vegetation_gradient_10_50", "vegetation_gradient_50_250",
        "urban_gradient_10_50", "urban_gradient_50_250",
        "water_gradient_10_100", "tree_gradient_10_100",
        "temp_x_vegetation_50", "temp_x_builtup_50", "temp_x_water_50",
        "humidity_x_vegetation_50", "humidity_x_wetland_50",
        "wind_x_vegetation_100", "wind_x_tree_shelter",
        "solar_x_vegetation", "solar_x_builtup",
        "vpd_x_vegetation", "vpd_x_water_proximity",
        "urban_heat_index", "floral_resources", "cavity_nesting_habitat", "ground_nesting_habitat", "habitat_edges_50",
        "spring_x_vegetation", "summer_x_water", "morning_x_flowers", "afternoon_x_flowers",
    ]
    wc_cols = [c for c in df.columns if c.startswith("wc") and "_" in c]
    feats = [c for c in candidates if c in df.columns] + wc_cols
    seen = set()
    feats = [f for f in feats if not (f in seen or seen.add(f))]
    return feats


# -----------------------------
# EXPERIMENTAL PIPELINE (modular)
# -----------------------------
def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (np.sin((lat2 - lat1) / 2.0) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


def _solar_declination(doy: np.ndarray) -> np.ndarray:
    return np.radians(23.44) * np.sin(2 * np.pi * (doy - 81) / 365.0)


def _day_length_hours(lat_deg: np.ndarray, doy: np.ndarray) -> np.ndarray:
    lat = np.radians(lat_deg)
    dec = _solar_declination(doy)
    x = -np.tan(lat) * np.tan(dec)
    x = np.clip(x, -1, 1)
    h0 = np.arccos(x)
    return (24.0 / np.pi) * 2 * h0


def _sun_elev_simple(lat_deg: np.ndarray, doy: np.ndarray, hour_local: np.ndarray) -> np.ndarray:
    lat = np.radians(lat_deg)
    dec = _solar_declination(doy)
    H = np.radians(15.0 * (hour_local - 12.0))
    sin_alt = np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(H)
    sin_alt = np.clip(sin_alt, -1, 1)
    return np.degrees(np.arcsin(sin_alt))


def _add_time_features_experimental(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    _require_cols(df, ["observed_at"], "_add_time_features_experimental")
    out = df.copy()
    dt_local = pd.to_datetime(out["observed_at"], utc=True, errors="coerce").dt.tz_convert(tz)

    out["obs_month_local"] = dt_local.dt.month.astype("Int64")
    out["obs_hour_local"] = dt_local.dt.hour.astype("Int64")
    out["obs_weekday"] = dt_local.dt.weekday.astype("Int64")
    out["doy"] = dt_local.dt.dayofyear.astype("Int64")
    out["week_of_year"] = dt_local.dt.isocalendar().week.astype("Int64")

    doy_f = out["doy"].astype(float)
    hour_f = out["obs_hour_local"].astype(float)
    week_f = out["week_of_year"].astype(float)

    out["doy_sin"] = np.sin(2 * np.pi * doy_f / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy_f / 365.25)
    out["hour_sin"] = np.sin(2 * np.pi * hour_f / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour_f / 24.0)
    out["week_sin"] = np.sin(2 * np.pi * week_f / 52.0)
    out["week_cos"] = np.cos(2 * np.pi * week_f / 52.0)

    out["day_utc"] = pd.to_datetime(out["observed_at"], utc=True, errors="coerce").dt.floor("D")
    return out


def _add_spatial_solar_experimental(df: pd.DataFrame, cfg: AmsterdamConfig) -> pd.DataFrame:
    out = df.copy()
    lat, lon = _get_lat_lon(out)
    scale = 10 ** cfg.grid_decimals

    out["lat_bin"] = np.floor(lat * scale) / scale
    out["lon_bin"] = np.floor(lon * scale) / scale

    out["dist_city_km"] = _haversine_km(lat.to_numpy(), lon.to_numpy(), cfg.city_lat, cfg.city_lon)
    out["log1p_dist_city_km"] = np.log1p(out["dist_city_km"].astype(float))

    _require_cols(out, ["doy", "obs_hour_local"], "_add_spatial_solar_experimental")
    doy = out["doy"].astype(float).to_numpy()
    hr = out["obs_hour_local"].astype(float).to_numpy()

    out["day_length_hours"] = _day_length_hours(lat.to_numpy(), doy)
    out["sun_elev_simple"] = _sun_elev_simple(lat.to_numpy(), doy, hr)
    out["is_twilight"] = ((out["sun_elev_simple"] > -6) & (out["sun_elev_simple"] < 6)).astype(int)
    out["is_night"] = (out["sun_elev_simple"] <= -6).astype(int)
    out["is_day"] = (out["sun_elev_simple"] > 0).astype(int)
    return out


def _add_rolling_weather_experimental(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    _require_cols(out, ["lat_bin", "lon_bin", "day_utc"], "_add_rolling_weather_experimental")
    gkeys = ["lat_bin", "lon_bin"]
    out = out.sort_values(gkeys + ["day_utc"])

    def _roll(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("day_utc")
        if "temp_c" in g.columns:
            m7 = g["temp_c"].rolling(7, min_periods=3).mean()
            g["temp_c_anom7"] = g["temp_c"] - m7
        if "vpd_kpa" in g.columns:
            m7 = g["vpd_kpa"].rolling(7, min_periods=3).mean()
            g["vpd_kpa_anom7"] = g["vpd_kpa"] - m7
        if "prcp_mm" in g.columns:
            g["prcp_mm_sum3"] = g["prcp_mm"].rolling(3, min_periods=1).sum()
        if "wspd_ms" in g.columns:
            g["wspd_ms_mean3"] = g["wspd_ms"].rolling(3, min_periods=2).mean()
        return g

    out = out.groupby(gkeys, group_keys=False).apply(_roll)

    for c in ["temp_c_anom7", "vpd_kpa_anom7", "prcp_mm_sum3", "wspd_ms_mean3"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out


def _add_habitat_entropy_experimental(
    df: pd.DataFrame,
    radii: Sequence[int] = (50, 100, 250),
    habitat_classes: Sequence[str] = ("builtup", "water", "tree", "grass", "cropland", "shrub", "bare"),
) -> pd.DataFrame:
    out = df.copy()
    for rad in radii:
        cols = [f"wc{rad}_{h}" for h in habitat_classes if f"wc{rad}_{h}" in out.columns]
        if not cols:
            continue
        M = out[cols].clip(lower=0).astype(float).to_numpy()
        row_sum = M.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        P = M / row_sum
        out[f"habitat_entropy_{rad}"] = -np.nansum(P * np.log(P + 1e-12), axis=1)
    return out


def _add_hour_x_habitat_experimental(
    df: pd.DataFrame,
    cols: Sequence[str] = ("wc50_tree", "wc100_builtup", "wc250_water", "wc100_grass"),
) -> pd.DataFrame:
    out = df.copy()
    _require_cols(out, ["hour_sin"], "_add_hour_x_habitat_experimental")
    hour = out["hour_sin"].astype(float)
    for c in cols:
        if c in out.columns:
            out[f"hx_{c}"] = hour * out[c].astype(float)
    return out


def engineer_features_experimental(df: pd.DataFrame, cfg: AmsterdamConfig = AmsterdamConfig()) -> pd.DataFrame:
    out = _add_time_features_experimental(df, tz=cfg.tz)
    out = _add_spatial_solar_experimental(out, cfg=cfg)
    out = _add_rolling_weather_experimental(out)
    out = _add_habitat_entropy_experimental(out)
    out = _add_hour_x_habitat_experimental(out)
    return out


def build_feature_list_experimental(
    df: pd.DataFrame,
    habitat_classes: Sequence[str] = ("builtup", "water", "tree", "grass", "cropland"),
    radii: Sequence[int] = (10, 50, 100, 250),
    hx_keep: Sequence[str] = ("hx_wc50_tree", "hx_wc100_builtup"),
    include_coords: bool = False,
) -> List[str]:
    feats: List[str] = []

    if include_coords:
        feats += ["final_latitude", "final_longitude"]

    feats += [
        "lat_bin", "lon_bin", "dist_city_km", "log1p_dist_city_km",
        "temp_c", "rhum", "vpd_kpa", "cloud_cover", "swrad", "prcp_mm", "wspd_ms", "pres_hpa",
        "temp_c_anom7", "vpd_kpa_anom7", "prcp_mm_sum3", "wspd_ms_mean3",
        "obs_month_local", "obs_hour_local", "obs_weekday",
        "doy_sin", "doy_cos", "hour_sin", "hour_cos", "week_sin", "week_cos",
        "is_day", "day_length_hours", "sun_elev_simple", "is_twilight", "is_night",
        "habitat_entropy_50", "habitat_entropy_100", "habitat_entropy_250",
    ]

    for rad in radii:
        for h in habitat_classes:
            c = f"wc{rad}_{h}"
            if c in df.columns:
                feats.append(c)

    for c in hx_keep:
        if c in df.columns:
            feats.append(c)

    seen = set()
    feats = [f for f in feats if not (f in seen or seen.add(f))]
    feats = [f for f in feats if f in df.columns]
    return feats


# -----------------------------
# Public dispatch + matrix prep
# -----------------------------
def engineer_context_features(
    df: pd.DataFrame,
    feature_set: str = "status_quo",
    cfg: AmsterdamConfig = AmsterdamConfig(),
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Dispatches to the selected feature engineering pipeline.

    feature_set:
      - "status_quo": original notebook feature set
      - "experimental": newer modular set
    """
    fs = str(feature_set).lower().strip()
    if fs in {"status_quo", "status-quo", "baseline", "original"}:
        return engineer_features_status_quo(df, verbose=verbose)
    if fs in {"experimental", "exp", "new"}:
        return engineer_features_experimental(df, cfg=cfg)
    raise ValueError(f"Unknown feature_set='{feature_set}'. Use 'status_quo' or 'experimental'.")


def prepare_X(df: pd.DataFrame, feature_names: Sequence[str]) -> np.ndarray:
    """Return numeric matrix with missing features filled as 0."""
    X = df.copy()
    for f in feature_names:
        if f not in X.columns:
            X[f] = np.nan
    return X[list(feature_names)].astype(float).fillna(0.0).to_numpy()


def top_k_accuracy(y_true: np.ndarray, y_proba: np.ndarray, k: int) -> float:
    top_k = np.argsort(y_proba, axis=1)[:, -k:]
    return float(np.mean([y_true[i] in top_k[i] for i in range(len(y_true))]))
