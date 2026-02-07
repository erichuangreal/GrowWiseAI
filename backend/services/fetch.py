"""
Auto-fetch service: Open-Meteo, Open-Elevation, SoilGrids, fire proxy, medians.
Cache by (round(lat, 2), round(lon, 2)).
"""
import asyncio
from functools import lru_cache
from pathlib import Path

import httpx

# Feature names matching model (exact casing)
FEATURE_NAMES = [
    "Slope",
    "Elevation",
    "Temperature",
    "Humidity",
    "Soil_TN",
    "Soil_TP",
    "Soil_AP",
    "Soil_AN",
    "Menhinick_Index",
    "Gleason_Index",
    "Disturbance_Level",
    "Fire_Risk_Index",
]

# Medians from forest_health_data_with_target.csv (for default/fallback)
MEDIANS = {
    "Slope": 21.808936091032585,
    "Elevation": 1503.5730226128198,
    "Temperature": 21.754533316862897,
    "Humidity": 59.614943703539744,
    "Soil_TN": 0.5113024573782637,
    "Soil_TP": 0.24975360936595653,
    "Soil_AP": 0.24747083523271096,
    "Soil_AN": 0.24380308068303527,
    "Menhinick_Index": 1.7524116930921474,
    "Gleason_Index": 2.9693736440949037,
    "Disturbance_Level": 0.5230227736391104,
    "Fire_Risk_Index": 0.5164885287315552,
}

# Keys that come from APIs (snake_case in response)
API_KEYS = [
    "elevation",
    "temperature",
    "humidity",
    "soil_tn",
    "soil_tp",
    "soil_ap",
    "soil_an",
    "fire_risk_index",
]


def _cache_key(lat: float, lon: float) -> tuple[float, float]:
    return (round(lat, 2), round(lon, 2))


# In-memory cache for fetch results keyed by (lat, lon) rounded to 0.01°
_fetch_cache: dict[tuple[float, float], dict] = {}
_cache_lock = asyncio.Lock()


async def _open_meteo(client: httpx.AsyncClient, lat: float, lon: float) -> tuple[float | None, float | None]:
    """Fetch temperature (°C) and relative humidity (%). Uses daily max temp (expected high) for better desert/daytime representation; falls back to current. Humidity from current."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m",
        "daily": "temperature_2m_max,temperature_2m_mean",
        "timezone": "auto",
        "forecast_days": 1,
    }
    try:
        r = await client.get(url, params=params, timeout=6.0)
        r.raise_for_status()
        data = r.json()
        # Prefer daily max (expected high for the day) so desert/daytime heat is represented
        daily = data.get("daily") or {}
        max_temps = daily.get("temperature_2m_max") or []
        temp = float(max_temps[0]) if max_temps else None
        if temp is None:
            cur = data.get("current") or {}
            temp = cur.get("temperature_2m")
            temp = float(temp) if temp is not None else None
        cur = data.get("current") or {}
        humidity = cur.get("relative_humidity_2m")
        humidity = float(humidity) if humidity is not None else None
        return (temp, humidity)
    except Exception:
        return (None, None)


async def _open_elevation(client: httpx.AsyncClient, lat: float, lon: float) -> float | None:
    """Fetch elevation (m) for a point."""
    url = "https://api.open-elevation.com/api/v1/lookup"
    params = {"locations": f"{lat},{lon}"}
    try:
        r = await client.get(url, params=params, timeout=6.0)
        r.raise_for_status()
        data = r.json()
        results = data.get("results") or []
        if results:
            return float(results[0].get("elevation", 0))
        return None
    except Exception:
        return None


async def _soilgrids(client: httpx.AsyncClient, lat: float, lon: float) -> dict[str, float | None]:
    """Fetch SoilGrids properties; return soil_tn, soil_tp, soil_ap, soil_an (or None)."""
    # SoilGrids v2 properties/query: lat, lon, depth for nitrogen etc.
    url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
    params = {"lat": lat, "lon": lon}
    out = {"soil_tn": None, "soil_tp": None, "soil_ap": None, "soil_an": None}
    try:
        r = await client.get(url, params=params, timeout=8.0)
        r.raise_for_status()
        data = r.json()
        # Response has properties per depth; use 0-5cm or first depth
        props = data.get("properties") or {}
        layers = props.get("layers") or []
        for layer in layers:
            name = (layer.get("name") or "").lower()
            depths = layer.get("depths") or []
            if not depths:
                continue
            # Take mean of depth bands or first band
            vals = []
            for d in depths:
                lab = d.get("label", "")
                values = d.get("values") or {}
                # values can be mean, min, max, etc.
                v = values.get("mean") or values.get("Q0.5")
                if v is not None:
                    vals.append(float(v))
            if not vals:
                continue
            mean_val = sum(vals) / len(vals)
            # Nitrogen in cg/kg -> scale to training range ~0.04-0.9 (g/kg would be /10; training is 0.1-0.6 often)
            if "nitrogen" in name or "n_total" in name:
                # cg/kg = 0.01 g/kg; training Soil_TN is ~0.2-0.6
                out["soil_tn"] = (mean_val / 100.0) * 2.0  # rough scale into range
                out["soil_an"] = out["soil_tn"] * 0.5  # proxy if no separate AN
            if "phosphorus" in name or "phh2o" in name:
                # Use as proxy for P if available
                out["soil_tp"] = min(1.0, max(0.0, mean_val / 100.0)) if mean_val else None
                out["soil_ap"] = out["soil_tp"]
        # If nitrogen not found, try common key
        if out["soil_tn"] is None and layers:
            for layer in layers:
                name = (layer.get("name") or "").lower()
                if "n_" in name or "nitrogen" in name:
                    depths = layer.get("depths") or []
                    for d in depths:
                        v = (d.get("values") or {}).get("mean") or (d.get("values") or {}).get("Q0.5")
                        if v is not None:
                            out["soil_tn"] = (float(v) / 100.0) * 2.0
                            out["soil_an"] = out["soil_tn"] * 0.5
                            break
                    break
    except Exception:
        pass
    return out


def _fire_risk_proxy(temperature: float | None, humidity: float | None) -> float:
    """Simple proxy: (1 - humidity/100) * min(1, temp/40), normalized to [0,1]."""
    if temperature is None:
        temperature = MEDIANS["Temperature"]
    if humidity is None:
        humidity = MEDIANS["Humidity"]
    h = max(0, min(100, humidity)) / 100.0
    t = max(0, min(50, temperature)) / 40.0
    raw = (1.0 - h) * min(1.0, t)
    return round(min(1.0, max(0.0, raw)), 4)


def _model_feature_dict(
    elevation: float | None,
    temperature: float | None,
    humidity: float | None,
    soil: dict[str, float | None],
    fire_risk: float,
) -> dict[str, float]:
    """Build full feature dict for the model (all 12 features)."""
    features = dict(MEDIANS)
    if elevation is not None:
        features["Elevation"] = elevation
    if temperature is not None:
        features["Temperature"] = temperature
    if humidity is not None:
        features["Humidity"] = humidity
    features["Fire_Risk_Index"] = fire_risk
    if soil.get("soil_tn") is not None:
        features["Soil_TN"] = soil["soil_tn"]
    if soil.get("soil_tp") is not None:
        features["Soil_TP"] = soil["soil_tp"]
    if soil.get("soil_ap") is not None:
        features["Soil_AP"] = soil["soil_ap"]
    if soil.get("soil_an") is not None:
        features["Soil_AN"] = soil["soil_an"]
    return features


def _source_dict(
    elevation: float | None,
    temperature: float | None,
    humidity: float | None,
    soil: dict[str, float | None],
) -> dict[str, str]:
    """Which source each feature came from: 'api' or 'default'."""
    source = {}
    for k in FEATURE_NAMES:
        key_lower = k.lower().replace("_", "")
        if k == "Elevation":
            source["elevation"] = "api" if elevation is not None else "default"
        elif k == "Temperature":
            source["temperature"] = "api" if temperature is not None else "default"
        elif k == "Humidity":
            source["humidity"] = "api" if humidity is not None else "default"
        elif k in ("Soil_TN", "Soil_TP", "Soil_AP", "Soil_AN"):
            sk = k.lower()
            source[sk] = "api" if (soil.get(sk) is not None) else "default"
        elif k == "Fire_Risk_Index":
            source["fire_risk_index"] = "proxy"
        else:
            source[k.lower()] = "default"
    return source


async def fetch_features_for_point(lat: float, lon: float) -> dict:
    """
    Fetch all features for (lat, lon). Uses cache key (round(lat,2), round(lon,2)).
    Returns dict with snake_case keys for API response + 'source' map.
    """
    key = _cache_key(lat, lon)
    async with _cache_lock:
        if key in _fetch_cache:
            return _fetch_cache[key].copy()

    async with httpx.AsyncClient() as client:
        meteo_task = _open_meteo(client, lat, lon)
        elev_task = _open_elevation(client, lat, lon)
        soil_task = _soilgrids(client, lat, lon)
        (temp, humidity), elevation, soil = await asyncio.gather(meteo_task, elev_task, soil_task)

    fire_risk = _fire_risk_proxy(temp, humidity)
    features = _model_feature_dict(elevation, temp, humidity, soil, fire_risk)
    source = _source_dict(elevation, temp, humidity, soil)

    # Response: snake_case for JSON, plus source
    response = {
        "elevation": features["Elevation"],
        "temperature": features["Temperature"],
        "humidity": features["Humidity"],
        "soil_tn": features["Soil_TN"],
        "soil_tp": features["Soil_TP"],
        "soil_ap": features["Soil_AP"],
        "soil_an": features["Soil_AN"],
        "fire_risk_index": features["Fire_Risk_Index"],
        "slope": features["Slope"],
        "menhinick_index": features["Menhinick_Index"],
        "gleason_index": features["Gleason_Index"],
        "disturbance_level": features["Disturbance_Level"],
        "source": source,
    }

    async with _cache_lock:
        _fetch_cache[key] = response.copy()

    return response
