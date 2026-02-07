"""
FastAPI app: CORS, /api/fetch-features, /api/predict.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from backend.services.fetch import fetch_features_for_point
    from backend.services.predict import predict
except ImportError:
    from services.fetch import fetch_features_for_point
    from services.predict import predict

app = FastAPI(title="GrowWiseAI API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/fetch-features")
async def get_fetch_features(lat: float, lon: float):
    """Fetch all features for a point (lat, lon). Cached by rounded coordinates."""
    try:
        result = await fetch_features_for_point(lat, lon)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class PredictRequest(BaseModel):
    features: dict


@app.post("/api/predict")
def post_predict(request: PredictRequest):
    """
    Run prediction on provided features.
    Expects keys matching model (e.g. Elevation, Temperature, ...).
    Accepts snake_case keys and normalizes to model names.
    """
    raw = request.features or {}
    # Normalize: accept both PascalCase and snake_case
    feature_map = {
        "elevation": "Elevation",
        "temperature": "Temperature",
        "humidity": "Humidity",
        "soil_tn": "Soil_TN",
        "soil_tp": "Soil_TP",
        "soil_ap": "Soil_AP",
        "soil_an": "Soil_AN",
        "fire_risk_index": "Fire_Risk_Index",
        "slope": "Slope",
        "menhinick_index": "Menhinick_Index",
        "gleason_index": "Gleason_Index",
        "disturbance_level": "Disturbance_Level",
    }
    features = dict(raw)
    for snake, pascal in feature_map.items():
        if snake in features and pascal not in features:
            features[pascal] = features[snake]
    try:
        return predict(features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}
