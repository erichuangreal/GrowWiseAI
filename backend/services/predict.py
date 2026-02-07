"""
Predict service: load forest_health_model.pkl, scale, predict, return survivability.
"""
from pathlib import Path

import joblib
import numpy as np

# Load model once at module import
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_MODEL_PATH = _BACKEND_DIR / "forest_health_model.pkl"

_components = None


def _load_model():
    global _components
    if _components is None:
        _components = joblib.load(_MODEL_PATH)
    return _components


def get_feature_names():
    comp = _load_model()
    return list(comp["feature_names"])


def get_label_encoder():
    return _load_model()["label_encoder"]


def predict(features_dict: dict) -> dict:
    """
    Run prediction on a features dict (keys = feature names as in model).
    Fills missing features with medians from training data.
    Returns: status, label, survivability, confidence, key_factors, explanation.
    """
    comp = _load_model()
    model = comp["model"]
    scaler = comp["scaler"]
    le = comp["label_encoder"]
    feature_names = comp["feature_names"]

    # Build row with all required features; fill missing with median from training
    medians = comp.get("medians")
    if medians is None:
        # Fallback medians if not stored in pkl
        medians = {
            "Slope": 21.81,
            "Elevation": 1503.57,
            "Temperature": 21.75,
            "Humidity": 59.61,
            "Soil_TN": 0.511,
            "Soil_TP": 0.250,
            "Soil_AP": 0.247,
            "Soil_AN": 0.244,
            "Menhinick_Index": 1.75,
            "Gleason_Index": 2.97,
            "Disturbance_Level": 0.523,
            "Fire_Risk_Index": 0.516,
        }

    row = {}
    for name in feature_names:
        if name in features_dict and features_dict[name] is not None:
            try:
                row[name] = float(features_dict[name])
            except (TypeError, ValueError):
                row[name] = medians.get(name, 0.0)
        else:
            row[name] = medians.get(name, 0.0)

    # Ensure column order
    X = np.array([[row[n] for n in feature_names]], dtype=np.float64)
    X_scaled = scaler.transform(X)
    pred_numeric = model.predict(X_scaled)[0]
    pred_proba = model.predict_proba(X_scaled)[0]

    pred_class = le.inverse_transform([pred_numeric])[0]
    proba_dict = {le.classes_[i]: float(pred_proba[i]) for i in range(len(le.classes_))}

    # Survivability: max probability (or probability of "healthy" class if present)
    healthy_labels = [c for c in le.classes_ if "healthy" in c.lower() and "un" not in c.lower()]
    if healthy_labels:
        survivability = sum(proba_dict.get(c, 0) for c in healthy_labels)
    else:
        survivability = float(np.max(pred_proba))

    confidence = float(np.max(pred_proba))

    # Status for frontend: healthy | unhealthy
    status = "healthy" if "healthy" in pred_class.lower() and "un" not in pred_class.lower() else "unhealthy"

    return {
        "status": status,
        "label": pred_class,
        "survivability": round(survivability, 4),
        "confidence": round(confidence, 4),
        "key_factors": [],
        "explanation": f"Model predicts {pred_class} (survivability {survivability:.0%}).",
        "probabilities": proba_dict,
    }
