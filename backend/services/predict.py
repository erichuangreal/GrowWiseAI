"""
Predict service: load tree_health_rf_model.pkl (RandomForestClassifier), predict, return survivability.
"""
from pathlib import Path

import joblib
import numpy as np

# Load model once at module import (model is raw RandomForestClassifier, saved from RandomForestModel.py)
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_MODEL_PATH = _BACKEND_DIR / "tree_health_rf_model.pkl"

_model = None

# Class labels from RandomForestModel.py
CLASS_LABELS = {
    0: "unhealthy",
    1: "subhealthy",
    2: "healthy",
    3: "very_healthy",
}


def _load_model():
    global _model
    if _model is None:
        _model = joblib.load(_MODEL_PATH)
    return _model


def get_feature_names():
    model = _load_model()
    return list(getattr(model, "feature_names_in_", []))


def get_label_encoder():
    return None  # Model uses numeric classes; we map via CLASS_LABELS


def predict(features_dict: dict) -> dict:
    """
    Run prediction on a features dict.
    Model expects 7 features (lowercase): elevation, temperature, humidity, soil_TN, soil_TP, soil_AP, soil_AN.
    Accepts PascalCase/snake_case and normalizes to model names.
    Returns: status, label, survivability, confidence, key_factors, explanation.
    """
    model = _load_model()
    feature_names = list(getattr(model, "feature_names_in_", []))

    # Map common incoming keys to model's lowercase names
    key_map = {
        "Elevation": "elevation",
        "elevation": "elevation",
        "Temperature": "temperature",
        "temperature": "temperature",
        "Humidity": "humidity",
        "humidity": "humidity",
        "Soil_TN": "soil_TN",
        "soil_tn": "soil_TN",
        "Soil_TP": "soil_TP",
        "soil_tp": "soil_TP",
        "Soil_AP": "soil_AP",
        "soil_ap": "soil_AP",
        "Soil_AN": "soil_AN",
        "soil_an": "soil_AN",
    }

    # Medians from training data (us_tree_health_realistic.csv style) for missing values
    medians = {
        "elevation": 1503.57,
        "temperature": 21.75,
        "humidity": 59.61,
        "soil_TN": 0.511,
        "soil_TP": 0.250,
        "soil_AP": 0.247,
        "soil_AN": 0.244,
    }

    row = {}
    for name in feature_names:
        value = None
        for k, v in key_map.items():
            if v == name and k in features_dict and features_dict[k] is not None:
                value = features_dict[k]
                break
        if value is not None:
            try:
                row[name] = float(value)
            except (TypeError, ValueError):
                row[name] = medians.get(name, 0.0)
        else:
            row[name] = medians.get(name, 0.0)

    X = np.array([[row[n] for n in feature_names]], dtype=np.float64)
    pred_numeric = int(model.predict(X)[0])
    pred_proba = model.predict_proba(X)[0]

    pred_class = CLASS_LABELS.get(pred_numeric, f"class_{pred_numeric}")
    proba_dict = {
        CLASS_LABELS.get(i, f"class_{i}"): float(pred_proba[i])
        for i in range(len(pred_proba))
    }

    # Survivability: probability of healthy or very_healthy
    survivability = float(
        proba_dict.get("healthy", 0) + proba_dict.get("very_healthy", 0)
    )
    confidence = float(np.max(pred_proba))

    status = "healthy" if pred_class in ("healthy", "very_healthy") else "unhealthy"

    return {
        "status": status,
        "label": pred_class,
        "survivability": round(survivability, 4),
        "confidence": round(confidence, 4),
        "key_factors": [],
        "explanation": f"Model predicts {pred_class} (survivability {survivability:.0%}).",
        "probabilities": proba_dict,
    }
