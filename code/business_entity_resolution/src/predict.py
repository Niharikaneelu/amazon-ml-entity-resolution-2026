from pathlib import Path

import joblib
import pandas as pd

from .features import build_features


def predict_pairs(pairs: pd.DataFrame, model_path: Path, threshold: float = 0.5) -> pd.DataFrame:
    bundle = joblib.load(model_path)
    features = build_features(pairs, bundle["fields"])
    result = pairs.copy()
    result["match_probability"] = bundle["model"].predict_proba(features)[:, 1]
    result["match"] = (result["match_probability"] >= threshold).astype(int)
    return result
