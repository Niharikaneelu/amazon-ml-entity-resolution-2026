from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .features import build_features


def train_model(pairs: pd.DataFrame, fields: list[str], label_column: str, model_path: Path):
    features = build_features(pairs, fields)
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(features, pairs[label_column].astype(int))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "fields": fields}, model_path)
    return model
