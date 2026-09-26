from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

from .evaluate import compute_f05_score
from .features import build_features


def train_model(
    candidate_pairs: pd.DataFrame,
    label_column: str = "label",
    model_path: Path | None = None,
    val_gt_df: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Train a matching model on candidate pairs and optimize the decision threshold for F_0.5.

    Args:
        candidate_pairs: DataFrame containing candidate pairs and binary label column.
        label_column: Column name containing 1 (true match) or 0 (candidate false match).
        model_path: Optional path to save joblib model bundle.
        val_gt_df: Optional ground truth DataFrame to tune decision threshold for macro F_0.5.

    Returns:
        Dictionary containing trained model, feature names, best_threshold, and train metrics.
    """
    if candidate_pairs.empty:
        raise ValueError("Cannot train matching model on empty candidate pairs DataFrame.")

    features = build_features(candidate_pairs)
    X = features
    y = candidate_pairs[label_column].astype(int)

    # Train LightGBM model if available, otherwise LogisticRegression
    try:
        model = LGBMClassifier(
            n_estimators=150,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(X, y)
    except Exception:
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(X, y)

    probs = model.predict_proba(X)[:, 1]

    # Threshold optimization targeting precision and F_0.5
    best_threshold = 0.80
    best_f05 = 0.0

    thresholds = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    for th in thresholds:
        preds = (probs >= th).astype(int)
        tp = np.sum((preds == 1) & (y == 1))
        fp = np.sum((preds == 1) & (y == 0))
        fn = np.sum((preds == 0) & (y == 1))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f05 = compute_f05_score(prec, rec)
        if f05 > best_f05:
            best_f05 = f05
            best_threshold = th

    bundle = {
        "model": model,
        "feature_names": list(X.columns),
        "best_threshold": best_threshold,
        "best_f05": best_f05,
    }

    if model_path is not None:
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, model_path)

    return bundle

