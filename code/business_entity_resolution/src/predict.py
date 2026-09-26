from pathlib import Path
import joblib
import pandas as pd
from .features import build_features


def predict_pairs(
    pairs: pd.DataFrame,
    model_or_path: object,
    threshold: float | None = None
) -> pd.DataFrame:
    """Predict match probabilities and binary matches for candidate pairs."""
    if pairs.empty:
        res = pairs.copy()
        res["match_probability"] = []
        res["match"] = []
        return res

    if isinstance(model_or_path, (str, Path)):
        bundle = joblib.load(model_or_path)
    elif isinstance(model_or_path, dict):
        bundle = model_or_path
    else:
        bundle = {"model": model_or_path, "best_threshold": 0.80}

    model = bundle["model"]
    if threshold is None:
        threshold = bundle.get("best_threshold", 0.80)

    features = build_features(pairs)
    probs = model.predict_proba(features)[:, 1]

    result = pairs.copy()
    result["match_probability"] = probs
    result["match"] = (probs >= threshold).astype(int)
    return result


def format_matching_results_for_submission(
    predicted_pairs: pd.DataFrame,
    s1_entity_ids: list[str] | set[str],
) -> pd.DataFrame:
    """Format predicted positive matches into submission schema matching_results.tsv.

    Columns: source1_entity_id, matched_entity_ids (comma-separated string).
    Guarantees every S1 entity appears exactly once.
    """
    if predicted_pairs.empty:
        positive_pairs = pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"])
    else:
        positive_pairs = predicted_pairs[predicted_pairs["match"] == 1]

    if positive_pairs.empty:
        grouped = {}
    else:
        grouped = positive_pairs.groupby("source1_entity_id")["candidate_entity_id"].apply(
            lambda ids: ",".join(dict.fromkeys(ids))
        ).to_dict()

    rows = []
    for s1_id in s1_entity_ids:
        matches = grouped.get(s1_id, "")
        rows.append({"source1_entity_id": s1_id, "matched_entity_ids": matches})

    return pd.DataFrame(rows)

