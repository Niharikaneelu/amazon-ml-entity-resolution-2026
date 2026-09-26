from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from .evaluate import evaluate_predictions, find_optimal_threshold
from .features import FEATURE_COLUMNS, build_features, build_matching_features


def prepare_labeled_candidate_pairs(
    candidate_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    include_unblocked_ground_truth: bool = False,
) -> pd.DataFrame:
    """Label candidate pairs using ground truth.

    Args:
        candidate_df: DataFrame with columns: source1_entity_id, candidate_entity_id, candidate_source
        ground_truth_df: DataFrame with columns: source1_entity_id, matched_entity_ids
        include_unblocked_ground_truth: If True, positive ground truth pairs missed by candidates
                                       are appended as positive examples.

    Returns:
        DataFrame with columns: source1_entity_id, candidate_entity_id, candidate_source, label
    """
    positive_pairs: set[tuple[str, str]] = set()
    for _, row in ground_truth_df.iterrows():
        s1 = str(row["source1_entity_id"]).strip()
        matched = str(row["matched_entity_ids"])
        if pd.isna(matched) or not matched.strip():
            continue
        for m_id in matched.split(","):
            m_id = m_id.strip()
            if m_id:
                positive_pairs.add((s1, m_id))

    labeled = candidate_df.copy()
    labeled["label"] = [
        int((s1, cid) in positive_pairs)
        for s1, cid in zip(labeled["source1_entity_id"], labeled["candidate_entity_id"])
    ]

    if include_unblocked_ground_truth:
        s1_in_cands = set(candidate_df["source1_entity_id"])
        cand_pairs_set = set(zip(labeled["source1_entity_id"], labeled["candidate_entity_id"]))
        extra_positives = []
        for s1, cid in positive_pairs:
            if s1 in s1_in_cands and (s1, cid) not in cand_pairs_set:
                src = "source2" if str(cid).startswith("S2") else "source3"
                extra_positives.append({
                    "source1_entity_id": s1,
                    "candidate_entity_id": cid,
                    "candidate_source": src,
                    "label": 1,
                })
        if extra_positives:
            labeled = pd.concat([labeled, pd.DataFrame(extra_positives)], ignore_index=True)

    return labeled


def split_train_validation_by_source1(
    pairs: pd.DataFrame,
    val_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split pairs strictly by source1_entity_id to prevent data leakage.

    All candidate pairs for a given source1_entity_id are assigned exclusively
    to either the train set or the validation set.
    """
    if "source1_entity_id" not in pairs.columns:
        raise ValueError("pairs DataFrame must contain 'source1_entity_id' to split by entity")

    unique_s1 = pairs["source1_entity_id"].drop_duplicates().to_numpy()
    if len(unique_s1) <= 1:
        return pairs.copy(), pairs.copy()

    train_s1, val_s1 = train_test_split(unique_s1, test_size=val_size, random_state=random_state)
    train_s1_set = set(train_s1)
    val_s1_set = set(val_s1)

    train_pairs = pairs[pairs["source1_entity_id"].isin(train_s1_set)].copy().reset_index(drop=True)
    val_pairs = pairs[pairs["source1_entity_id"].isin(val_s1_set)].copy().reset_index(drop=True)
    return train_pairs, val_pairs


def train_matching_model(
    train_pairs: pd.DataFrame,
    val_pairs: pd.DataFrame | None = None,
    s1_df: pd.DataFrame | None = None,
    cand_df: pd.DataFrame | dict[str, pd.DataFrame] | list[pd.DataFrame] | None = None,
    label_column: str = "label",
    model_path: Path | str | None = None,
    val_size: float = 0.2,
    random_state: int = 42,
    model_type: str = "gbdt",
    beta: float = 0.5,
    **model_kwargs: Any,
) -> dict[str, Any]:
    """Train a Gradient Boosted Decision Tree (or baseline) and optimize probability threshold for F0.5.

    Args:
        train_pairs: Candidate pairs DataFrame with label column.
        val_pairs: Optional validation candidate pairs DataFrame.
        s1_df: Optional Source 1 DataFrame to merge attributes.
        cand_df: Optional candidate DataFrame(s) to merge attributes.
        label_column: Target binary match column name (default 'label').
        model_path: Optional path to save serialized model bundle.
        val_size: Fraction of source1 entities for validation if val_pairs is None.
        random_state: Random seed for reproducibility.
        model_type: Classifier type: 'gbdt' (LightGBM/HistGBDT), 'lightgbm',
                    'hist_gradient_boosting', or 'logistic_regression'.
        beta: Beta parameter for F-score optimization (default 0.5).
        **model_kwargs: Additional parameters passed to the classifier.

    Returns:
        Dictionary containing trained model, best_threshold, metrics, and feature list.
    """
    if val_pairs is None:
        if "source1_entity_id" in train_pairs.columns:
            train_set, val_set = split_train_validation_by_source1(
                train_pairs, val_size=val_size, random_state=random_state
            )
        else:
            train_set, val_set = train_test_split(
                train_pairs, test_size=val_size, random_state=random_state
            )
    else:
        train_set = train_pairs
        val_set = val_pairs

    X_train = build_matching_features(train_set, s1_df=s1_df, cand_df=cand_df)
    y_train = train_set[label_column].astype(int)

    X_val = build_matching_features(val_set, s1_df=s1_df, cand_df=cand_df)
    y_val = val_set[label_column].astype(int)

    feature_cols = [c for c in FEATURE_COLUMNS if c in X_train.columns]

    # Calculate class imbalance weighting
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    scale_pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0

    model: Any = None
    resolved_type = model_type.lower()

    if resolved_type in ("gbdt", "lightgbm") and HAS_LIGHTGBM:
        default_lgb_params = {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
            "scale_pos_weight": scale_pos_weight,
            "random_state": random_state,
            "verbose": -1,
            "n_jobs": -1,
        }
        default_lgb_params.update(model_kwargs)
        model = lgb.LGBMClassifier(**default_lgb_params)
        model.fit(
            X_train[feature_cols],
            y_train,
            eval_set=[(X_val[feature_cols], y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
        )
    elif resolved_type in ("gbdt", "hist_gradient_boosting"):
        default_hgb_params = {
            "max_iter": 200,
            "learning_rate": 0.05,
            "max_leaf_nodes": 31,
            "class_weight": "balanced",
            "random_state": random_state,
            "early_stopping": True,
        }
        default_hgb_params.update(model_kwargs)
        model = HistGradientBoostingClassifier(**default_hgb_params)
        model.fit(X_train[feature_cols], y_train)
    elif resolved_type == "logistic_regression":
        default_lr_params = {
            "C": 1.0,
            "max_iter": 1000,
            "class_weight": "balanced",
            "random_state": random_state,
            "solver": "lbfgs",
        }
        default_lr_params.update(model_kwargs)
        model = LogisticRegression(**default_lr_params)
        model.fit(X_train[feature_cols], y_train)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    val_probs = model.predict_proba(X_val[feature_cols])[:, 1]
    best_th, best_f_beta, eval_table = find_optimal_threshold(y_val, val_probs, beta=beta)

    val_predictions = val_set.copy()
    val_predictions["match_probability"] = val_probs
    val_predictions["match"] = (val_probs >= best_th).astype(int)
    metrics = evaluate_predictions(
        val_predictions,
        label_column=label_column,
        probability_column="match_probability",
        threshold=best_th,
        beta=beta,
    )

    bundle: dict[str, Any] = {
        "model": model,
        "model_type": resolved_type,
        "fields": feature_cols,
        "feature_columns": feature_cols,
        "best_threshold": best_th,
        "best_f_beta": best_f_beta,
        "val_metrics": metrics,
        "eval_table": eval_table,
    }

    if model_path is not None:
        p = Path(model_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, p)

    return bundle


def predict_matching_candidates(
    pairs: pd.DataFrame,
    model_bundle: dict[str, Any] | Path | str,
    threshold: float | None = None,
    s1_df: pd.DataFrame | None = None,
    cand_df: pd.DataFrame | dict[str, pd.DataFrame] | list[pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Generate predictions for candidate pairs using a trained model bundle.

    Returns:
        DataFrame containing:
            source1_entity_id
            candidate_entity_id
            candidate_source
            match_probability
            match
    """
    if isinstance(model_bundle, (str, Path)):
        bundle = joblib.load(model_bundle)
    else:
        bundle = model_bundle

    model = bundle["model"]
    feature_cols = bundle.get("feature_columns", bundle.get("fields", FEATURE_COLUMNS))
    th = threshold if threshold is not None else bundle.get("best_threshold", 0.5)

    features = build_matching_features(pairs, s1_df=s1_df, cand_df=cand_df)
    missing_cols = [c for c in feature_cols if c not in features.columns]
    for c in missing_cols:
        features[c] = 0.0

    probs = model.predict_proba(features[feature_cols])[:, 1]

    cand_source = (
        pairs["candidate_source"]
        if "candidate_source" in pairs.columns
        else [""] * len(pairs)
    )

    result = pd.DataFrame(
        {
            "source1_entity_id": pairs["source1_entity_id"],
            "candidate_entity_id": pairs["candidate_entity_id"],
            "candidate_source": cand_source,
            "match_probability": probs,
            "match": (probs >= th).astype(int),
        },
        index=pairs.index,
    )
    return result


def train_model(
    pairs: pd.DataFrame,
    fields: list[str] | None = None,
    label_column: str = "label",
    model_path: Path | str | None = None,
    model_type: str = "gbdt",
    **kwargs: Any,
) -> Any:
    """Train a matching model. Maintains backward compatibility with legacy pipelines."""
    if "source1_entity_id" in pairs.columns:
        bundle = train_matching_model(
            train_pairs=pairs,
            label_column=label_column,
            model_path=model_path,
            model_type=model_type,
            **kwargs,
        )
        return bundle["model"]

    # Legacy fallback for pairs with _left and _right columns
    features = build_features(pairs, columns=fields)
    resolved_type = model_type.lower()
    if resolved_type in ("gbdt", "lightgbm") and HAS_LIGHTGBM:
        model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, verbose=-1, random_state=42)
    elif resolved_type in ("gbdt", "hist_gradient_boosting"):
        model = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, random_state=42)
    else:
        model = LogisticRegression(max_iter=1000, class_weight="balanced")

    model.fit(features, pairs[label_column].astype(int))

    bundle = {
        "model": model,
        "fields": list(features.columns),
        "feature_columns": list(features.columns),
        "best_threshold": 0.5,
    }
    if model_path is not None:
        p = Path(model_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, p)

    return model
