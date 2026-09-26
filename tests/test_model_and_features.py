"""Unit tests for feature engineering, model training, threshold evaluation, and prediction."""
import sys
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "business_entity_resolution"))

from src.features import (
    FEATURE_COLUMNS,
    build_features,
    build_matching_features,
    merge_entity_attributes,
)
from src.evaluate import (
    calculate_f_beta,
    evaluate_thresholds,
    find_optimal_threshold,
    evaluate_predictions,
)
from src.train import (
    prepare_labeled_candidate_pairs,
    split_train_validation_by_source1,
    train_matching_model,
    predict_matching_candidates,
)


def test_feature_columns_present_and_values():
    s1_df = pd.DataFrame([
        {
            "entity_id": "S1-001",
            "business_name": "Target Store 101",
            "business_address": "123 Main St, Minneapolis, MN",
            "country": "US",
        },
        {
            "entity_id": "S1-002",
            "business_name": "Acme Widgets",
            "business_address": "456 Oak Avenue",
            "country": "US",
        },
        {
            "entity_id": "S1-003",
            "business_name": None,
            "business_address": None,
            "country": "US",
        },
    ])

    cand_df = pd.DataFrame([
        {
            "entity_id": "S2-001",
            "business_name": "Target Store 101",
            "business_address": "123 Main Street, Minneapolis, MN",
            "country": "US",
        },
        {
            "entity_id": "S2-002",
            "business_name": "Widgets Acme",
            "business_address": "999 Pine Rd",
            "country": "Canada",
        },
        {
            "entity_id": "S3-003",
            "business_name": "Random Corp",
            "business_address": "789 Elm St",
            "country": "US",
        },
    ])

    pairs = pd.DataFrame([
        {"source1_entity_id": "S1-001", "candidate_entity_id": "S2-001", "candidate_source": "source2"},
        {"source1_entity_id": "S1-002", "candidate_entity_id": "S2-002", "candidate_source": "source2"},
        {"source1_entity_id": "S1-003", "candidate_entity_id": "S3-003", "candidate_source": "source3"},
    ])

    feats = build_matching_features(pairs, s1_df=s1_df, cand_df=cand_df)

    # Check that all minimum required features are present
    expected_features = [
        "name_exact",
        "name_similarity",
        "name_token_similarity",
        "address_exact",
        "address_similarity",
        "address_token_similarity",
        "country_match",
        "house_number_match",
        "name_missing",
        "address_missing",
    ]
    for col in expected_features:
        assert col in feats.columns

    # Pair 1: Exact name match, high address similarity, matching house number 123, matching country US
    assert feats.loc[0, "name_exact"] == 1.0
    assert feats.loc[0, "name_similarity"] == 1.0
    assert feats.loc[0, "name_token_similarity"] == 1.0
    assert feats.loc[0, "house_number_match"] == 1.0
    assert feats.loc[0, "country_match"] == 1.0
    assert feats.loc[0, "name_missing"] == 0.0
    assert feats.loc[0, "address_missing"] == 0.0

    # Pair 2: Reordered tokens "Acme Widgets" vs "Widgets Acme", different address numbers, country mismatch
    assert feats.loc[1, "name_exact"] == 0.0
    assert feats.loc[1, "name_token_similarity"] == 1.0
    assert feats.loc[1, "house_number_match"] == 0.0
    assert feats.loc[1, "country_match"] == 0.0

    # Pair 3: Missing name and address on S1
    assert feats.loc[2, "name_missing"] == 1.0
    assert feats.loc[2, "address_missing"] == 1.0
    assert feats.loc[2, "name_exact"] == 0.0
    assert feats.loc[2, "name_similarity"] == 0.0


def test_prepare_labeled_candidate_pairs_multi_match():
    candidate_df = pd.DataFrame([
        {"source1_entity_id": "S1-10", "candidate_entity_id": "S2-100", "candidate_source": "source2"},
        {"source1_entity_id": "S1-10", "candidate_entity_id": "S3-200", "candidate_source": "source3"},
        {"source1_entity_id": "S1-10", "candidate_entity_id": "S2-999", "candidate_source": "source2"},
        {"source1_entity_id": "S1-20", "candidate_entity_id": "S2-300", "candidate_source": "source2"},
    ])

    # Ground truth: S1-10 matches both S2-100 and S3-200 (multi-match)
    ground_truth_df = pd.DataFrame([
        {"source1_entity_id": "S1-10", "matched_entity_ids": "S2-100,S3-200"},
        {"source1_entity_id": "S1-20", "matched_entity_ids": "S2-300"},
    ])

    labeled = prepare_labeled_candidate_pairs(candidate_df, ground_truth_df)
    assert "label" in labeled.columns

    # S1-10 with S2-100 and S3-200 should be 1
    assert labeled.loc[0, "label"] == 1
    assert labeled.loc[1, "label"] == 1
    # S1-10 with S2-999 should be 0 (negative candidate)
    assert labeled.loc[2, "label"] == 0
    # S1-20 with S2-300 should be 1
    assert labeled.loc[3, "label"] == 1


def test_split_train_validation_by_source1_no_leakage():
    pairs = pd.DataFrame([
        {"source1_entity_id": f"S1-{i // 4}", "candidate_entity_id": f"CAND-{i}", "candidate_source": "source2", "label": i % 2}
        for i in range(40)
    ])

    train_pairs, val_pairs = split_train_validation_by_source1(pairs, val_size=0.3, random_state=42)

    train_s1 = set(train_pairs["source1_entity_id"])
    val_s1 = set(val_pairs["source1_entity_id"])

    # Strict zero leakage: set intersection must be empty
    assert len(train_s1 & val_s1) == 0
    assert len(train_s1) > 0
    assert len(val_s1) > 0
    assert len(train_pairs) + len(val_pairs) == len(pairs)


def test_f05_and_threshold_evaluation():
    # Test F0.5 formula
    prec = 0.8
    rec = 0.5
    expected_f05 = (1.25 * (prec * rec)) / (0.25 * prec + rec)
    assert np.isclose(calculate_f_beta(prec, rec, beta=0.5), expected_f05)

    # Test threshold evaluation and optimal threshold finding
    y_true = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 0])
    y_prob = np.array([0.9, 0.85, 0.7, 0.65, 0.4, 0.3, 0.75, 0.2, 0.8, 0.1])

    best_th, best_f05, eval_table = find_optimal_threshold(y_true, y_prob, beta=0.5)

    assert 0.0 < best_th < 1.0
    assert best_f05 > 0.0
    assert not eval_table.empty
    assert "threshold" in eval_table.columns
    assert "f_0_5" in eval_table.columns
    assert "precision" in eval_table.columns
    assert "recall" in eval_table.columns


def test_train_and_predict_pipeline():
    # Build toy dataset with realistic features
    n_samples = 60
    s1_rows = []
    cand_rows = []
    pairs_rows = []

    for i in range(n_samples):
        s1_id = f"S1-{i}"
        match_id = f"S2-{i}"
        non_match_id = f"S2-{1000 + i}"

        s1_rows.append({
            "entity_id": s1_id,
            "business_name": f"Enterprise Business {i} LLC",
            "business_address": f"{100 + i} Market Street, City {i % 5}",
            "country": "US",
        })

        # True match
        cand_rows.append({
            "entity_id": match_id,
            "business_name": f"Enterprise Business {i}",
            "business_address": f"{100 + i} Market St, City {i % 5}",
            "country": "US",
        })
        pairs_rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_id": match_id,
            "candidate_source": "source2",
            "label": 1,
        })

        # Negative candidate
        cand_rows.append({
            "entity_id": non_match_id,
            "business_name": f"Completely Different Name {i}",
            "business_address": f"{9000 + i} Random Ave, OtherCity",
            "country": "UK",
        })
        pairs_rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_id": non_match_id,
            "candidate_source": "source2",
            "label": 0,
        })

    s1_df = pd.DataFrame(s1_rows)
    cand_df = pd.DataFrame(cand_rows)
    pairs_df = pd.DataFrame(pairs_rows)

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / "matching_model.joblib"

        bundle = train_matching_model(
            train_pairs=pairs_df,
            s1_df=s1_df,
            cand_df=cand_df,
            model_path=model_path,
            val_size=0.25,
            random_state=42,
            beta=0.5,
        )

        assert model_path.exists()
        assert "model" in bundle
        assert "best_threshold" in bundle
        assert "best_f_beta" in bundle
        assert bundle["best_f_beta"] > 0.8  # Clean separable data should yield high F0.5

        # Test predict_matching_candidates output schema
        test_pairs = pairs_df.head(10).drop(columns=["label"])
        preds = predict_matching_candidates(
            test_pairs,
            model_bundle=model_path,
            s1_df=s1_df,
            cand_df=cand_df,
        )

        expected_cols = [
            "source1_entity_id",
            "candidate_entity_id",
            "candidate_source",
            "match_probability",
            "match",
        ]
        for col in expected_cols:
            assert col in preds.columns

        assert len(preds) == len(test_pairs)
        assert ((preds["match_probability"] >= 0.0) & (preds["match_probability"] <= 1.0)).all()
        assert set(preds["match"].unique()).issubset({0, 1})


def test_train_logistic_regression_fallback():
    # Verify logistic regression can also still be trained explicitly
    pairs_df = pd.DataFrame([
        {"source1_entity_id": f"S1-{i}", "candidate_entity_id": f"S2-{i}", "candidate_source": "source2", "label": i % 2,
         "business_name_s1": f"Acme {i}", "business_name_cand": f"Acme {i}" if i % 2 == 1 else "Different",
         "business_address_s1": f"{100+i} Main St", "business_address_cand": f"{100+i} Main St",
         "country_s1": "US", "country_cand": "US"}
        for i in range(30)
    ])

    bundle = train_matching_model(
        train_pairs=pairs_df,
        model_type="logistic_regression",
        val_size=0.3,
        random_state=42,
    )
    assert bundle["model_type"] == "logistic_regression"
    assert "best_threshold" in bundle
    assert "model" in bundle

