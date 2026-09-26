"""Unit test suite for end-to-end pipeline, feature extraction, ML training, and evaluation."""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "business_entity_resolution"))

from src.blocking import (
    get_exact_normalized_name_key,
    get_country_normalized_name_key,
)
from src.candidates import (
    generate_all_candidates,
    enrich_candidate_pairs,
    attach_ground_truth_labels,
    format_candidates_for_submission,
)
from src.features import build_features, extract_pair_features
from src.train import train_model
from src.predict import predict_pairs, format_matching_results_for_submission
from src.evaluate import compute_macro_f05, compute_f05_score


def test_feature_extraction():
    feat = extract_pair_features(
        name_a="Starbucks Coffee Inc",
        name_b="Starbucks Store 102",
        addr_a="100 Main St Suite 5, Seattle WA 98101",
        addr_b="100 Main Street Seattle WA 98101",
        country_a="US",
        country_b="US"
    )
    assert feat["name_token_set_ratio"] > 0.6
    assert feat["address_number_match"] == 1.0
    assert feat["country_match"] == 1.0


def test_pipeline_integration():
    s1 = pd.DataFrame([
        {"entity_id": "S1-001", "business_name": "Apple Inc", "business_address": "1 Infinite Loop Cupertino CA 95014", "country": "US"},
        {"entity_id": "S1-002", "business_name": "Google LLC", "business_address": "1600 Amphitheatre Pkwy Mountain View CA", "country": "US"},
        {"entity_id": "S1-003", "business_name": "Random Singleton Corp", "business_address": "999 Unknown Street France", "country": "France"},
    ])

    s2 = pd.DataFrame([
        {"entity_id": "S2-101", "business_name": "Apple Corporation", "business_address": "1 Infinite Loop Cupertino 95014", "country": "US"},
        {"entity_id": "S2-102", "business_name": "Google Mountain View", "business_address": "1600 Amphitheatre Parkway CA 94043", "country": "US"},
    ])

    s3 = pd.DataFrame([
        {"entity_id": "S3-201", "business_name": "Apple Store Cupertino", "business_address": "1 Infinite Loop CA", "country": "US"},
    ])

    gt = pd.DataFrame([
        {"source1_entity_id": "S1-001", "matched_entity_ids": "S2-101,S3-201"},
        {"source1_entity_id": "S1-002", "matched_entity_ids": "S2-102"},
        {"source1_entity_id": "S1-003", "matched_entity_ids": ""},
    ])

    cands = generate_all_candidates(s1, s2, s3)
    assert not cands.empty

    enriched = enrich_candidate_pairs(cands, s1, s2, s3)
    labeled = attach_ground_truth_labels(enriched, gt)

    bundle = train_model(labeled, label_column="label")
    assert bundle["model"] is not None
    assert bundle["best_threshold"] >= 0.5

    preds = predict_pairs(enriched, bundle)
    assert "match_probability" in preds.columns
    assert "match" in preds.columns

    matching_tsv = format_matching_results_for_submission(preds, s1["entity_id"])
    candidate_tsv = format_candidates_for_submission(cands, s1["entity_id"])

    assert len(matching_tsv) == 3
    assert len(candidate_tsv) == 3

    # Macro F0.5 evaluation
    eval_metrics = compute_macro_f05(matching_tsv, gt)
    assert eval_metrics["macro_f05"] > 0.0
    assert eval_metrics["singleton_accuracy"] == 1.0  # S1-003 correctly predicted empty

    print("END-TO-END PIPELINE INTEGRATION TEST PASSED!")


if __name__ == "__main__":
    test_feature_extraction()
    test_pipeline_integration()
