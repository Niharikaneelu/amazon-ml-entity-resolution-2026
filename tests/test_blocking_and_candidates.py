"""Unit tests for blocking and candidate generation modules."""
import sys
from pathlib import Path
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "business_entity_resolution"))

from src.blocking import (
    get_exact_normalized_name_key,
    get_country_normalized_name_key,
    get_country_name_token_keys,
    get_country_address_number_keys,
    build_blocking_index,
    block_pairs_for_strategy,
    union_candidate_sets,
)
from src.candidates import (
    generate_candidates_for_source,
    generate_all_candidates,
    evaluate_candidate_recall,
    format_candidates_for_submission,
)


def test_key_generation_functions():
    name = "Acme Corporation Inc."
    country = "US"
    address = "123 Main St Suite 400, Chicago IL 60601"

    # Exact normalized name
    exact_key = get_exact_normalized_name_key(name)
    assert exact_key == "acmecorporationinc"

    # Country + normalized name (suffixes stripped)
    cn_key = get_country_normalized_name_key(country, name)
    assert cn_key == "us__acme"

    # Country + name tokens
    tokens = get_country_name_token_keys(country, name)
    assert any("us__acme" in k for k in tokens)

    # Country + address number
    addr_keys = get_country_address_number_keys(country, name, address)
    assert any("us__123" in k for k in addr_keys)


def test_blocking_and_candidates_pipeline():
    s1_data = pd.DataFrame([
        {"entity_id": "S1-001", "business_name": "McDonalds Restaurant", "business_address": "123 Main St, Chicago 60601", "country": "US"},
        {"entity_id": "S1-002", "business_name": "Tata Consultancy Services", "business_address": "Tech Park Phase 1 Pune 411057", "country": "India"},
    ])

    s2_data = pd.DataFrame([
        {"entity_id": "S2-101", "business_name": "McDonalds Inc", "business_address": "123 Main Street Chicago 60601", "country": "US"},
        {"entity_id": "S2-102", "business_name": "TCS", "business_address": "Tech Park Hinjewadi Pune 411057", "country": "India"},
    ])

    s3_data = pd.DataFrame([
        {"entity_id": "S3-201", "business_name": "McDonald Corporation", "business_address": "123 N Main St Chicago 60601", "country": "US"},
    ])


    gt_data = pd.DataFrame([
        {"source1_entity_id": "S1-001", "matched_entity_ids": "S2-101,S3-201"},
        {"source1_entity_id": "S1-002", "matched_entity_ids": "S2-102"},
    ])

    # S1 -> S2 candidates
    cands_s2 = generate_candidates_for_source(s1_data, s2_data, "source2")
    assert not cands_s2.empty
    assert "source1_entity_id" in cands_s2.columns
    assert "candidate_entity_id" in cands_s2.columns
    assert "candidate_source" in cands_s2.columns
    assert set(cands_s2["candidate_source"]) == {"source2"}

    # S1 -> S3 candidates
    cands_s3 = generate_candidates_for_source(s1_data, s3_data, "source3")
    assert not cands_s3.empty
    assert set(cands_s3["candidate_source"]) == {"source3"}

    # All candidates
    all_cands = generate_all_candidates(s1_data, s2_data, s3_data)
    assert len(all_cands) >= 3

    # Recall evaluation
    recall_metrics = evaluate_candidate_recall(all_cands, gt_data)
    assert recall_metrics["candidate_recall_s2"] > 0.0
    assert recall_metrics["candidate_recall_s3"] > 0.0

    # Submission formatting
    formatted = format_candidates_for_submission(all_cands, ["S1-001", "S1-002"])
    assert len(formatted) == 2
    assert "source1_entity_id" in formatted.columns
    assert "candidate_entity_ids" in formatted.columns
    print("ALL TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_key_generation_functions()
    test_blocking_and_candidates_pipeline()

