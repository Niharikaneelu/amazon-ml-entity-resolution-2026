"""End-to-end Business Entity Resolution Pipeline execution script.

Optimized for high-precision macro F_0.5 score evaluation.
"""
import sys
import os
from pathlib import Path
import time
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "code" / "business_entity_resolution"))

from src.config import PipelineConfig
from src.load_data import read_tsv
from src.candidates import (
    generate_all_candidates,
    enrich_candidate_pairs,
    attach_ground_truth_labels,
    format_candidates_for_submission,
    evaluate_candidate_recall,
)
from src.train import train_model
from src.predict import predict_pairs, format_matching_results_for_submission
from src.evaluate import compute_macro_f05


def run_pipeline(
    sample_size: int | None = 10000,
    test_sample_size: int | None = 5000,
    target_nrows: int | None = 100000,
) -> None:
    start_time = time.time()
    config = PipelineConfig()
    print("=" * 70)
    print("AMAZON ML CHALLENGE 2026: BUSINESS ENTITY RESOLUTION PIPELINE")
    print(f"Dataset directory: {config.dataset_dir}")
    print(f"Output directory:  {config.output_dir}")
    print("=" * 70)

    # 1. Load Training Data
    print("\n[1/6] Loading training dataset...")
    train_s1 = read_tsv(config.train_s1_path, nrows=sample_size * 2 if sample_size else None)
    train_s2 = read_tsv(config.train_s2_path, nrows=target_nrows)
    train_s3 = read_tsv(config.train_s3_path, nrows=target_nrows)
    train_gt = read_tsv(config.train_gt_path, nrows=sample_size * 2 if sample_size else None)


    print(f"  Loaded Train S1: {len(train_s1):,} records")
    print(f"  Loaded Train S2: {len(train_s2):,} records")
    print(f"  Loaded Train S3: {len(train_s3):,} records")
    print(f"  Loaded Train GT: {len(train_gt):,} records")

    # Sample for training if sample_size specified
    if sample_size and sample_size < len(train_s1):
        print(f"  Sampling {sample_size:,} S1 records for fast, high-coverage model training...")
        train_s1_sample = train_s1.sample(n=sample_size, random_state=42).reset_index(drop=True)
        train_gt_sample = train_gt[train_gt["source1_entity_id"].isin(set(train_s1_sample["entity_id"]))].reset_index(drop=True)
    else:
        train_s1_sample = train_s1
        train_gt_sample = train_gt

    # 2. Stage 1: Candidate Generation (Blocking)
    print("\n[2/6] Stage 1: Candidate Generation & Blocking (Maximizing Recall Ceiling)...")
    train_cands = generate_all_candidates(train_s1_sample, train_s2, train_s3)
    recall_stats = evaluate_candidate_recall(train_cands, train_gt_sample)
    print(f"  Total Candidate Pairs Generated: {recall_stats['total_candidate_pairs']:,}")
    print(f"  S2 Candidate Recall Ceiling:     {recall_stats['candidate_recall_s2']:.4f}")
    print(f"  S3 Candidate Recall Ceiling:     {recall_stats['candidate_recall_s3']:.4f}")
    print(f"  Overall Candidate Recall Ceiling: {recall_stats['candidate_recall_overall']:.4f}")
    print(f"  Avg Candidates per S1 Entity:    {recall_stats['avg_candidates_per_s1']:.2f}")

    # 3. Stage 2: Feature Engineering & Model Training
    print("\n[3/6] Stage 2: Feature Engineering & LightGBM Training (Optimizing F_0.5)...")
    enriched_train = enrich_candidate_pairs(train_cands, train_s1_sample, train_s2, train_s3)
    labeled_train = attach_ground_truth_labels(enriched_train, train_gt_sample)

    model_bundle = train_model(
        candidate_pairs=labeled_train,
        label_column="label",
        model_path=config.model_path,
        val_gt_df=train_gt_sample,
    )
    best_th = model_bundle["best_threshold"]
    print(f"  Trained LightGBM Matcher Model.")
    print(f"  Optimized Decision Threshold (F_0.5): {best_th:.2f}")
    print(f"  Training F_0.5 Score:                 {model_bundle['best_f05']:.4f}")

    # Validation Macro F_0.5 Evaluation
    val_preds = predict_pairs(enriched_train, model_bundle, threshold=best_th)
    val_matching_df = format_matching_results_for_submission(val_preds, train_s1_sample["entity_id"])
    val_macro_metrics = compute_macro_f05(val_matching_df, train_gt_sample)
    print("\n  --- VALIDATION METRICS ---")
    print(f"  Macro F_0.5 Score:      {val_macro_metrics['macro_f05']:.4f}")
    print(f"  Macro Precision:        {val_macro_metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:           {val_macro_metrics['macro_recall']:.4f}")
    print(f"  Singleton Accuracy:     {val_macro_metrics['singleton_accuracy']:.4f}")

    # 4. Load Test Data & Run Inference
    print("\n[4/6] Loading Test Dataset & Running Full Pipeline Inference...")
    test_s1 = read_tsv(config.test_s1_path, nrows=test_sample_size * 2 if test_sample_size else None)
    test_s2 = read_tsv(config.test_s2_path, nrows=target_nrows)
    test_s3 = read_tsv(config.test_s3_path, nrows=target_nrows)

    print(f"  Loaded Test S1: {len(test_s1):,} records")
    print(f"  Loaded Test S2: {len(test_s2):,} records (capped at {target_nrows:,})")
    print(f"  Loaded Test S3: {len(test_s3):,} records (capped at {target_nrows:,})")


    if test_sample_size and test_sample_size < len(test_s1):
        print(f"  Running pipeline on test set slice of {test_sample_size:,} S1 entities for validation...")
        test_s1_eval = test_s1.iloc[:test_sample_size].reset_index(drop=True)
    else:
        test_s1_eval = test_s1

    # Stage 1 Test Candidates
    test_cands = generate_all_candidates(test_s1_eval, test_s2, test_s3)
    print(f"  Test Candidates Generated: {len(test_cands):,} pairs")

    # Stage 2 Test Match Prediction
    enriched_test = enrich_candidate_pairs(test_cands, test_s1_eval, test_s2, test_s3)
    test_preds = predict_pairs(enriched_test, model_bundle, threshold=best_th)

    # 5. Format & Save Outputs
    print("\n[5/6] Formatting & Saving Output Submissions...")
    config.output_dir.mkdir(parents=True, exist_ok=True)

    matching_df = format_matching_results_for_submission(test_preds, test_s1_eval["entity_id"])
    candidate_df = format_candidates_for_submission(test_cands, test_s1_eval["entity_id"])

    matching_df.to_csv(config.matching_output_path, sep="\t", index=False)
    candidate_df.to_csv(config.candidate_output_path, sep="\t", index=False)

    print(f"  Saved matching_results.tsv ({len(matching_df):,} rows) to: {config.matching_output_path}")
    print(f"  Saved candidate_pairs.tsv  ({len(candidate_df):,} rows) to: {config.candidate_output_path}")

    # 6. Validate Output Formatting
    print("\n[6/6] Running Submission Validator (validate_submission.py)...")
    validator_script = (
        config.project_root
        / "6ab10eb3b23ba_student_resource"
        / "student_resource"
        / "utils"
        / "validate_submission.py"
    )
    if validator_script.exists():
        import subprocess
        cmd = [
            sys.executable,
            str(validator_script),
            "--matching", str(config.matching_output_path),
            "--candidate", str(config.candidate_output_path),
            "--test-dir", str(config.test_dir),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print("  Validator Output:\n" + res.stdout)
        if res.returncode == 0:
            print("  SUCCESS: Submission files PASSED validation successfully!")
        else:
            print("  WARNING/ERROR in Validator:\n" + res.stderr)

    elapsed = time.time() - start_time
    print("=" * 70)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS!")
    print("=" * 70)


if __name__ == "__main__":
    # Run pipeline:
    #   sample_size     = how many S1 train entities used for model training
    #   test_sample_size= how many S1 test entities to run inference on
    #   target_nrows    = cap on rows loaded from S2/S3 (keeps runtime fast)
    run_pipeline(sample_size=5000, test_sample_size=3000, target_nrows=100000)

