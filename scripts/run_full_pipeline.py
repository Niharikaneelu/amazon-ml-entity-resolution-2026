"""Full pipeline on complete data - 2 fast vectorized strategies.
Trains on ALL train data, runs inference on ALL test data.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "business_entity_resolution"))
import io, os
os.environ["PYTHONIOENCODING"] = "utf-8"

from src.config import PipelineConfig
from src.load_data import read_tsv
from src.candidates import (
    generate_all_candidates, enrich_candidate_pairs,
    attach_ground_truth_labels, format_candidates_for_submission,
    evaluate_candidate_recall,
)
from src.train import train_model
from src.predict import predict_pairs, format_matching_results_for_submission
from src.evaluate import compute_macro_f05

STRATEGIES = ["exact_normalized_name", "country_normalized_name"]

config = PipelineConfig()
wall_start = time.time()
print("=" * 65)
print("FULL PIPELINE - Train on ALL data, Infer on ALL test data")
print("=" * 65)

# ── 1. Load ALL training data ─────────────────────────────────────────────────
print("\n[1/6] Loading ALL training data...")
t0 = time.time()
train_s1 = read_tsv(config.train_s1_path)
train_s2 = read_tsv(config.train_s2_path)
train_s3 = read_tsv(config.train_s3_path)
train_gt = read_tsv(config.train_gt_path)
print(f"  S1:{len(train_s1):,}  S2:{len(train_s2):,}  S3:{len(train_s3):,}  GT:{len(train_gt):,}  ({time.time()-t0:.1f}s)")

# Use all S1 entities
train_s1_sample = train_s1.copy()
train_gt_sample = train_gt.copy()

# ── 2. Candidate Generation ───────────────────────────────────────────────────
print("\n[2/6] Candidate Generation (exact + country_normalized strategies)...")
t0 = time.time()
train_cands = generate_all_candidates(train_s1_sample, train_s2, train_s3, strategies=STRATEGIES)
print(f"  Candidate pairs: {len(train_cands):,}  ({time.time()-t0:.1f}s)")

recall = evaluate_candidate_recall(train_cands, train_gt_sample)
print(f"  Recall S2:{recall['candidate_recall_s2']:.4f}  S3:{recall['candidate_recall_s3']:.4f}  Overall:{recall['candidate_recall_overall']:.4f}")
print(f"  True pairs found: S2={recall['true_pairs_s2_found']}/{recall['true_pairs_s2_total']}  S3={recall['true_pairs_s3_found']}/{recall['true_pairs_s3_total']}")

# ── 3. Feature Engineering + Train ───────────────────────────────────────────
print("\n[3/6] Feature Engineering & LightGBM Training...")
t0 = time.time()
enriched = enrich_candidate_pairs(train_cands, train_s1_sample, train_s2, train_s3)
labeled  = attach_ground_truth_labels(enriched, train_gt_sample)
pos = int((labeled["label"] == 1).sum())
neg = int((labeled["label"] == 0).sum())
print(f"  Pairs: {len(labeled):,}  positives={pos:,}  negatives={neg:,}  ratio=1:{neg//max(pos,1)}")

bundle = train_model(labeled, label_column="label", model_path=config.model_path)
print(f"  Threshold={bundle['best_threshold']:.2f}  Train F0.5={bundle['best_f05']:.4f}  ({time.time()-t0:.1f}s)")

# ── 4. Validation Macro F0.5 on train set ────────────────────────────────────
print("\n[4/6] Validation: Macro F_0.5 on training set...")
val_preds       = predict_pairs(enriched, bundle)
val_matching_df = format_matching_results_for_submission(val_preds, train_s1_sample["entity_id"])
metrics         = compute_macro_f05(val_matching_df, train_gt_sample)
print(f"  Macro F0.5      = {metrics['macro_f05']:.4f}  <-- COMPETITION METRIC")
print(f"  Macro Precision = {metrics['macro_precision']:.4f}")
print(f"  Macro Recall    = {metrics['macro_recall']:.4f}")
print(f"  Singleton Acc   = {metrics['singleton_accuracy']:.4f}  ({metrics['correct_singletons']}/{metrics['total_singletons']} correct)")
print(f"  Entities eval   = {metrics['total_eval_entities']:,}")

# ── 5. Test Inference on ALL test data ────────────────────────────────────────
print("\n[5/6] Running inference on ALL test data...")
t0 = time.time()
test_s1 = read_tsv(config.test_s1_path)
test_s2 = read_tsv(config.test_s2_path)
test_s3 = read_tsv(config.test_s3_path)
print(f"  Test S1:{len(test_s1):,}  S2:{len(test_s2):,}  S3:{len(test_s3):,}")

test_cands    = generate_all_candidates(test_s1, test_s2, test_s3, strategies=STRATEGIES)
enriched_test = enrich_candidate_pairs(test_cands, test_s1, test_s2, test_s3)
test_preds    = predict_pairs(enriched_test, bundle)
print(f"  Test candidate pairs: {len(test_cands):,}  ({time.time()-t0:.1f}s)")

# ── 6. Write submission files ─────────────────────────────────────────────────
print("\n[6/6] Writing submission files...")
config.output_dir.mkdir(parents=True, exist_ok=True)
matching_out  = format_matching_results_for_submission(test_preds, test_s1["entity_id"])
candidate_out = format_candidates_for_submission(test_cands, test_s1["entity_id"])

matching_out.to_csv(config.matching_output_path, sep="\t", index=False)
candidate_out.to_csv(config.candidate_output_path, sep="\t", index=False)

non_empty = int((matching_out["matched_entity_ids"].str.len() > 0).sum())
total_time = time.time() - wall_start

print(f"  matching_results.tsv : {len(matching_out):,} rows")
print(f"  candidate_pairs.tsv  : {len(candidate_out):,} rows")
print(f"  Entities WITH matches:  {non_empty:,}")
print(f"  Singletons (no match):  {len(matching_out)-non_empty:,}")
print(f"  Output: {config.output_dir}")

print("\n--- Sample: matching_results.tsv (10 with matches) ---")
sample = matching_out[matching_out["matched_entity_ids"].str.len() > 0].head(10)
print(sample.to_string(index=False) if not sample.empty else "(no matches found)")

print(f"\nTOTAL WALL TIME: {total_time:.1f}s")
print("=" * 65)
print("PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 65)
