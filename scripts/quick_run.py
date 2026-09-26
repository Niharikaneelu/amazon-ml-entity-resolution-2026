"""Quick smoke test pipeline - loads all S2/S3 for real match discovery."""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "business_entity_resolution"))

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

config = PipelineConfig()
print("=" * 65)
print("PIPELINE RUN - Full S2/S3 for real match discovery")
print("=" * 65)
print(f"Dataset dir: {config.dataset_dir}")

# ── Config ─────────────────────────────────────────────────────────────────────
# KEY FIX: Load ALL of S2 and S3 so true match partners are reachable.
# Blocking is fully vectorized (pandas merge), so this is fast.
TRAIN_S1_N      = 5000    # S1 training entities
TARGET_N        = None    # None = load ALL rows from S2/S3
TEST_S1_N       = 5000    # S1 test entities

# Use all 4 strategies for best recall
ALL_STRATEGIES = [
    "exact_normalized_name",
    "country_normalized_name",
    "country_name_tokens",
    "country_address_number",
]

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("\n[1/6] Loading data...")
t0 = time.time()
train_s1 = read_tsv(config.train_s1_path, nrows=TRAIN_S1_N)
train_s2 = read_tsv(config.train_s2_path, nrows=TARGET_N)
train_s3 = read_tsv(config.train_s3_path, nrows=TARGET_N)
train_gt = read_tsv(config.train_gt_path, nrows=TRAIN_S1_N)
print(f"  S1:{len(train_s1):,}  S2:{len(train_s2):,}  S3:{len(train_s3):,}  GT:{len(train_gt):,}  ({time.time()-t0:.1f}s)")

train_s1_sample = train_s1.sample(n=min(TRAIN_S1_N, len(train_s1)), random_state=42).reset_index(drop=True)
train_gt_sample = train_gt[
    train_gt["source1_entity_id"].isin(set(train_s1_sample["entity_id"]))
].reset_index(drop=True)

# ── 2. Candidate Generation ───────────────────────────────────────────────────
print("\n[2/6] Stage 1: Candidate Generation & Blocking (all 4 strategies)...")
t0 = time.time()
train_cands = generate_all_candidates(
    train_s1_sample, train_s2, train_s3, strategies=ALL_STRATEGIES
)
elapsed = time.time() - t0
print(f"  Total candidate pairs: {len(train_cands):,}  ({elapsed:.1f}s)")

recall = evaluate_candidate_recall(train_cands, train_gt_sample)
print(f"  Candidate Recall -> S2:{recall['candidate_recall_s2']:.4f}  S3:{recall['candidate_recall_s3']:.4f}  Overall:{recall['candidate_recall_overall']:.4f}")
print(f"  True pairs found: S2={recall['true_pairs_s2_found']}/{recall['true_pairs_s2_total']}  S3={recall['true_pairs_s3_found']}/{recall['true_pairs_s3_total']}")
print(f"  Avg candidates per S1: {recall['avg_candidates_per_s1']:.1f}")

# ── 3. Enrich + Label + Train ────────────────────────────────────────────────
print("\n[3/6] Stage 2: Feature Engineering & LightGBM Training...")
t0 = time.time()
enriched = enrich_candidate_pairs(train_cands, train_s1_sample, train_s2, train_s3)
labeled  = attach_ground_truth_labels(enriched, train_gt_sample)
pos = int((labeled["label"] == 1).sum())
neg = int((labeled["label"] == 0).sum())
ratio = pos / neg if neg > 0 else 0
print(f"  Labeled pairs: {len(labeled):,}  positives={pos:,}  negatives={neg:,}  ratio=1:{ratio:.0f}")

bundle = train_model(labeled, label_column="label", model_path=config.model_path)
print(f"  Model trained. Best threshold={bundle['best_threshold']:.2f}  Training F0.5={bundle['best_f05']:.4f}  ({time.time()-t0:.1f}s)")

# ── 4. Validation Macro F0.5 ─────────────────────────────────────────────────
print("\n[4/6] Validation: Macro F_0.5 on training sample...")
val_preds       = predict_pairs(enriched, bundle)
val_matching_df = format_matching_results_for_submission(val_preds, train_s1_sample["entity_id"])
metrics         = compute_macro_f05(val_matching_df, train_gt_sample)

print(f"  {'Macro F0.5':<20} = {metrics['macro_f05']:.4f}")
print(f"  {'Macro Precision':<20} = {metrics['macro_precision']:.4f}")
print(f"  {'Macro Recall':<20} = {metrics['macro_recall']:.4f}")
print(f"  {'Singleton Accuracy':<20} = {metrics['singleton_accuracy']:.4f}  ({metrics['correct_singletons']}/{metrics['total_singletons']} correct)")
print(f"  {'Total entities eval':<20} = {metrics['total_eval_entities']:,}")

# ── 5. Test Inference ─────────────────────────────────────────────────────────
print("\n[5/6] Running inference on TEST dataset...")
t0 = time.time()
test_s1 = read_tsv(config.test_s1_path, nrows=TEST_S1_N)
test_s2 = read_tsv(config.test_s2_path, nrows=TARGET_N)
test_s3 = read_tsv(config.test_s3_path, nrows=TARGET_N)
test_s1_eval = test_s1.iloc[:TEST_S1_N].reset_index(drop=True)
print(f"  Test S1: {len(test_s1_eval):,}  Test S2: {len(test_s2):,}  Test S3: {len(test_s3):,}")

test_cands    = generate_all_candidates(test_s1_eval, test_s2, test_s3, strategies=ALL_STRATEGIES)
enriched_test = enrich_candidate_pairs(test_cands, test_s1_eval, test_s2, test_s3)
test_preds    = predict_pairs(enriched_test, bundle)
print(f"  Test candidate pairs: {len(test_cands):,}  ({time.time()-t0:.1f}s)")

# ── 6. Write outputs ──────────────────────────────────────────────────────────
print("\n[6/6] Writing output files...")
config.output_dir.mkdir(parents=True, exist_ok=True)
matching_out  = format_matching_results_for_submission(test_preds, test_s1_eval["entity_id"])
candidate_out = format_candidates_for_submission(test_cands, test_s1_eval["entity_id"])

matching_out.to_csv(config.matching_output_path, sep="\t", index=False)
candidate_out.to_csv(config.candidate_output_path, sep="\t", index=False)

non_empty = int((matching_out["matched_entity_ids"].str.len() > 0).sum())
print(f"  matching_results.tsv : {len(matching_out):,} rows  (matched={non_empty:,}  singletons={len(matching_out)-non_empty:,})")
print(f"  candidate_pairs.tsv  : {len(candidate_out):,} rows")
print(f"  Output: {config.output_dir}")

print("\n--- First 10 rows of matching_results.tsv ---")
print(matching_out.head(10).to_string(index=False))

print("\n--- First 5 rows of candidate_pairs.tsv (non-empty) ---")
has_cands = candidate_out[candidate_out["candidate_entity_ids"].str.len() > 0]
print(has_cands.head(5).to_string(index=False) if not has_cands.empty else "(none found in this slice)")

print("\n" + "=" * 65)
print("PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 65)
