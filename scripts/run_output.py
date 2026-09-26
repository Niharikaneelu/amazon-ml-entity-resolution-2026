"""
Optimized pipeline: pre-filter target key DFs once, then fast merges.
Trains on 50k S1 sample with FULL S2/S3, runs test inference.
"""
import sys, time, os
from pathlib import Path
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "business_entity_resolution"))

import pandas as pd
from src.config import PipelineConfig
from src.load_data import read_tsv
from src.blocking import _exact_name_keydf, _country_norm_name_keydf
from src.candidates import (
    enrich_candidate_pairs, attach_ground_truth_labels,
    format_candidates_for_submission, evaluate_candidate_recall,
    union_candidate_sets,
)
from src.train import train_model
from src.predict import predict_pairs, format_matching_results_for_submission
from src.evaluate import compute_macro_f05

config = PipelineConfig()
wall   = time.time()
MAX_BLOCK  = 5000
TRAIN_S1_N = 50_000
TEST_S1_N  = 50_000

print("=" * 65)
print("FINAL PIPELINE - Pre-filtered keys, full S2/S3")
print("=" * 65)

# ── [1] Load full S2 and S3 (one-time) ───────────────────────────────────────
print("\n[1] Loading full train S2 and S3...", flush=True)
t0 = time.time()
s2 = read_tsv(config.train_s2_path)
s3 = read_tsv(config.train_s3_path)
print(f"  S2:{len(s2):,}  S3:{len(s3):,}  ({time.time()-t0:.1f}s)", flush=True)

# ── [2] Pre-build AND pre-filter target key DFs (one-time cost) ───────────────
print("\n[2] Pre-building + pre-filtering target key DFs (one-time)...", flush=True)
t0 = time.time()

def build_and_filter(df, builder, max_block):
    kdf    = builder(df)
    bsizes = kdf.groupby("blocking_key").size()
    valid  = bsizes[bsizes <= max_block].index
    return kdf[kdf["blocking_key"].isin(valid)].reset_index(drop=True)

s2_exact_filt   = build_and_filter(s2, _exact_name_keydf,   MAX_BLOCK)
s2_country_filt = build_and_filter(s2, _country_norm_name_keydf, MAX_BLOCK)
s3_exact_filt   = build_and_filter(s3, _exact_name_keydf,   MAX_BLOCK)
s3_country_filt = build_and_filter(s3, _country_norm_name_keydf, MAX_BLOCK)
print(f"  Done in {time.time()-t0:.1f}s", flush=True)
print(f"  S2 exact:{len(s2_exact_filt):,}  S2 country:{len(s2_country_filt):,}", flush=True)
print(f"  S3 exact:{len(s3_exact_filt):,}  S3 country:{len(s3_country_filt):,}", flush=True)

def fast_block(s1_df, tgt_filtered_kdf, source_label, builder):
    """Fast merge: target already filtered, just build S1 keys and merge."""
    s1_kdf = builder(s1_df)
    if s1_kdf.empty or tgt_filtered_kdf.empty:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "candidate_source"])
    merged = s1_kdf.merge(
        tgt_filtered_kdf.rename(columns={"entity_id": "candidate_entity_id"}),
        on="blocking_key", how="inner"
    )
    merged = merged.rename(columns={"entity_id": "source1_entity_id"})
    merged["candidate_source"] = source_label
    return merged[["source1_entity_id", "candidate_entity_id", "candidate_source"]]

def get_candidates(s1_df, s2_ef, s2_cf, s3_ef, s3_cf):
    parts = [
        fast_block(s1_df, s2_ef, "source2", _exact_name_keydf),
        fast_block(s1_df, s2_cf, "source2", _country_norm_name_keydf),
        fast_block(s1_df, s3_ef, "source3", _exact_name_keydf),
        fast_block(s1_df, s3_cf, "source3", _country_norm_name_keydf),
    ]
    combined = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    return combined.drop_duplicates(subset=["source1_entity_id", "candidate_entity_id"])

# ── [3] Train S1 blocking ─────────────────────────────────────────────────────
print("\n[3] Loading train S1 + blocking...", flush=True)
t0 = time.time()
train_s1 = read_tsv(config.train_s1_path, nrows=TRAIN_S1_N)
train_gt = read_tsv(config.train_gt_path)
train_gt = train_gt[train_gt["source1_entity_id"].isin(set(train_s1["entity_id"]))]
print(f"  Train S1:{len(train_s1):,}  GT:{len(train_gt):,}", flush=True)

train_cands = get_candidates(train_s1, s2_exact_filt, s2_country_filt, s3_exact_filt, s3_country_filt)
print(f"  Train candidates: {len(train_cands):,}  ({time.time()-t0:.1f}s)", flush=True)

recall = evaluate_candidate_recall(train_cands, train_gt)
print(f"  Recall  S2:{recall['candidate_recall_s2']:.4f}  S3:{recall['candidate_recall_s3']:.4f}  Overall:{recall['candidate_recall_overall']:.4f}", flush=True)
print(f"  Found   S2:{recall['true_pairs_s2_found']}/{recall['true_pairs_s2_total']}  S3:{recall['true_pairs_s3_found']}/{recall['true_pairs_s3_total']}", flush=True)

# ── [4] Feature Engineering + Train ──────────────────────────────────────────
print("\n[4] Feature Engineering & LightGBM Training...", flush=True)
t0 = time.time()
enriched = enrich_candidate_pairs(train_cands, train_s1, s2, s3)
labeled  = attach_ground_truth_labels(enriched, train_gt)
pos = int((labeled["label"] == 1).sum())
neg = int((labeled["label"] == 0).sum())
print(f"  Pairs:{len(labeled):,}  pos={pos:,}  neg={neg:,}", flush=True)

bundle = train_model(labeled, label_column="label", model_path=config.model_path)
print(f"  Threshold={bundle['best_threshold']:.2f}  Train F0.5={bundle['best_f05']:.4f}  ({time.time()-t0:.1f}s)", flush=True)

# ── [5] Validation Macro F0.5 ─────────────────────────────────────────────────
print("\n[5] Validation Macro F0.5...", flush=True)
val_preds = predict_pairs(enriched, bundle)
val_df    = format_matching_results_for_submission(val_preds, train_s1["entity_id"])
m         = compute_macro_f05(val_df, train_gt)
print(f"  +--------------------------------------+")
print(f"  |  Macro F0.5      = {m['macro_f05']:.4f}            |")
print(f"  |  Macro Precision = {m['macro_precision']:.4f}            |")
print(f"  |  Macro Recall    = {m['macro_recall']:.4f}            |")
print(f"  |  Singleton Acc   = {m['singleton_accuracy']:.4f}            |")
print(f"  +--------------------------------------+")
print(f"  Entities evaluated: {m['total_eval_entities']:,}", flush=True)

# ── [6] Test Inference ────────────────────────────────────────────────────────
print("\n[6] Test inference on 50k test S1...", flush=True)
t0 = time.time()
test_s1 = read_tsv(config.test_s1_path, nrows=TEST_S1_N)
print(f"  Test S1: {len(test_s1):,}", flush=True)

# REUSE the pre-filtered train S2/S3 key DFs — same product catalogs, no reload
print("  Reusing pre-filtered S2/S3 key DFs (no rebuild needed)...", flush=True)
test_cands    = get_candidates(test_s1, s2_exact_filt, s2_country_filt, s3_exact_filt, s3_country_filt)
enriched_test = enrich_candidate_pairs(test_cands, test_s1, s2, s3)
test_preds    = predict_pairs(enriched_test, bundle)
print(f"  Test candidates:{len(test_cands):,}  ({time.time()-t0:.1f}s)", flush=True)


# ── [7] Write submission files ────────────────────────────────────────────────
print("\n[7] Writing submission files...", flush=True)
config.output_dir.mkdir(parents=True, exist_ok=True)
matching_out  = format_matching_results_for_submission(test_preds, test_s1["entity_id"])
candidate_out = format_candidates_for_submission(test_cands, test_s1["entity_id"])

matching_out.to_csv(config.matching_output_path,   sep="\t", index=False)
candidate_out.to_csv(config.candidate_output_path, sep="\t", index=False)

non_empty = int((matching_out["matched_entity_ids"].str.len() > 0).sum())
print(f"  matching_results.tsv: {len(matching_out):,} rows  matched={non_empty:,}  singletons={len(matching_out)-non_empty:,}", flush=True)
print(f"  candidate_pairs.tsv:  {len(candidate_out):,} rows", flush=True)

print("\n  --- First 10 rows with matches ---")
sample = matching_out[matching_out["matched_entity_ids"].str.len() > 0].head(10)
for _, r in sample.iterrows():
    print(f"  {r['source1_entity_id']}  ->  {r['matched_entity_ids'][:80]}")

total = time.time() - wall
print(f"\nTOTAL WALL TIME: {total:.0f}s ({total/60:.1f} min)")
print("=" * 65)
print("DONE! Output files written.")
print("=" * 65)
