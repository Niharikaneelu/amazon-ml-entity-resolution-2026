"""Benchmark blocking candidate generation."""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "business_entity_resolution"))

from src.config import PipelineConfig
from src.load_data import read_tsv
from src.candidates import generate_all_candidates, evaluate_candidate_recall

config = PipelineConfig()

print("Loading data...")
train_s1 = read_tsv(config.train_s1_path, nrows=5000)
train_s2 = read_tsv(config.train_s2_path, nrows=100000)
train_s3 = read_tsv(config.train_s3_path, nrows=100000)
train_gt = read_tsv(config.train_gt_path, nrows=5000)

train_s1_sample = train_s1.sample(n=min(5000, len(train_s1)), random_state=42).reset_index(drop=True)
train_gt_sample = train_gt[train_gt["source1_entity_id"].isin(set(train_s1_sample["entity_id"]))].reset_index(drop=True)
print(f"  Loaded. S1={len(train_s1_sample)} GT={len(train_gt_sample)}")

print("Running candidate generation...")
t0 = time.time()
train_cands = generate_all_candidates(train_s1_sample, train_s2, train_s3)
elapsed = time.time() - t0
print(f"  Done in {elapsed:.1f}s. Candidates: {len(train_cands)}")

recall = evaluate_candidate_recall(train_cands, train_gt_sample)
print(f"  Recall S2={recall['candidate_recall_s2']:.4f}  S3={recall['candidate_recall_s3']:.4f}  Overall={recall['candidate_recall_overall']:.4f}")
print(f"  True pairs S2 found: {recall['true_pairs_s2_found']} / {recall['true_pairs_s2_total']}")
print(f"  True pairs S3 found: {recall['true_pairs_s3_found']} / {recall['true_pairs_s3_total']}")
print(f"  Avg candidates per S1: {recall['avg_candidates_per_s1']:.1f}")
