"""Minimal timed test: blocking merge 50k S1 vs full S2 (no isin filter)."""
import sys, time, os
from pathlib import Path
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "business_entity_resolution"))

from src.config import PipelineConfig
from src.load_data import read_tsv
from src.blocking import _exact_name_keydf, _country_norm_name_keydf
import pandas as pd

config = PipelineConfig()

print("Loading S2 (full)...", flush=True)
t0 = time.time()
s2 = read_tsv(config.train_s2_path)
print(f"  {len(s2):,} rows in {time.time()-t0:.1f}s", flush=True)

print("Building S2 exact key df...", flush=True)
t0 = time.time()
s2_kdf = _exact_name_keydf(s2)
print(f"  {len(s2_kdf):,} key rows in {time.time()-t0:.1f}s", flush=True)

# Block size filter BEFORE merge (keep only keys with <= 50 matches for precision)
print("Block size filtering (<=50 per key)...", flush=True)
t0 = time.time()
bsizes = s2_kdf.groupby("blocking_key").size()
valid  = bsizes[bsizes <= 50].index   # tight filter = high precision
s2_kdf_filt = s2_kdf[s2_kdf["blocking_key"].isin(valid)]
print(f"  Filtered: {len(s2_kdf_filt):,} key rows (from {len(s2_kdf):,}) in {time.time()-t0:.1f}s", flush=True)

print("Loading 50k S1...", flush=True)
t0 = time.time()
s1 = read_tsv(config.train_s1_path, nrows=50000)
s1_kdf = _exact_name_keydf(s1)
print(f"  S1 key rows: {len(s1_kdf):,} in {time.time()-t0:.1f}s", flush=True)

print("Merging S1 vs S2 keys...", flush=True)
t0 = time.time()
merged = s1_kdf.merge(
    s2_kdf_filt.rename(columns={"entity_id": "candidate_entity_id"}),
    on="blocking_key", how="inner"
)
merged = merged.rename(columns={"entity_id": "source1_entity_id"})
print(f"  Merged: {len(merged):,} candidate pairs in {time.time()-t0:.1f}s", flush=True)
print(f"  Unique S1 with candidates: {merged['source1_entity_id'].nunique():,}", flush=True)
print("Done!", flush=True)
