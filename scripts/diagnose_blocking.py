"""Diagnose blocking key distribution to find what's slow."""
import sys, time, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "business_entity_resolution"))
os.environ["PYTHONIOENCODING"] = "utf-8"

from src.config import PipelineConfig
from src.load_data import read_tsv
from src.blocking import _exact_name_keydf, _country_norm_name_keydf, _norm

config = PipelineConfig()

# Load only S2 to diagnose
print("Loading S2 (full)...", flush=True)
t0 = time.time()
s2 = read_tsv(config.train_s2_path)
print(f"  {len(s2):,} rows in {time.time()-t0:.1f}s", flush=True)

# Strategy 1: exact_normalized_name
print("\n[1] exact_normalized_name key distribution...", flush=True)
t0 = time.time()
kdf1 = _exact_name_keydf(s2)
sizes1 = kdf1.groupby("blocking_key").size()
print(f"  Time: {time.time()-t0:.2f}s", flush=True)
print(f"  Unique keys: {len(sizes1):,}", flush=True)
print(f"  Max block size: {sizes1.max():,}", flush=True)
print(f"  Keys with >5000 entries: {(sizes1>5000).sum():,}", flush=True)
print(f"  Keys with >1000 entries: {(sizes1>1000).sum():,}", flush=True)
print(f"  Keys with 1 entry: {(sizes1==1).sum():,}", flush=True)
print(f"  Median block size: {sizes1.median():.1f}", flush=True)

# Strategy 2: country_normalized_name  
print("\n[2] country_normalized_name key distribution...", flush=True)
t0 = time.time()
kdf2 = _country_norm_name_keydf(s2)
sizes2 = kdf2.groupby("blocking_key").size()
print(f"  Time: {time.time()-t0:.2f}s", flush=True)
print(f"  Unique keys: {len(sizes2):,}", flush=True)
print(f"  Max block size: {sizes2.max():,}", flush=True)
print(f"  Keys with >5000 entries: {(sizes2>5000).sum():,}", flush=True)
print(f"  Keys with >1000 entries: {(sizes2>1000).sum():,}", flush=True)
print(f"  Keys with 1 entry: {(sizes2==1).sum():,}", flush=True)
print(f"  Median block size: {sizes2.median():.1f}", flush=True)

# Now load S1 (small) and time the full merge
print("\n[3] Timing merge: 1000 S1 entities vs full S2 (exact_name)...", flush=True)
s1 = read_tsv(config.train_s1_path, nrows=1000)
kdf_s1 = _exact_name_keydf(s1)

# Filter target keys to max_block_size=5000
valid_keys = sizes1[sizes1 <= 5000].index
kdf1_filtered = kdf1[kdf1["blocking_key"].isin(valid_keys)]

t0 = time.time()
merged = kdf_s1.merge(
    kdf1_filtered.rename(columns={"entity_id": "candidate_entity_id"}),
    on="blocking_key", how="inner"
)
print(f"  Merge done: {len(merged):,} pairs in {time.time()-t0:.2f}s", flush=True)

print("\nDone.", flush=True)
