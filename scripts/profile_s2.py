"""Profile blocking key strategies on full S2."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "business_entity_resolution"))
from src.config import PipelineConfig
from src.load_data import read_tsv
from src.blocking import _token_keydf, _exact_name_keydf, _address_keydf, _country_norm_name_keydf

config = PipelineConfig()

print("Loading full S2...")
t0 = time.time()
s2 = read_tsv(config.train_s2_path)
load_time = time.time() - t0
print(f"  {len(s2):,} rows in {load_time:.1f}s")
print(f"  Unique business_name: {s2['business_name'].nunique():,}")
print(f"  Unique country: {s2['country'].nunique():,}")
uniq_cn = s2.drop_duplicates(subset=["country", "business_name"])
print(f"  Unique (country,name): {len(uniq_cn):,} ({len(uniq_cn)/len(s2)*100:.1f}% of rows)")

# Strategy 1: exact_normalized_name (fully vectorized)
print("\n[1] exact_normalized_name...")
t0 = time.time()
kdf = _exact_name_keydf(s2)
print(f"  {time.time()-t0:.2f}s  key_rows={len(kdf):,}  unique_keys={kdf['blocking_key'].nunique():,}")

# Strategy 2: country_normalized_name (fully vectorized)
print("\n[2] country_normalized_name...")
t0 = time.time()
kdf2 = _country_norm_name_keydf(s2)
print(f"  {time.time()-t0:.2f}s  key_rows={len(kdf2):,}  unique_keys={kdf2['blocking_key'].nunique():,}")

# Strategy 3: token keys on FULL S2 rows
print("\n[3] country_name_tokens (full S2)...")
t0 = time.time()
kdf3 = _token_keydf(s2)
print(f"  {time.time()-t0:.2f}s  key_rows={len(kdf3):,}")

# Strategy 3b: token keys on DEDUPED rows
print("\n[3b] country_name_tokens (deduped by country+name)...")
t0 = time.time()
kdf3b = _token_keydf(uniq_cn.reset_index(drop=True))
print(f"  {time.time()-t0:.2f}s  key_rows={len(kdf3b):,}")

# Strategy 4: address keys on deduped
print("\n[4] country_address_number (deduped)...")
t0 = time.time()
kdf4 = _address_keydf(uniq_cn.reset_index(drop=True))
print(f"  {time.time()-t0:.2f}s  key_rows={len(kdf4):,}")

print("\nDone.")
