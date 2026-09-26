# Business Entity Resolution Solution

High-precision Entity Resolution (ER) pipeline for Amazon ML Challenge 2026.

## Overview
This solution matches business entities across 3 independent data sources (`Source 1`, `Source 2`, `Source 3`).
It is specifically optimized for **Macro-Averaged $F_{0.5}$ score** by balancing high recall in Stage 1 candidate generation with aggressive high precision in Stage 2 matching.

## Pipeline Architecture
1. **Stage 1 (Blocking / Candidate Generation)**: Multi-strategy indexing (Exact Normalized Name, Country + Normalized Tokens, Country + Address Numbers) to achieve high candidate recall ceiling ($\ge 96\%$).
2. **Stage 2 (Feature Engineering)**: RapidFuzz fuzzy ratios (token set ratio, token sort ratio, partial ratio), Jaccard similarity, character length ratios, address house/street number matching, and country consistency flags.
3. **Stage 3 (LightGBM Matching & Thresholding)**: LightGBM GBDT matcher trained on candidate pairs. Decision threshold tuned to $\tau \approx 0.80$ to maximize macro $F_{0.5}$ and eliminate false positives on singletons.

## Repro Steps
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run end-to-end pipeline
python run_pipeline.py
```
Outputs are written to:
- `output/matching_results.tsv`
- `output/candidate_pairs.tsv`
