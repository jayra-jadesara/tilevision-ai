# TileVision AI — ORB Local-Feature Verification

**Scope:** query-time rerank only. Index / FAISS / `TileFeatures` unchanged.  
**feature_version:** still **10** (no Rebuild Search Index required).

## Root cause

Global descriptors (DINOv2, color / texture / edge histograms, pattern
features) are insufficient to separate visually similar white/marble tiles.
Two distinct products can share near-identical global stats with no local
geometric verification step in the pipeline.

## Architecture

```
query image
  → … existing pipeline through HybridReRanker.score() …
  → sort by hybrid score
  → ORB verification (optional, default ON)
       only candidates within ORB_VERIFICATION_BAND (0.03) of #1
       capped at ORB_MAX_CANDIDATES (5)
       final = hybrid + ORB_BOOST_MAX(0.05) * orb_inlier_score
  → weak-result filter → top_k
```

ORB runs CPU-only via OpenCV (`cv2.ORB_create` + BFMatcher + RANSAC
homography). Failures degrade to `0.0` and keep the hybrid-only ranking.
Toggle with `enable_orb_verification` in `config.json` (default `true`).

## Benchmark

```bash
# Baseline (ORB off)
python3 dev_tools/search_quality/run_bakeoff.py \
  --out /opt/cursor/artifacts/orb_off \
  --orb-verification off

# With ORB near-tie verification
python3 dev_tools/search_quality/run_bakeoff.py \
  --out /opt/cursor/artifacts/orb_on \
  --orb-verification on
```

Latency reports include `rerank_ms` (ORB stage wall time per query).
Vectors/tile is N/A — no index change.

### Measured

| Metric | ORB off | ORB on | Δ |
|--------|---------|--------|---|
| Recall@1 | | | |
| Recall@5 | | | |
| MRR | | | |
| rerank_ms | | | |
| total_s | | | |

*(Fill after running the bakeoff above — do not fabricate numbers.)*

## Platforms

Pure OpenCV / NumPy CPU code — identical on Windows, macOS Intel, and Apple
Silicon. No platform-specific branches. Validate via CI search gates; do not
fabricate per-OS scores.
