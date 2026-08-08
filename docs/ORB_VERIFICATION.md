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
Toggle with `enable_orb_verification` in `config.json` (default **on** as of
v1.2.34 — see real-customer gate below).

## Real-customer gate (v1.2.34 release decision)

Production `SearchTilesUseCase` A/B on the committed release eval set
(`eval/real_customer_release.jsonl`, 70 queries, 4 confusable white/marble
pairs, crops / WhatsApp / phone / perspective variants):

| Metric | ORB off | ORB on | Δ |
|--------|---------|--------|---|
| Recall@1 (overall) | 0.5143 | **0.6286** | +0.1143 |
| Recall@5 (overall) | 0.8714 | 0.8714 | 0.0 |
| MRR (overall) | 0.6607 | **0.7286** | +0.0679 |
| Recall@1 (confusable pairs) | 0.5179 | **0.6607** | +0.1429 |

ORB materially improves top-1 on confusable marble pairs and customer-like
captures without hurting Recall@5. **Default flipped to ON** in
`AppSettings` for new installs; existing `config.json` files keep their
saved value.

Reproduce:

```bash
python3 dev_tools/search_quality/build_real_customer_eval_set.py
python3 dev_tools/search_quality/run_real_customer_orb_gate.py \
  --manifest eval/real_customer_release.jsonl \
  --out /tmp/real_customer_orb_gate
```

## Synthetic bakeoff (reference only)

```bash
python3 dev_tools/search_quality/run_bakeoff.py \
  --out /tmp/orb_off --orb-verification off

python3 dev_tools/search_quality/run_bakeoff.py \
  --out /tmp/orb_on --orb-verification on
```

Do **not** use golden synthetic numbers alone to decide ORB — the
real-customer gate above is authoritative for the production default.

## Platforms

Pure OpenCV / NumPy CPU code — identical on Windows, macOS Intel, and Apple
Silicon. No platform-specific branches.
