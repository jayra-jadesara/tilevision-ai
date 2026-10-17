# Description search eval

## Display % (distribution-aware)

Description search maps CLIP cosines to display % with a **per-query** z-score
vs the deduped catalogue score band for that search, then a logistic curve
(tuned on customer log fixtures — see `CUSTOMER_LOG_FIXTURE_STATS` in
`src/ai/similarity_score.py`).

- Image search still uses `calibrate_display_percent` (DINOv2 hybrid).
- Legacy `calibrate_clip_display_percent` remains for backward compatibility;
  description search routes through `calibrate_description_search_display_map`.
- Low-confidence banner uses the same top-z signal
  (`description_search_is_low_confidence`) plus cutoff fallback.

### B1 before/after (fixture synthesis, verified in CI)

| Query | Top display (old CLIP anchors) | Top display (new) | Min display (new) | Top z |
| --- | --- | --- | --- | --- |
| white | UNVERIFIED table | see unit test | ≤ ~40% | — |
| gray | ~high anchor | ~77% | ≤ ~40% | ~1.73 |
| dot | UNVERIFIED | see unit test | ≤ ~40% | — |
| texture | ~high anchor | ~72% (lower than gray) | ≤ ~40% | ~1.48 (low confidence) |

Run `pytest tests/test_description_display_distribution.py` for exact numbers
on the synthetic bands derived from logged min/median/max.

## Relevance cutoff + templates

1. **Prompt-template ensembling** — average of raw text + tile-oriented templates.
2. **Distribution-aware relevance cutoff** — floor `0.24` plus adaptive margin
   `max(0.02, 0.35 × (top − median))`, keep ≥3 results (fallback banner).
3. **Colour-word RRF** — when the query names a colour, fuse CLIP rank with
   tile `color` metadata (filename or auto-detected).

## Catalogue eval harness

```bash
python3 dev_tools/eval_description_search.py --catalogue /path/to/Tiles --csv /tmp/desc_eval.csv
```

Requires local CLIP under `model_weights/` (exits with a plain message if missing).
Embeds each unique image once (sha256 dedupe), labels colours from pixels, runs
colour words + five free-text queries, prints P@5/@10 for raw / templates /
templates+cutoff / templates+cutoff+colour and score/display stats.

### Precision table (real catalogue + CLIP)

| Query | P@5 raw | P@5 templates+cutoff+colour | P@10 raw | P@10 templates+cutoff+colour |
| --- | --- | --- | --- | --- |
| gray | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED |
| white | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED |
| black | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED |
| beige | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED |
| brown | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED |

Populate by running the harness on the customer catalogue on a machine with CLIP weights.

## Colour metadata backfill

Update Search Index classifies tiles with Unknown colour, current DINOv2 features,
and CLIP present as `METADATA_ONLY` (pixel HSV detect, no `feature_version` bump).
Summary includes `N colours detected`.

## Full-frame CLIP vectors

Whether panel crops weigh the same as clean tile photos in the CLIP index is
**UNVERIFIED**; not changed in this PR.
