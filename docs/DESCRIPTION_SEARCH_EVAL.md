# Description search eval

## Changes (cutoff tune)

1. **Prompt-template ensembling** — average of raw text + tile-oriented templates,
   embedded in **one batched** `get_text_features` call (not four sequential).
2. **Distribution-aware relevance cutoff** — absolute CLIP cosine floor (`0.24`)
   plus adaptive margin `max(0.02, 0.35 × (top − median))` instead of a fixed
   `0.10` top-margin. Always keep ≥3 results (low-confidence banner when falling
   back).
3. **Score logging** — each description search emits an INFO line with query,
   template count, raw top-10 cosines, min/max/median/spread, and counts above
   floor / within margin so constants can be retuned from real logs.
4. CLIP display calibration anchors **unchanged**.

### Why adaptive margin

Real CLIP image–text cosines on a small catalogue cluster in a narrow band
(~0.18–0.30). A fixed `0.10` margin is often larger than the whole catalogue
spread, so `matched_pre_cutoff ≈ catalogue` (observed: every query reported
38/38 pre-cutoff). Scaling the margin to a fraction of `(top − median)` tracks
that band; the `0.02` floor avoids over-pruning when top ≈ median.

## Objective colour-word ground truth

`dev_tools/eval_description_colour_words.py` labels tiles from HSV statistics
(not opinions). Run on the real 38-tile catalogue:

```bash
python3 dev_tools/eval_description_colour_words.py --catalogue /path/to/Tiles
```

### Precision table (real 38-tile catalogue + CLIP)

| Query | P@5 before | P@5 after | P@10 before | P@10 after |
| --- | --- | --- | --- | --- |
| gray | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED |
| white | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED |
| black | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED |
| beige | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED |
| brown | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED |

This CI environment does not have the customer’s 38-tile files or a warmed
CLIP index. Thresholds were chosen from the logged score-band shape (narrow
~0.18–0.30, fixed 0.10 margin keeping ~all) and synthetic unit tests that
mimic that spread — **not** from a live precision@5 table on the colour words
above. Do not retune on the same reported words without saying so.

## Full-frame CLIP vectors

Whether panel crops / room shots weigh the same as clean tile photos in the
CLIP index is **UNVERIFIED**; not changed in this PR.
