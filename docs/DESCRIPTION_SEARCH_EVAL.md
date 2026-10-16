# Description search eval (PR C Item 4)

## Changes

1. **Prompt-template ensembling** — average of raw text + tile-oriented templates.
2. **Relevance cutoff** — absolute CLIP cosine floor (`0.24`) and margin from top (`0.10`); always keep ≥3 results (low-confidence banner when falling back).
3. Log / status use the description text (`Search for 'gray'…`, `Showing N of M tiles that match 'gray'`).
4. CLIP display calibration anchors **unchanged**.

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
CLIP index. Thresholds (`CLIP_ABS_FLOOR`, `CLIP_TOP_MARGIN`) were chosen from
CLIP literature / log shape (25/38 unfiltered hits) **before** filling any
precision cells above — do not retune on the same reported words without
saying so.

## Full-frame CLIP vectors

Whether panel crops / room shots weigh the same as clean tile photos in the
CLIP index is **UNVERIFIED**; not changed in this PR.
