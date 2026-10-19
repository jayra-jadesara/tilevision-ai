# Indexing speed investigation

**Default accuracy is unchanged.** No Fast mode setting ships until a
variant clears the ship gate on the customer laptop. Numbers below mix
**measured** evidence from the user’s Windows logs with **code-derived**
costs. Anything not measured on that machine is marked **UNVERIFIED**.

## Baseline (measured — updated from 2026-10 rebuild log)

Full rebuild 14:31:59 → 14:38:14: **6 min 15 s for 38 files**
(`reused embeddings for 15 identical files` → **23 distinct images**).
Reuse is complete (23 forwards for 23 distinct); nothing left to dedupe
across identical copies.

DINOv2 time by pass (from `DINOv2 index batch:` lines):

| Pass | images | views | seconds |
| --- | --- | --- | --- |
| primary (4 batches) | 4+10+5+4 = 23 | 69 | 165 |
| aux crops (4 batches) | 7+9+8+6 = 30 | 90 | 167 |
| total DINOv2 | | 159 | **332 of 375 s (88 %)** |

Steady cost ≈ **12.7 s per distinct image** (≈ 1.65 s/view × ~7.5 views).
Customers without copies get no benefit from dedupe: 1000 distinct images
≈ 3.5 h on this laptop.

Earlier log (app 1.2.41, same machine): ~11–11.5 s/image, 38-tile rebuild
~11–14 min — consistent order of magnitude once reuse is accounted for.

Aux vectors matter for retrieval (`Parent sheet Top-1 via FAISS aux` in
logs) and must not be deleted without a measured quality trade-off.

## B1 — Query warm-up vs first index batch (fixed in code)

**Measured:** startup DINOv2 warm-up 14:31:54–14:32:51 (60.9 s) overlapped
rebuild start 14:31:59. First batch ~26.4 s/file vs ~12 s steady
(forward 49.4 s for 12 views = 4.1 s/view vs 1.6–1.7 s/view later). On
2026-10-14 the same warm-up took **301 s** while CLIP downloaded.

**Fix shipped:** `indexing_busy_scope` on `IndexingWorker` /
`RebuildIndexWorker`; `run_query_path_warmup` skips/aborts when indexing
is active (index forwards already warm the model).

**First-batch s/view before vs after on the customer laptop:**
**UNVERIFIED here** (no customer catalogue / DINOv2 weights in this CI
image). Re-measure after installing this build.

## B2 — Aux pass waste (logging shipped; prefilter opt-in)

Log: 30 aux crops embedded, only 18 `Index aux … vector` lines →
**12 crops (~67 s, ~20 % of all DINOv2 time)** computed and discarded by
`cos_vs_primary ≥ 0.985` / pair ≥ 0.99.

**Shipped:**
1. Every dropped aux crop is logged (`Index aux {label} dropped … cos=…`).
2. Cheap pixel prefilter (`src/ai/aux_crop_prefilter.py`) — thumbnail
   cosine + area fraction — available behind
   ``TILEVISION_AUX_PREFILTER=1`` only.
3. **Not enabled by default.** Validation on the 23 distinct images
   (false-skips = 0 vs the embedding drop rule) is **UNVERIFIED** in this
   environment (no catalogue weights). Seconds saved per 23-image run:
   **UNVERIFIED** until the filter is validated and turned on.

## B3 — Bench harness (customer laptop)

```text
python dev_tools/bench_indexing_speed.py --catalogue <folder> --variants threads,int8
```

Loads weights from `model_weights/`, works on Windows paths, prints
progress, writes CSV (`ms/view`, `ms/image`, cosine vs fp32, top-1/top-5
proxy). Variants: torch threads 2/3/4, dynamic int8 Linear, aux-1-view /
2-view stubs.

### Ship gate for opt-in Fast

A variant may ship as **opt-in Fast** only if, on the distinct-image
catalogue:

- top-1 identical vs Accurate fp32
- top-5 within 1 pp
- mean cosine ≥ 0.99

**Default Accurate must not change.** No `CURRENT_FEATURE_VERSION` bump,
no multi-process embedding.

### Variant table (status)

| Variant | Expected speed-up (CPU) | Cosine vs current | Retrieval | Status |
| --- | --- | --- | --- | --- |
| Torch threads 2 / 3 / 4 | HT often hurts GEMM on 2P/4L — **UNVERIFIED** | N/A (same fp32) | N/A | Measure with harness; do not change default threads |
| Dynamic int8 DINOv2 Linear | ~1.3–2× typical; Skylake-U has AVX2 **no VNNI** so gain may be small | **UNVERIFIED** | **UNVERIFIED** | Harness only; not shipped |
| Aux crops 1 view (not 3) | ~aux DINOv2 ÷ 3 | **UNVERIFIED** | **UNVERIFIED** | Harness stub; not shipped |
| 3 → 2 index views | Linear in view count | **UNVERIFIED** | **UNVERIFIED** | Harness stub; not shipped |
| Aux pixel prefilter | Up to ~20 % DINOv2 if false-skips=0 | N/A for skipped | **UNVERIFIED** on 23 images | Code ready, **default OFF** |
| Lower TARGET_SIZE 392/336 | Theoretical FLOPs | Incompatible with 518 index | Would need rebuild | Not shipped |

## Multi-process embedding (measured sandbox — do not ship)

| Finding | Value |
| --- | --- |
| Ceiling | Caps at physical core count; torch already uses intra-op threads |
| Extra process cost | Another DINOv2-large ~1.2 GB RAM per worker |
| Decision | **Do not ship** multi-process embedding |

## Byte-identical file reuse (already on main via #99)

Cross-batch `ContentEmbedCache` reuses embeddings for identical sha256
across flush batches. Log: `reused embeddings for N identical files across batches`.

## Recommendation

Keep **Accurate (default)** unchanged. Use the bench harness on the
customer laptop to fill the UNVERIFIED cells. Only then expose
“Indexing speed: Accurate / Fast” as an opt-in setting.
