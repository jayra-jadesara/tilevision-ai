# Indexing speed investigation (PR A Item 2 — report only)

**Default accuracy is unchanged.** No Fast mode setting ships in this PR.
Numbers below mix **measured** evidence from the user’s Windows log
(`bec564b7-13tilevision.log`, app 1.2.41, 4-thread Intel CPU, no GPU,
38-tile catalogue) with **code-derived** costs. Anything not measured on
this machine is marked **UNVERIFIED**.

## Baseline (measured from user log)

| Metric | Value |
| --- | --- |
| Steady DINOv2 cost | ~11–11.5 s/image (batch of 12 ≈ 132–135 s embed) |
| CLIP cost | ~0.2 s/image (2.2–2.8 s / batch of 12; first batch includes model load) |
| Views per tile | ~7 DINOv2 forwards (multi-scale primary + aux), 518×518 (`TARGET_SIZE`) |
| Tokens / view | 37×37 = 1369 |
| ~s / view | ≈ 1.5 s on this CPU |
| 38-tile full rebuild | ~11–14 min wall |
| 1000-tile projection | ≈ 3+ hours at ~11.5 s/image |

Aux vectors matter for retrieval (`Parent sheet Top-1 via FAISS aux` in logs)
and must not be deleted without a measured quality trade-off.

## Variant analysis

Threshold used for “keep retrieval quality”: **no drop in top-1 on the
38-tile crop-query suite, and ≤ 1 pp drop in top-5** (stated for future
measurement; not applied as a ship gate here).

| Variant | Expected speed-up (CPU) | Cosine vs current | Retrieval quality (38-tile) | Status |
| --- | --- | --- | --- | --- |
| 1. Dynamic int8 quantization of DINOv2 Linear layers | ~1.3–2× wall (torch dynamo / ONNX Runtime typical) | UNVERIFIED | UNVERIFIED | Not shipped. Needs A/B on real catalogue. |
| 2. Lower index/query size (392 / 336) — **must match** | ~(518/S)² token reduction → ~1.7× / ~2.4× theoretical FLOPs | UNVERIFIED (vectors incompatible with 518 index) | UNVERIFIED | Would require full rebuild; query+index must both change. |
| 3. Skip aux when `cos_vs_primary > 0.95` (already computed) | Up to ~30–40% fewer forwards when aux is near-dup | N/A for skipped | UNVERIFIED — aux wins matter on marketing sheets | Code already gates near-dups; raising the threshold further is UNVERIFIED. |
| 4. Fewer multi-scale views per image | Linear in view count (e.g. 7→3 ≈ ~2×) | UNVERIFIED | UNVERIFIED | Highest risk to crop/partial-tile recall. |

### Notes

- Variant 2 changes the embedding space: an existing 518 index cannot be
  queried with 392/336 vectors. Opt-in Fast mode would need a separate
  index or a guided rebuild.
- Variant 1 (int8) preserves dimension and can be transparent if cosine
  drift is tiny — **UNVERIFIED on this catalogue**.
- This environment did not run a full 38-tile DINOv2 A/B in CI (too slow /
  weights may be absent). Treat the table as a plan, not a ship decision.

## Recommendation

Keep **Accurate (default)** unchanged. If a future PR exposes
“Indexing speed: Accurate / Fast”, gate Fast behind measured top-1/top-5
on the real 38-tile crop suite plus a customer acceptance note that Fast
requires rebuild. Do not change views or `TARGET_SIZE` in the default path.
