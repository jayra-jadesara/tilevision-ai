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

## Multi-process embedding (measured sandbox — do not ship)

**Decision: do not add worker processes for DINOv2 indexing.**

Measured in this cloud sandbox (2 vCPU, CPU-only PyTorch, ViT-sized matmul
micro-benchmark and multi-process throughput experiment):

| Finding | Value |
| --- | --- |
| ViT-sized matmul (single job) | ~200 ms/job |
| Throughput, 4 worker processes | ~9.4 jobs/s |
| Throughput, 10 worker processes | ~9.4 jobs/s (same ceiling) |
| Ceiling | ~2 jobs/s per core → caps at physical core count |
| Torch in one process | Already uses all cores via intra-op threads |
| Extra process cost | Each worker loads another DINOv2-large (~1.2 GB RAM) |
| Overlap vs DINOv2 | Preprocessing ~0.8 s vs DINOv2 ~128 s per batch of 12 images — overlapping decode saves nothing meaningful |

Parallelism across files is already handled by batched forwards inside one
process; spawning more Python processes only duplicates model weights and
contends for the same CPUs.

## Byte-identical file reuse (PR C)

When several pending FULL paths share the same `sha256_hash` in one flush,
indexing embeds one representative and copies DINOv2 (primary + aux), CLIP,
and handcrafted descriptors to the other SQLite/FAISS ids (paths and filename
metadata stay distinct). Log line: `reused embeddings for N identical files`.

## GPU indexing path (code review)

- **Device:** `DINOv2Embedder` selects CUDA or MPS via `detect_gpu_runtime()`
  when `inference_device` is `auto`; `embedder.using_gpu` drives
  `IndexingPerformanceConfig.from_settings(..., use_gpu=...)`.
- **Flush batch size:** CPU default `index_batch_size` 12; when GPU is active,
  `gpu_index_batch_size` 24 (see `tests/test_indexing_performance.py`).
- **Views per forward:** `_INDEX_VIEW_FORWARD_CHUNK` defaults to **24** for all
  devices (override: `TILEVISION_INDEX_VIEW_BATCH`). This is **not** scaled up
  on GPU in code today — **UNVERIFIED** whether a larger GPU-only chunk would
  help without OOM on customer hardware.

Variant 1 (int8) and Variant 2 (smaller `TARGET_SIZE`) remain **opt-in /
plan-only**; defaults unchanged.
