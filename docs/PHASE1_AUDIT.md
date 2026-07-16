# Phase 1 — TileVision AI Implementation Audit

**Repository:** `tilevision-ai`  
**Audit date:** July 2026  
**Model:** `facebook/dinov2-large` (1024D)  
**Feature versions:** `feature_version=5`, `pattern_feature_version=3`

This document records verified bottlenecks and accuracy problems found in the
repository before and during the production pipeline overhaul. All findings were
confirmed by reading source code and running tests — not assumed.

---

## Indexing Pipeline (Verified)

```
scan folder
  → validate image
  → SHA256/dhash (incremental skip if unchanged)
  → ImagePreprocessor (load, EXIF, RGB, border trim, letterbox 518)
  → FeatureExtractor.extract_batch() [batch size 8]
      → DINOv2Embedder: 3 views → ONE batched forward pass → fused 1024D embedding
      → ColorDescriptor (LAB), TextureDescriptor (multi-scale LBP)
      → EdgeDescriptor (36-bin orientation), PatternDescriptor (12-dim)
  → SQLite persist (TileFeatures blobs + metadata + version fields)
  → FaissIndexManager.update_vectors() [checkpoint every 25 files]
```

## Search Pipeline (Verified)

```
query image
  → validate + SHA256/dhash
  → reuse cached TileFeatures if query is already indexed (same SHA256)
  → else FeatureExtractor.extract() once
  → PatternClassifier.classify(query)
  → FAISS IndexFlatIP search (L2-normalized cosine)
  → SQLite batch-fetch candidate TileFeatures
  → CandidateFilter (LAB dominant-color gate)
  → HybridReRanker (DINOv2 + descriptors + pattern compatibility)
  → calibrate_display_percent() for UI
```

---

## Bottlenecks Found (Original State)

| Issue | Severity | Status |
|-------|----------|--------|
| DINOv2 inference run twice per image during indexing | Critical | **Fixed** — single `extract()` path |
| Image loaded/preprocessed multiple times per operation | High | **Fixed** — `PreprocessedImage` shared |
| FAISS index saved after every single file | High | **Fixed** — checkpoint every 25 files |
| Candidate descriptors recomputed during search | Critical | **Fixed** — SQLite-cached features only |
| Model reloaded per request | High | **Fixed** — singleton warmup in `app.py` |
| No batch DINOv2 during folder scan | High | **Fixed** — batch size 8, all views in one pass |
| Debug prints in hot paths | Medium | **Fixed** |
| No pipeline timing logs | Medium | **Fixed** — `PipelineTimer` |

## Remaining Performance Bottleneck

| Issue | Severity | Status |
|-------|----------|--------|
| DINOv2-large on CPU (~8–10 s/image) | High | **Hardware** — architecture is correct; GPU needed for speed |

---

## Accuracy Problems Found

| Problem | Root cause | Status |
|---------|------------|--------|
| Cream marble classified as speckled | Rule-based classifier too aggressive on fine noise | **Fixed** — soft scoring + vein/structure dims |
| White speckled tile ranked below cream marble | Speckled reranker over-weighted pattern vs embedding | **Fixed** — 70% embedding weight for speckled |
| Room photos with furniture score low | Scene clutter, no catalog match | **Partial** — manual crop UI; no auto-segmentation |
| Filters show only "Unknown" | Filenames not structured; old metadata | **Fixed** — improved parsing; re-index required |
| Filters had no effect on results | FAISS top-K only, not full filtered rerank | **Fixed** — DB pre-filter + full rerank ≤2000 |
| Settings Rebuild FAISS Index hidden | Maintenance section commented out | **Fixed** |
| Stale features silently compared | No version checks | **Fixed** — `feature_versions.py` + startup warning |

---

## FAISS Verification

- Index type: `IndexIDMap(IndexFlatIP)` — correct for cosine on L2-normalized vectors
- Embeddings L2-normalized before insert (`embedder.py`)
- `update_vectors()` removes stale duplicates before re-add
- ID mapping: SQLite primary keys used as FAISS IDs

**Gap (Phase 7):** Unfiltered `search_k` used floor 100 with no 200 cap — addressed in Phase 7 fix.

---

## SQLite Verification

- All descriptor blobs stored at index time
- `get_by_ids()` batch-fetches for search reranking
- Incremental indexing via SHA256 comparison
- Migrations in `db_context.py` are backward-compatible
- Transactions used for batch writes

---

## Thread Safety (Phase 15 — Fixed)

- Indexing runs on `QThread` (`indexing_worker.py`)
- Search runs on `QThread` (`search_worker.py`)
- Folder monitor can trigger indexing during search
- **Fixed** — `inference_guard.py` RLock serializes DINOv2 forward passes and FAISS load/mutate/clear.

---

## Files Inspected

| Area | Path |
|------|------|
| DINOv2 embedder | `src/ai/embedder.py` |
| Feature extraction | `src/ai/feature_extractor.py` |
| Preprocessing | `src/ai/preprocess/image_preprocessor.py` |
| TileFeatures model | `src/ai/models.py` |
| Descriptors | `src/ai/descriptors/*.py` |
| Pattern classifier | `src/ai/pattern_classifier.py` |
| Candidate filter | `src/ai/candidate_filter.py` |
| Hybrid reranker | `src/ai/reranker.py` |
| Search use case | `src/core/use_cases/search_tiles.py` |
| Index use case | `src/core/use_cases/index_images.py` |
| FAISS index | `src/ai/vector_index.py` |
| SQLite repo | `src/data/sqlite_repository.py` |
| DB migrations | `src/data/db_context.py` |
| Workers | `src/presentation/workers/*.py` |
| Composition root | `src/app.py` |
| Feature versioning | `src/ai/feature_versions.py` |

---

## Prioritized Fix Order (Completed / Remaining)

1. ✅ Eliminate duplicate DINOv2 inference
2. ✅ Unified preprocessing pipeline
3. ✅ Batched 3-view multi-scale embeddings
4. ✅ Hybrid reranker with dynamic weights
5. ✅ Feature versioning + stale detection
6. ✅ Fast search (zero candidate inference)
7. ✅ Search filters + metadata parsing
8. ✅ FAISS pool sizing 50–200 (Phase 7)
9. ✅ Soft color filter (Phase 8)
10. ✅ Stale index UI banner + Settings status + search block (Phase 14)
11. ✅ Thread locks on DINOv2 + FAISS load/mutate (Phase 15)
12. ✅ Content-region crop + lighting normalize + query scene focus (Phase 3)
13. ✅ Full accuracy evaluation manifest + 90% PASS target (Phase 16)
