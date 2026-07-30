# Search Reliability Audit — TileVision AI

**Scope:** Drop → results pipeline only. Architecture unchanged (DINOv2, IndexFlatIP, SQLite, SAM2 Precise Crop, hybrid rerank).

## Root causes found

| # | Severity | Cause | Fix |
|---|----------|-------|-----|
| 1 | Critical | Weak-result filter could keep **0** of N FAISS hits → empty UI | Always retain top match(es) |
| 2 | Critical | FAISS IDs with missing SQLite/features → silent `[]` | Raise rebuild error |
| 3 | Critical | Pre-search FAISS lock before search priority → false “empty index” | Claim priority first; treat busy ≠ empty |
| 4 | High | Worker interrupt exited with **no signal** → hung SEARCHING | Always emit cancelled/failed |
| 5 | High | Metadata filters matching nothing → generic empty | Explicit filter error |
| 6 | High | Invalid/non-local drops ignored silently | Status + dialog + log |
| 7 | Medium | Concurrent drop discarded | Queue one pending query |
| 8 | Medium | Missing stage breadcrumbs | `[SEARCH]` stage logging + UI progress |

## Files modified

- `src/core/use_cases/search_tiles.py`
- `src/presentation/workers/search_worker.py`
- `src/presentation/viewmodels/search_viewmodel.py`
- `src/presentation/views/search_view.py`
- `src/utils/image_formats.py` (`.jfif`)
- `src/utils/search_stages.py` (new)
- Tests: `test_search_reliability.py`, filter/viewmodel updates

## Stage log sequence (success)

```
Drop accepted
Index health OK
Image decoded / Embedding cache hit
Preprocess complete
Embedding generated
Embedding normalized
FAISS search complete — N IDs
SQLite metadata loaded — N records
Hybrid rerank complete
Weak-result filter applied
Thumbnails queued
Results ready for UI
Results displayed
```

## Cross-platform notes

- Formats: JPG/JPEG/JFIF/PNG/WEBP/BMP/TIFF (+ HEIC when pillow-heif present)
- Heartbeat → ViewModel uses `QueuedConnection`
- Cancelled worker no longer leaves Mac/Windows UI stuck on “Searching…”
- No change to DINOv2 / FlatIP / ranking math beyond never zeroing the top hit

## Tests

`pytest` search reliability + filters + viewmodel: **37 passed**
