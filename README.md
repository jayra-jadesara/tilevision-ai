# TileVision AI

> Offline AI-powered visual tile similarity search for showrooms and distributors.

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB.svg)](https://python.org)
[![PySide6](https://img.shields.io/badge/UI-PySide6-41CD52.svg)](https://doc.qt.io/qtforpython/)
[![License](https://img.shields.io/badge/License-Proprietary-red.svg)](#)

---

## Overview

TileVision AI lets tile showrooms **instantly find visually similar tiles** from a local image catalog using AI embeddings and vector search — completely offline, no cloud, no API keys required.

Search combines **DINOv2-large** semantic embeddings with handcrafted descriptors (LAB color, multi-scale LBP texture, edge orientation, pattern structure) and a hybrid reranker tuned per pattern family (speckled, marble, terrazzo, plain, textured).

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | PySide6 (Qt for Python) |
| AI Embeddings | Meta DINOv2-large (`facebook/dinov2-large`, 1024D) |
| Handcrafted Descriptors | LAB color, multi-scale LBP, edge orientation, pattern structure |
| Vector Search | FAISS CPU (IndexFlatIP, cosine via L2-normalized vectors) |
| Image Processing | Pillow, OpenCV, scikit-image |
| Database | SQLite 3 |
| Language | Python 3.12+ |
| Licensing | ECDSA offline hardware lock |

---

## Setup (Development)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

**Linux:** install Qt/X11 libraries first — see [docs/CROSS_PLATFORM.md](docs/CROSS_PLATFORM.md).

### 2. Enable developer mode and create a dev license

```bash
export TILEVISION_DEV_MODE=1          # macOS / Linux
# set TILEVISION_DEV_MODE=1           # Windows CMD

python dev_tools/create_dev_license.py
```

### 3. Launch the application

```bash
python scripts/preflight_check.py   # optional but recommended
python main.py
```

Data is stored under `~/.tilevision_ai/` (Windows: `%USERPROFILE%\.tilevision_ai\`).

### Vendor license manager (you only — never ship to customers)

```bash
python admin_tool/main.py
```

See [docs/VENDOR_LICENSING.md](docs/VENDOR_LICENSING.md) for the full workflow: issue trial/full keys, customer registry, renewals, cancellation, and production checklist.

### Supported platforms

| Platform | Status |
|---|---|
| **Windows 10/11** | Production target — installer, GPU (CUDA), full QA |
| **macOS** | Run from source; Apple Silicon uses MPS GPU; licensing works |
| **Linux** | Run from source; NVIDIA CUDA supported; licensing works |

See [docs/CROSS_PLATFORM.md](docs/CROSS_PLATFORM.md) for per-OS setup, Qt dependencies, and GPU install scripts.

**Release builds:** tag `v*` pushes trigger [GitHub Actions builds](.github/workflows/build.yml) for Windows, macOS, and Linux artifacts.

---

## Project Structure

```
tilevision-ai/
├── main.py                          # Entry point
├── requirements.txt
├── src/
│   ├── app.py                       # Composition Root (DI bootstrapper)
│   ├── ai/
│   │   ├── embedder.py              # DINOv2 multi-view embedder (3 views, batched)
│   │   ├── feature_extractor.py     # Unified preprocess + descriptor pipeline
│   │   ├── reranker.py              # HybridReRanker (DINOv2 + descriptors)
│   │   ├── pattern_classifier.py    # Rule-based pattern family classifier
│   │   ├── candidate_filter.py      # LAB dominant-color pre-filter
│   │   ├── vector_index.py          # FAISS index manager
│   │   ├── feature_versions.py      # Pipeline version tracking
│   │   └── descriptors/             # Color, texture, edge, pattern descriptors
│   ├── config/
│   │   └── settings.py              # JSON-backed application settings
│   ├── core/
│   │   ├── models.py                # Domain entities (TileImage, SearchResult)
│   │   └── use_cases/
│   │       ├── index_images.py      # Folder indexing (batched DINOv2)
│   │       ├── search_tiles.py      # Visual similarity search
│   │       ├── monitor_folder.py    # Auto watch-folder indexing
│   │       ├── find_duplicates.py   # Near-duplicate detection
│   │       └── validate_license.py  # License check use case
│   ├── data/
│   │   ├── db_context.py            # SQLite connection manager
│   │   ├── repository_interface.py  # Abstract repository interfaces
│   │   └── sqlite_repository.py     # SQLite implementations
│   ├── licensing/
│   │   ├── hardware.py              # Cross-platform hardware fingerprinting
│   │   └── validator.py             # ECDSA license validator
│   └── presentation/                # Views, ViewModels, workers
├── tests/                           # Unit and integration tests
├── admin_tool/                      # Vendor-only license manager (do not ship)
├── docs/
│   └── VENDOR_LICENSING.md          # Vendor workflow + production checklist
└── dev_tools/
    ├── create_dev_license.py        # Dev: seed a wildcard dev license
    ├── generate_license.py          # Vendor: generate real license keys
    └── eval_recall_precision.py     # Offline Recall@K / Precision@K evaluation
```

---

## Architecture

TileVision AI follows **Clean Architecture** with strict layer boundaries:

```
Presentation (Views + ViewModels)
        │  Qt Signals/Slots
        ▼
   Use Cases (Business Logic)
        │  Interfaces
        ▼
  Data Layer (SQLite + FAISS)
        │
  Domain Models (Pure Python)
```

### Indexing pipeline

```
scan folder → batch(4 images) → letterbox preprocess → batched DINOv2 (3 views)
           → LAB color + LBP + edge + pattern descriptors → SQLite + FAISS
```

### Search pipeline

```
query image → extract features once → FAISS coarse retrieval
           → SQLite hydrate → LAB candidate filter → HybridReRanker
           → calibrated display % (self-match → 100%)
```

- All dependencies are constructor-injected in `src/app.py`.
- ViewModels expose Qt Signals only — Views are purely reactive.
- DINOv2 model and FAISS index are warmed up at startup (singleton per process).

---

## Features

### Folder Indexing

- Recursive scan for `.jpg`, `.jpeg`, `.png`, `.webp` files.
- Batched DINOv2 inference (4 images/batch, 3 views/image in one forward pass).
- Letterbox resize (aspect-ratio preserved), border trim, alpha compositing.
- Stores 1024D vectors in FAISS, metadata + descriptors in SQLite.
- **Incremental indexing**: skips unchanged files via SHA-256 hash comparison.
- Background thread with progress bar, ETA, and Pause/Resume/Cancel.

### Visual Similarity Search

- Query any tile image against the indexed catalog.
- Hybrid reranking with DINOv2-primary weights (embedding floor ≥ 50%).
- Pattern-family-aware scoring (speckled, marble, terrazzo, plain, textured).
- Metadata filters: brand, category, color, size.
- Self-match detection (SHA-256 + perceptual hash).
- Calibrated similarity percentages for display.

### Re-indexing after pipeline updates

The feature pipeline is versioned (`feature_version`, `pattern_feature_version`). After upgrading TileVision AI, **rebuild the FAISS index** if the app reports stale features:

1. Delete the old index (e.g. `%USERPROFILE%\.tilevision_ai\index\tiles.index`), or use Settings → Rebuild FAISS Index.
2. Run a full folder scan.

---

## Development Tools

### Search quality evaluation

```bash
# Auto mode: each indexed tile is a query; relevant = same product_code
python dev_tools/eval_recall_precision.py --catalog-auto --max-queries 50

# Explicit ground-truth manifest (JSONL)
python dev_tools/eval_recall_precision.py --manifest eval/queries.jsonl --output eval/results.json
```

### Run tests

```bash
python -m pytest tests/ -q
```

---

## Licensing

TileVision AI uses **offline hardware-locked ECDSA licensing**:

- The public key is embedded in the binary.
- Licenses are locked to a specific machine fingerprint (BIOS + CPU + Registry).
- No internet connection required for validation.

For production key generation, use `python admin_tool/main.py` and follow [docs/VENDOR_LICENSING.md](docs/VENDOR_LICENSING.md).
