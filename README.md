# TileVision AI

> Offline AI-powered visual tile similarity search for showrooms and distributors.

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB.svg)](https://python.org)
[![PySide6](https://img.shields.io/badge/UI-PySide6-41CD52.svg)](https://doc.qt.io/qtforpython/)
[![License](https://img.shields.io/badge/License-Proprietary-red.svg)](#)

---

## Overview

TileVision AI lets tile showrooms **instantly find visually similar tiles** from a local image catalog using AI embeddings and vector search — completely offline, no cloud, no API keys required.

## Tech Stack

| Layer | Technology |
|---|---|
| UI | PySide6 (Qt for Python) |
| AI Embeddings | OpenCLIP (ViT-B/32) |
| Vector Search | FAISS CPU |
| Image Processing | Pillow, OpenCV |
| Database | SQLite 3 |
| Language | Python 3.12+ |
| Licensing | ECDSA offline hardware lock |

---

## Setup (Development)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Create a development license (bypasses activation dialog)

```bash
python dev_tools/create_dev_license.py
```

### 3. Launch the application

```bash
python main.py
```

---

## Project Structure

```
tilevision_ai/
├── main.py                          # Entry point
├── requirements.txt
├── src/
│   ├── app.py                       # Composition Root (DI bootstrapper)
│   ├── ai/
│   │   ├── embedder.py              # OpenCLIP embedding wrapper
│   │   └── vector_index.py          # FAISS index manager
│   ├── config/
│   │   └── settings.py              # JSON-backed application settings
│   ├── core/
│   │   ├── models.py                # Domain entities (TileImage, LicenseInfo)
│   │   └── use_cases/
│   │       ├── index_images.py      # Feature 1: Folder Indexing logic
│   │       ├── search_tiles.py      # Feature 2: Visual Search logic
│   │       ├── monitor_folder.py    # Feature 3: Auto Watch Folder
│   │       └── validate_license.py  # License check use case
│   ├── data/
│   │   ├── db_context.py            # SQLite connection manager
│   │   ├── repository_interface.py  # Abstract repository interfaces
│   │   ├── sqlite_repository.py     # SQLite implementations
│   │   └── settings_store.py        # Settings data service
│   ├── licensing/
│   │   ├── hardware.py              # Windows hardware fingerprinting
│   │   └── validator.py             # ECDSA license validator
│   ├── presentation/
│   │   ├── viewmodels/
│   │   │   └── indexing_viewmodel.py # Feature 1: Indexing ViewModel
│   │   ├── views/
│   │   │   ├── main_window.py       # Main application window + sidebar nav
│   │   │   ├── indexing_view.py     # Feature 1: Indexing UI panel
│   │   │   └── license_view.py      # License activation dialog
│   │   └── workers/
│   │       └── indexing_worker.py   # QThread background indexing worker
│   └── utils/
│       ├── image_utils.py           # Hashing, thumbnails, validation
│       └── logger.py                # Rotating file + console logger
└── dev_tools/
    ├── generate_license.py          # Vendor: generate real license keys
    └── create_dev_license.py        # Dev: seed a wildcard dev license
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

- **No layer references the layer above it.**
- All dependencies are constructor-injected in `src/app.py`.
- ViewModels expose Qt Signals only — Views are purely reactive.

---

## Feature 1: Folder Indexing

- Select any folder via the Browse dialog.
- Recursive scan for `.jpg`, `.jpeg`, `.png`, `.webp` files.
- Generates OpenCLIP embeddings for each image.
- Stores vectors in FAISS (CPU), metadata in SQLite.
- **Incremental indexing**: skips unchanged files via SHA-256 hash comparison.
- Background thread with live **progress bar**, **ETA**, **file counter**.
- Full **Pause**, **Resume**, **Cancel** support.

---

## Licensing

TileVision AI uses **offline hardware-locked ECDSA licensing**:
- The public key is embedded in the binary.
- Licenses are locked to a specific machine fingerprint (BIOS + CPU + Registry).
- No internet connection required for validation.

For production key generation, see `dev_tools/generate_license.py`.

---

## Upcoming Features

| Feature | Description |
|---|---|
| Feature 2 | Visual Similarity Search |
| Feature 3 | Tile Catalog Browser |
| Feature 4 | Auto Folder Watch |
| Feature 5 | Export & Reporting |
