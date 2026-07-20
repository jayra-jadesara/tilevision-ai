# Cross-Platform Setup — TileVision AI

TileVision AI runs on **Windows**, **macOS**, and **Linux**. Windows is the production target with a packaged installer; macOS and Linux are supported for run-from-source installs and evaluation.

---

## Quick start (all platforms)

```bash
pip install -r requirements.txt

# Development only — bypasses signature/hardware checks
export TILEVISION_DEV_MODE=1        # macOS / Linux
# set TILEVISION_DEV_MODE=1         # Windows CMD
# $env:TILEVISION_DEV_MODE=1        # Windows PowerShell

python dev_tools/create_dev_license.py
python main.py
```

For a **real license** (without dev mode), copy the Machine ID from the Activation screen and generate a signed key in `admin_tool/main.py` on that same machine/OS.

Data is stored under `~/.tilevision_ai/` on all platforms.

---

## macOS

### Requirements

- macOS 12+ (Monterey or newer recommended)
- Python 3.12+
- Xcode Command Line Tools (`xcode-select --install`)

### Install

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export TILEVISION_DEV_MODE=1
python dev_tools/create_dev_license.py
python main.py
```

### GPU acceleration

- **Apple Silicon (M1/M2/M3):** Standard PyTorch wheel uses **MPS (Metal)** automatically.
- **Intel Mac:** CPU inference only.

Check GPU status:

```bash
python dev_tools/check_gpu.py
```

### Machine ID sources

`IOPlatformUUID`, hardware serial number, and hostname (see `docs/VENDOR_LICENSING.md`).

### License storage

`~/Library/Application Support/TileVisionAI/.lic/`

---

## Linux

### Requirements

- Python 3.12+
- Qt/X11 runtime libraries for PySide6

Check common Qt dependencies:

```bash
bash scripts/check_qt_deps.sh
```

On **Debian/Ubuntu**:

```bash
sudo apt update
sudo apt install -y python3.12-venv python3-pip \
  libxcb-cursor0 libxcb-xinerama0 libxkbcommon-x11-0 libegl1 libglib2.0-0
```

### Install

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export TILEVISION_DEV_MODE=1
python dev_tools/create_dev_license.py
python main.py
```

### GPU acceleration (NVIDIA only)

```bash
bash scripts/install_pytorch_cuda.sh
python dev_tools/check_gpu.py
```

AMD/Intel GPUs use CPU inference.

### Large folder monitoring

If watch-folder indexing fails on very large trees, increase inotify limits:

```bash
sudo sysctl fs.inotify.max_user_watches=524288
```

### Machine ID sources

`/etc/machine-id`, DMI product UUID, and hostname.

### License storage

`~/.local/share/TileVisionAI/.lic/` (or `$XDG_DATA_HOME/TileVisionAI/.lic/`)

---

## Windows (production)

See `README.md` and `packaging/README.md` for the installer build. CUDA install:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_pytorch_cuda.ps1
```

---

## First-run AI model download

DINOv2 weights (~1 GB) are required for indexing and search.

**Option A — download on first run (needs internet once):**

The app downloads automatically from Hugging Face when you first index or search.

**Option B — pre-download for offline / packaging (recommended for vendors):**

```bash
python scripts/download_dinov2_model.py
python scripts/preflight_check.py
```

Weights are saved to `model_weights/dinov2-large/` and bundled by PyInstaller.

**Option C — use an existing Hugging Face cache:**

```bash
export TILEVISION_MODEL_DIR=~/.cache/huggingface/hub/models--facebook--dinov2-large/snapshots/<hash>
export TILEVISION_OFFLINE_MODEL=1
python main.py
```

---

## Platform comparison

| Feature | Windows | macOS | Linux |
|---------|---------|-------|-------|
| Packaged installer | Yes (.exe) | Yes (.app / .dmg) | Optional |
| Machine ID / licensing | Yes | Yes | Yes |
| NVIDIA CUDA | Yes | No | Yes |
| Apple MPS (Metal) | No | Yes (Apple Silicon) | No |
| CPU fallback | Yes | Yes | Yes |
| Folder watch (watchdog) | Yes | Yes | Yes* |

\*Linux may need higher `inotify` limits for very large catalogues.

---

## Vendor admin tool

The admin license manager runs on all platforms:

```bash
python admin_tool/main.py
```

Vendor keys and ledger are stored in `~/.tilevision_ai_vendor/`. Click **Backup Now** in the admin tool to save a zip to OneDrive, iCloud Drive, Dropbox, or Documents.
