# Packaging TileVision AI for Windows

## Prerequisites

On a Windows build machine (PyInstaller doesn't cross-compile — build on Windows for Windows):

```
pip install -r requirements.txt
pip install pyinstaller
```

## 1. Pre-download the CLIP model weights (one-time, needs internet)

The app is fully offline **at runtime**, but `open_clip`'s default
`pretrained="laion400m_e32"` setting downloads weights from a remote hub the
*first* time it's used. That download must happen on your build machine
before packaging — never on a customer's machine — otherwise a customer
with genuinely no internet access will hit a hard failure loading the AI
model on first launch.

```python
# Run once, with internet access, to populate the local cache:
import open_clip
open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="laion400m_e32")
```

Then either:
- Point `src/config/settings.py`'s default `pretrained` value at the cached
  weights file path directly, or
- Add the cache directory (`open_clip.pretrained.get_pretrained_cfg`'s
  cache location, typically `~/.cache/clip` or similar) to the `datas` list
  in `packaging/tilevision.spec` so PyInstaller bundles it.

## 2. Set up licensing

Follow `admin_tool/README.md` to generate a real signing keypair and embed
the public key into `src/licensing/validator.py`
(`EMBEDDED_PUBLIC_KEY_PEM`). **Do not ship a build with the placeholder
key** — it isn't a valid key and license activation will fail for every
customer.

Confirm `TILEVISION_DEV_MODE` is **not** set in your build environment:

```
echo %TILEVISION_DEV_MODE%
```
Should print nothing/empty. If it prints `1`, unset it before building —
otherwise you'll ship a build with signature verification disabled.

## 3. Build

From the project root:

```
pyinstaller packaging/tilevision.spec --clean
```

Output: `dist/TileVisionAI/TileVisionAI.exe` plus its supporting files —
distribute the whole `TileVisionAI/` folder, not just the `.exe`.

(A one-folder build is used deliberately over `--onefile`: `--onefile`
re-extracts the ~1-2GB of torch/CLIP weights to a temp directory on every
launch, adding many seconds to startup every time.)

## 4. Test the built .exe

Before shipping, run the actual built executable (not `python main.py`) on
a clean Windows VM with **no internet access** and confirm:
- The app launches and shows the trial/activation screen.
- Folder indexing completes and produces search results.
- Search actually returns results (confirms the model weights bundled
  correctly).
- License activation works with a key from the Admin tool.

## 5. Installer (optional, recommended for a commercial release)

PyInstaller produces a folder, not an installer. Wrap it with
[Inno Setup](https://jrsoftware.org/isinfo.php) (free) or
[NSIS](https://nsis.sourceforge.io/) to get a proper `TileVisionAI-Setup.exe`
that installs to Program Files, creates a Start Menu shortcut, and can set
up the `%PROGRAMDATA%\TileVisionAI` folder permissions needed for the
encrypted license store. This step isn't automated here — an Inno Setup
`.iss` script is a reasonable next addition once you've validated the raw
PyInstaller build above.

## Known packaging gaps (being upfront)

- No CI/build automation — this is a manual, documented process, not a
  one-command release pipeline.
- No code signing — Windows SmartScreen will warn on first run of an
  unsigned .exe. A code signing certificate is a separate (paid) step.
- No installer script yet (see step 5).
