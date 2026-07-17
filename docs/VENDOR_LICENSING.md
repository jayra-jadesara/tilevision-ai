# Vendor Licensing Guide — TileVision AI

## How offline licensing works (simple)

1. **Customer** installs TileVision AI and opens the Activation screen.
2. Customer copies their **Machine ID** (hardware fingerprint) and sends it to you.
3. **You (vendor)** open the Admin License Manager:
   ```powershell
   python admin_tool/main.py
   ```
4. Load your **private signing key** — saved automatically at
   `%USERPROFILE%\.tilevision_ai_vendor\vendor_private_key.pem` (or use **Import Key File**
   / **Create New Key** on first setup).
5. Enter customer name, paste Machine ID, choose license type, click **Generate License Key**.
6. Copy the key and email/WhatsApp it to the customer.
7. Customer pastes the key into TileVision AI → software works **fully offline** until expiry.

No internet is required on the customer PC after activation.

---

## License types you can issue

| Type | Duration |
|------|----------|
| 15-Day Trial | 15 days |
| 1-Month Trial | 30 days |
| 2-Month Trial | 60 days |
| 3-Month Trial | 90 days |
| 6-Month Trial | 180 days |
| 1-Year | 365 days |
| 3-Year | 1,095 days |
| Lifetime | No expiry |

The built-in **15-day auto trial** (no key) still starts automatically on first run if the customer has no paid license.

---

## Vendor registry (your customer list)

The admin tool saves every issued key locally in:

`%USERPROFILE%\.tilevision_ai_vendor\license_ledger.db`

Your **private signing key** is stored in one fixed place (same folder):

`%USERPROFILE%\.tilevision_ai_vendor\vendor_private_key.pem`

The tool loads it automatically on startup — no need to browse each time.

**Automatic cloud backup:** each time you open the vendor tool or issue a license,
a zip backup is saved under `%USERPROFILE%\OneDrive\TileVision-Vendor-Backup\`
(or `Documents\TileVision-Vendor-Backup\` if OneDrive is not available). If you
use OneDrive/Google Drive sync on that folder, your key survives PC loss.

Use the **License Registry** tab to:

- See all customers (active / cancelled)
- Export CSV for accounting
- **Cancel** a license (marks it cancelled in your records)

---

## Cancelling a license (important — offline limits)

**You cannot remotely switch off software on a PC that is already activated and never goes online.**

What cancellation **does**:

- Blocks **new activations** of that license ID (revocation list)
- Lets you refuse to issue replacement keys for that customer/machine
- Exports a revocation file for the next app update

What cancellation **does not** do instantly:

- Stop software on a PC that already activated the old key (until expiry or until they install an update that includes the revoked ID)

**Practical workflow for refunds:**

1. Cancel the license in the **License Registry** tab.
2. Click **Export Revocation List** → `revoked_licenses.json`.
3. Before your next release, add cancelled IDs to `src/licensing/revocation.py` → `EMBEDDED_REVOKED_LICENSE_IDS`.
4. Ship the updated installer — revoked keys fail validation on startup/activation.

Optional: place `revoked_licenses.json` on a customer PC at  
`%PROGRAMDATA%\TileVisionAI\revoked_licenses.json` (manual support step).

---

## Database password protection (customer catalogue)

The tile catalogue database (`tiles.db`) is **encrypted when the app is closed**.

- Default: key derived from the PC hardware fingerprint (not readable on another PC).
- Vendor override: set environment variable `TILEVISION_DB_PASSWORD=YourSecret` before launching for all installs at a site.

While the app is running, a decrypted working copy exists (required for SQLite). After exit, only `tiles.db.enc` remains.

---

## First-time customer installation (simplified wizard)

The setup wizard now has **3 main steps** (not dozens of package names):

1. Runtime check (Python + SQLite)
2. Application core (UI, PDF, monitoring)
3. AI search engine (PyTorch + FAISS + vision libraries)

All packages are still listed in `requirements.txt` for IT staff:
```powershell
pip install -r requirements.txt
```

---

## Production checklist (before selling)

1. Generate a **new** keypair in the admin tool (do not use `dev_tools/dev_private_key.pem`).
2. Embed the **public** key in `src/licensing/validator.py`.
3. Store the **private** key securely offline.
4. Ensure `TILEVISION_DEV_MODE` is **not** set in production builds.
5. Build the customer installer without `admin_tool/` or any `.pem` private keys.
