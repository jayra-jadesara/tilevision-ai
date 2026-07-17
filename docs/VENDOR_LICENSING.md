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

On first run, the customer sees one **License Activation** screen: copy Machine ID, request a key from you, then paste the trial or full license key in the same field.

**Trial workflow:**
1. Customer copies Machine ID and requests a trial.
2. You generate a key with license type **15-Day Trial** (or 1-Month Trial, etc.).
3. Customer pastes that key on the activation screen — same as a full license.

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

Use the **Customers & Licenses** tab to:

- Filter by **Current (1 per PC)** — default view; one active row per Machine ID
- Filter **All history** to see every key ever issued (active, trial, suspended, old key)
- **Renew / Extend** a license (old key becomes **Old key**, new row is active)
- **Cancel Selected** (status **Suspended**, red) — blocks new keys until **Allow Re-issue**
- **Allow Re-issue** — clear the block on a Machine ID after cancellation so you can issue a new key
- Export CSV for accounting

**Status colors:**

| Status | Meaning |
|--------|---------|
| **Active** (green) | Current full license — send this key |
| **Trial** (yellow) | Active trial |
| **Suspended** (red) | Cancelled — do not send |
| **Old key** (gray) | Replaced when you renewed — history only |

**Signing key sync:** the tool auto-writes `vendor_public_key.pem` next to your private key. For same-PC testing, the customer app uses that file automatically. For production installers, click **Show Public Key for Customer App** and embed it in `src/licensing/validator.py`.

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

Export catalogue **company profiles** (Settings → Export Profiles) are also stored in this database, scoped to the licensed customer name on that PC.

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

Complete every step and test on a clean PC before shipping to customers.

### Vendor setup (one time)

1. Run `python admin_tool/main.py` → **Create New Keypair** (do **not** ship `dev_tools/dev_private_key.pem`).
2. Copy the **public** key into `src/licensing/validator.py` → `EMBEDDED_PUBLIC_KEY_PEM` (or use **Show Public Key for Customer App**).
3. Store `vendor_private_key.pem` offline (encrypted USB / password manager). Enable OneDrive backup folder if desired.
4. Confirm `vendor_public_key.pem` is written beside the private key (auto-sync for local testing).

### Build the customer app

5. Ensure `TILEVISION_DEV_MODE` is **not** set in production builds (wildcard `*` Machine ID keys are rejected).
6. Build the installer **without** `admin_tool/`, `dev_tools/`, or any `.pem` private key files.
7. Embed the matching **public** key in the shipped app (step 2).

### Test on a clean customer PC

8. Install the customer build → activation screen appears.
9. Copy Machine ID → generate a **trial** key in the admin tool → paste in customer app → app opens.
10. Generate a **full license** key for the same Machine ID → customer re-activates → status bar shows license type with badge icon.
11. Settings → Export Profiles → save company details → Search → Export Catalogue PDF.
12. Cancel a test license in the admin tool → confirm new keys are blocked → **Allow Re-issue** → issue replacement key.

### Revocation (when refunding)

13. Cancel the license in **Customers & Licenses**.
14. Export revocation JSON / copy Python snippet → add IDs to `EMBEDDED_REVOKED_LICENSE_IDS` before the next release.
15. Ship an update so revoked keys fail on startup.

### Ongoing

16. Use **Current (1 per PC)** filter for day-to-day customer list; **All history** for audit.
17. Keep private key backup in `%USERPROFILE%\OneDrive\TileVision-Vendor-Backup\` (or Documents fallback).
