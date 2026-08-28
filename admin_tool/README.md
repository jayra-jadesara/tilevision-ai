# TileVision AI — Vendor Admin

**For you (the vendor) only.** Do not give this tool to customers.

**Works on Windows and Mac.** Same features on both. Data saves in your home folder:
- Windows: `C:\Users\You\.tilevision_ai_vendor\`
- Mac: `/Users/You/.tilevision_ai_vendor/`

Full guide: [docs/VENDOR_LICENSING.md](../docs/VENDOR_LICENSING.md)

## Navigation (left sidebar)

| Menu | Purpose |
|------|---------|
| **Overview** | Stats and quick steps |
| **Licenses** | Generate Key + Customers & Licenses (one place) |
| **Pricing** | Edit live rates and publish to GitHub |
| **Signing Key** | Import/create key, backup, public key for customer app |

## Start the tool

**Windows (easiest):** install `TileVisionAI-Admin-VENDOR-ONLY-*.exe` from the GitHub release
(vendor section — **never send this file to customers**), then open **TileVision AI Admin**
from the Start menu.

**From source (Windows or Mac):**

```
python admin_tool/main.py
```

## Admin login

The admin tool asks for a password when it opens.

Default password: `raj!RAJ!`

Change it later by editing the hash in `%USERPROFILE%\.tilevision_ai_vendor\admin_settings.json`
(contact support if you need help resetting it).

## First-time setup

1. Click **Create New Key (First Setup)** and save your private key somewhere safe.
2. Click **Show Public Key for Customer App** and put that key in `src/licensing/validator.py`.
3. Rebuild the customer app with that public key inside.

After that, your signing key loads automatically each time you open the tool.

Click **Backup Now** when you want a backup copy. Nothing is backed up until you click that button.

## Make a license key

1. Customer copies **Machine ID** from their TileVision app (Activation screen).
2. **Generate Key** tab → customer name, Machine ID, license type → **Generate License Key**.
3. **Copy to Clipboard** → send the key to the customer.

## Customers & Licenses tab

| Filter | What it shows |
|--------|----------------|
| **Current (1 per PC)** | One active row per PC |
| **All history** | Every key you ever made |
| **Trial (active)** | Active trials only |
| **Stopped** | Keys you blocked |
| **Old key** | Replaced when you made a newer key |

### Buttons (simple guide)

| Button | What it does |
|--------|----------------|
| **Extend License** | Make a new key for the same customer and PC. Old row becomes **Old key**. |
| **Copy Key Again** | Copy the saved key from the selected row. |
| **Stop License** | Block this key. Customer cannot get a new key for that PC until you click **Allow New Key**. |
| **Delete Row** | Remove a **Stopped** row from your list. Only works on Stopped rows. |
| **Allow New Key** | After stopping a key, use this so you can make a new key for the same PC. |
| **Export CSV** | Spreadsheet of all licenses. |
| **Export Block List** | JSON file of stopped keys for the next customer app update. |
| **Copy Block List Code** | Python code to paste into the customer app before release. |

## Important (offline apps)

The customer app does not phone home. So:

- **Stop License** blocks new keys and adds the key to the block list.
- PCs already using the key keep working until it expires, or until you ship an app update with the block list.
- **Delete Row** removes the row from your table. If you already shipped a block list, update the app to match.

## Pricing tab (live customer rates)

Use **Pricing** to edit `prices.json` and publish to GitHub. All customer apps
download the live file — no new installer needed.

1. Open **Pricing** tab.
2. Paste a GitHub **Personal access token** with `repo` contents write access.
3. Click **Test connection**, then **Save GitHub settings**.
4. Click **Load Live from GitHub** (or edit the form).
5. Change plan prices, location, vendor phone, taxes line, etc.
6. Click **Preview PDF** to see what customers will get.
7. Click **Publish to GitHub** — updates:
   - `pricing/prices.json` (live URL customers fetch)
   - `src/resources/pricing/prices.json` (bundled fallback for new installs)

Drafts save to `~/.tilevision_ai_vendor/prices_draft.json`. Backups go to
`~/.tilevision_ai_vendor/pricing_backups/` before each publish.

Create a token: GitHub → Settings → Developer settings → Personal access tokens
→ fine-grained or classic with **Contents: Read and write** on the repo.

## Security

- Never share your private key file.
- The "any machine" (wildcard) checkbox is for your testing only — production builds reject it.
- Never share your GitHub token — it is stored only in `admin_settings.json` on your PC.
