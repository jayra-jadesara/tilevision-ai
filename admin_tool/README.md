# TileVision AI — Admin License Manager

**For you (the vendor) only.** Do not give this tool to customers.

**Works on Windows and Mac.** Same features on both. Data saves in your home folder:
- Windows: `C:\Users\You\.tilevision_ai_vendor\`
- Mac: `/Users/You/.tilevision_ai_vendor/`

Full guide: [docs/VENDOR_LICENSING.md](../docs/VENDOR_LICENSING.md)

## Start the tool

```
python admin_tool/main.py
```

On Mac you can also use:
```
python3 admin_tool/main.py
```

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

## Security

- Never share your private key file.
- The "any machine" (wildcard) checkbox is for your testing only — production builds reject it.
