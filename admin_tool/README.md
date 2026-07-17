# TileVision AI — Admin License Manager

A standalone tool for **you (the vendor)** to issue offline license keys to
customers. This is separate from the customer-facing application and must
**never** be bundled into the customer installer — it can load your private
signing key.

Full workflow and production checklist: [docs/VENDOR_LICENSING.md](../docs/VENDOR_LICENSING.md)

## Setup (one-time)

1. Run the tool:
   ```
   python admin_tool/main.py
   ```
2. Click **"Generate New Keypair"**. Save the private key file somewhere
   safe and offline (e.g. an encrypted USB drive) — never commit it to git,
   never put it on the same machine you build installers on if you can
   avoid it.
3. Copy the public key PEM shown in the output panel into
   `src/licensing/validator.py`, replacing the placeholder
   `EMBEDDED_PUBLIC_KEY_PEM` value (or use **Show Public Key for Customer App** later).
4. Rebuild/re-package the customer application with the real public key
   embedded.

From then on, each time you open the Admin tool the signing key loads automatically from
`%USERPROFILE%\.tilevision_ai_vendor\vendor_private_key.pem`. Use **Import Key File**
only if you need to copy a key from elsewhere into that folder.

The tool also writes `vendor_public_key.pem` beside the private key so the customer app
can verify keys on the same PC during development.

## Issuing a license

1. Get the customer's **Machine ID** (Hardware Fingerprint) — they copy it from the
   Activation screen in their copy of the app.
2. Fill in Customer Name, paste their Machine ID, pick a License Type (trial or full).
3. Click **Generate License Key**, then **Copy to Clipboard**.
4. Send the key to the customer; they paste it into their Activation screen.

## Customers & Licenses tab

| Filter | Use |
|--------|-----|
| **Current (1 per PC)** | Default — one active row per Machine ID |
| **All history** | Every key ever issued |
| **Trial (active)** | Active trials only |
| **Suspended** | Cancelled keys |
| **Old key** | Replaced by renewal |

**Actions:**

- **Renew / Extend Selected** — generates a new key; old row becomes **Old key**
- **Cancel Selected** — marks **Suspended**; blocks new keys for that Machine ID
- **Allow Re-issue** — clears the block after cancellation so you can issue a new key
- **Export CSV** / **Export Revocation JSON** — accounting and app updates

## What this tool can and can't do (offline licensing limitations)

Because the end-user app never phones home (per the "fully offline, no
cloud" requirement), there's no remote kill-switch. Concretely:

| Spec item | How it actually works here |
|---|---|
| Generate License Keys | ✅ As above. |
| Extend License / Change Expiry | **Renew / Extend** — new key; old becomes **Old key**. |
| Generate Lifetime License | ✅ Select "Lifetime" — uses a far-future sentinel expiry date. |
| Deactivate License | ⚠️ **Cancel Selected** + revocation list in next app update. Already-activated offline installs keep working until expiry or update. |
| Reset Hardware Binding | Generate a new key using the customer's **new** Machine ID. |

## Security notes

- The wildcard "any machine" checkbox produces a `hardware_hash: "*"` key.
  The customer application only honors this when it's built with
  `TILEVISION_DEV_MODE=1` set — a real production build **rejects** wildcard
  keys outright. It exists purely for your own internal testing.
- Keep your private key file offline. Anyone who has it can mint valid
  licenses for any customer.
- Automatic backup zip is saved to OneDrive or Documents — see main docs.
