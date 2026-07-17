# TileVision AI — Admin License Manager

A standalone tool for **you (the vendor)** to issue offline license keys to
customers. This is separate from the customer-facing application and must
**never** be bundled into the customer installer — it can load your private
signing key.

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
   `EMBEDDED_PUBLIC_KEY_PEM` value.
4. Rebuild/re-package the customer application with the real public key
   embedded.

From then on, each time you open the Admin tool the signing key loads automatically from
`%USERPROFILE%\.tilevision_ai_vendor\vendor_private_key.pem`. Use **Import Key File**
only if you need to copy a key from elsewhere into that folder.

## Issuing a license

1. Get the customer's **Machine ID** (Hardware Fingerprint) — they can copy
   it from the Activation screen in their copy of the app.
2. Fill in Customer Name, paste their Machine ID, pick a License Type.
3. Click **Generate License Key**, then **Copy to Clipboard**.
4. Send the key to the customer; they paste it into their Activation screen.

## What this tool can and can't do (offline licensing limitations)

Because the end-user app never phones home (per the "fully offline, no
cloud" requirement), there's no remote kill-switch. Concretely:

| Spec item | How it actually works here |
|---|---|
| Generate License Keys | ✅ As above. |
| Extend License / Change Expiry | Generate a **new** key for the same customer + Machine ID with a later expiry. They re-activate with it, which overwrites the old one in their local database. |
| Generate Lifetime License | ✅ Select "Lifetime" — uses a far-future sentinel expiry date. |
| Deactivate License | ⚠️ Not truly possible for an already-offline-activated install without a check-in mechanism (which would contradict "no internet dependency"). The practical mitigation is to stop issuing new keys for a known Machine ID and keep your own log of issued keys. |
| Reset Hardware Binding | Generate a new key using the customer's **new** Machine ID. Their old key naturally won't validate on the new machine — nothing needs to be revoked on the old one for this to work. |

## Security notes

- The wildcard "any machine" checkbox produces a `hardware_hash: "*"` key.
  The customer application only honors this when it's built with
  `TILEVISION_DEV_MODE=1` set — a real production build **rejects** wildcard
  keys outright. It exists purely for your own internal testing.
- Keep your private key file offline. Anyone who has it can mint valid
  licenses for any customer.
