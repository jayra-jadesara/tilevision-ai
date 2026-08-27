# TileVision AI — Live pricing quote

`prices.json` is the **online source of truth** for the in-app Pricing Quote PDF.

## How customers get updates

1. Edit `pricing/prices.json` on the `main` branch.
2. Commit and push (no software version bump / installer rebuild).
3. In TileVision AI → **Help** → **Pricing Quote (PDF)**.
4. The app downloads this file, caches it locally, and generates a fresh PDF.

Remote URL used by the app:

```text
https://raw.githubusercontent.com/jayra-jadesara/tilevision-ai/main/pricing/prices.json
```

## Offline behaviour

1. Try live download.
2. If offline / failed → use `~/.tilevision_ai/cache/prices.json` (last successful download).
3. If no cache yet → use the bundled copy shipped with the app (`src/resources/pricing/prices.json`).

## What you can change without a new release

Everything in `prices.json`: plans, prices, discounts, features, vendor name/phone/email, location, taxes line, hero text, “why choose us” points, etc.

After editing `pricing/prices.json`, also copy the same file to
`src/resources/pricing/prices.json` when you next ship an installer (so new
offline installs get a recent fallback). Live online users do **not** need that.
