"""
Offline license revocation support.

Because TileVision AI runs fully offline, a vendor cannot remotely kill an
already-activated install. Revocation works in two layers:

1. Vendor ledger (admin tool) — marks a license as cancelled and blocks
   issuing new keys for that customer/machine.
2. Client revocation list — license IDs in EMBEDDED_REVOKED_LICENSE_IDS or
   revoked_licenses.json are rejected on activation/startup validation.

Ship app updates periodically with an updated embedded list to enforce
refunds on customers who install the update.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import AbstractSet, FrozenSet, Set

logger = logging.getLogger("tilevision.licensing.revocation")

# Updated by the vendor before each release (admin tool → Export for Release).
EMBEDDED_REVOKED_LICENSE_IDS: FrozenSet[str] = frozenset()


def _revocation_file_paths() -> list[Path]:
    paths: list[Path] = []
    program_data = os.environ.get("PROGRAMDATA")
    if program_data:
        paths.append(Path(program_data) / "TileVisionAI" / "revoked_licenses.json")
    paths.append(Path.home() / ".tilevision_ai" / "revoked_licenses.json")
    return paths


def load_revoked_license_ids() -> Set[str]:
    """Merge embedded and on-disk revocation lists."""
    revoked: Set[str] = set(EMBEDDED_REVOKED_LICENSE_IDS)
    for path in _revocation_file_paths():
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                revoked.update(str(item) for item in data)
            elif isinstance(data, dict):
                ids = data.get("revoked_license_ids", [])
                if isinstance(ids, list):
                    revoked.update(str(item) for item in ids)
        except Exception as exc:
            logger.warning("Could not read revocation file %s: %s", path, exc)
    return revoked


def is_license_revoked(license_id: str | None, revoked_ids: AbstractSet[str] | None = None) -> bool:
    if not license_id:
        return False
    active = revoked_ids if revoked_ids is not None else load_revoked_license_ids()
    return license_id in active


def save_revocation_file(revoked_ids: AbstractSet[str], path: Path) -> None:
    """Write a revocation manifest for manual import or release embedding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "revoked_license_ids": sorted(revoked_ids),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
