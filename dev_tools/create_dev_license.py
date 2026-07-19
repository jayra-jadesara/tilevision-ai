"""
TileVision AI — Quick Development License Creator.

Run this script ONCE before starting the app for the first time in development.
It creates a wildcard license key and saves it directly into the local SQLite database,
bypassing the activation dialog entirely — so you can test without a real license.

Usage:
    python dev_tools/create_dev_license.py

This script:
    1. Initializes the local SQLite database.
    2. Generates a wildcard dev license key (any machine, expires 2030-12-31).
    3. Saves it into the database.
    4. Prints the license key for reference.

IMPORTANT: Only use in development. Never ship or deploy this script.
"""

import sys
import json
import base64
import os
from pathlib import Path
from datetime import datetime

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import AppSettings
from src.data.db_context import DatabaseContext
from src.data.sqlite_repository import SQLiteLicenseRepository
from src.core.models import LicenseInfo


WILDCARD_LICENSE_PAYLOAD = {
    "customer_name": "TileVision Development License",
    "expires_at": "2030-12-31",
    "hardware_hash": "*",
    "signature": "DEVELOPMENT_BYPASS_NO_REAL_SIGNATURE",
}

WILDCARD_LICENSE_KEY = base64.b64encode(
    json.dumps(WILDCARD_LICENSE_PAYLOAD).encode("utf-8")
).decode("utf-8")


def main() -> None:
    """Create and save a development wildcard license key to the local database."""
    print("=" * 60)
    print("  TileVision AI — Dev License Creator")
    print("=" * 60)

    # Load settings to get the database path
    settings = AppSettings()
    db_path = settings.database_path
    print(f"\nDatabase path: {db_path}")

    # Initialize the database
    print("Initializing database schema...")
    db_context = DatabaseContext(db_path=db_path)
    license_repo = SQLiteLicenseRepository(db_context=db_context)

    # Create and save the development license
    dev_license = LicenseInfo(
        license_key=WILDCARD_LICENSE_KEY,
        hardware_hash="*",
        customer_name="TileVision Development License",
        expires_at="2030-12-31",
        activated_date=datetime.now(),
    )

    success = license_repo.save_license(dev_license)

    if success:
        print("\n✅ Development license created and saved successfully!")
        print(f"\n   Customer : {dev_license.customer_name}")
        print(f"   Expires  : {dev_license.expires_at}")
        print(f"   HW Hash  : * (wildcard — any machine)")
        print(f"\n   License Key:\n   {WILDCARD_LICENSE_KEY}")
        print("\n" + "=" * 60)
        print("Before launching, enable developer mode on all platforms:")
        print()
        if os.name == "nt":
            print("  set TILEVISION_DEV_MODE=1")
        else:
            print("  export TILEVISION_DEV_MODE=1")
        print()
        print("Then launch: python main.py")
        print("The activation dialog will be skipped in developer mode.")
    else:
        print("\n❌ Failed to save development license to database.")
        sys.exit(1)


if __name__ == "__main__":
    main()
