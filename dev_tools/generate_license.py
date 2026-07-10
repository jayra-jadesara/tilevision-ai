"""
TileVision AI — License Key Generator Utility.

This is a VENDOR-ONLY developer tool.
Run this script to:
    1. Generate a new ECDSA P-256 keypair.
    2. Print the public key PEM (embed in validator.py).
    3. Save the private key to a .pem file (keep offline, NEVER distribute).
    4. Generate a test license key for the current machine or a wildcard license.

Usage:
    python dev_tools/generate_license.py [--hardware-hash HASH] [--expires 2027-12-31]
    python dev_tools/generate_license.py --wildcard   # any machine

Design Decision:
    The private key is saved to `dev_tools/private_key.pem` by default.
    In production workflows, run this ONCE, extract the public key PEM, embed it
    in the application binary, then delete the private key from the build machine.
"""

import argparse
import base64
import json
import sys
from datetime import date
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PrivateFormat,
    PublicFormat,
    NoEncryption,
)
from cryptography.hazmat.primitives import hashes
from src.licensing.hardware import get_machine_fingerprint


def generate_keypair() -> tuple:
    """
    Generate a new ECDSA P-256 (SECP256R1) keypair.

    Returns:
        (private_key, public_key) cryptography key objects.
    """
    private_key = ec.generate_private_key(ec.SECP256R1())
    public_key = private_key.public_key()
    return private_key, public_key


def load_keypair(private_key_path: Path):
    """
    Load an existing ECDSA private key from a PEM file.

    Args:
        private_key_path: Path to an existing private key PEM file.

    Returns:
        (private_key, public_key) cryptography key objects.
    """
    from cryptography.hazmat.primitives.serialization import load_pem_private_key

    private_key = load_pem_private_key(private_key_path.read_bytes(), password=None)
    return private_key, private_key.public_key()


def sign_license(
    private_key,
    customer_name: str,
    expires_at: str,
    hardware_hash: str,
) -> str:
    """
    Generate a signed base64-encoded license key.

    Args:
        private_key: ECDSA private key object.
        customer_name: Customer name to embed.
        expires_at: Expiry date in YYYY-MM-DD format.
        hardware_hash: Machine fingerprint or '*' for wildcard.

    Returns:
        Base64-encoded license key string.
    """
    data = f"{customer_name}|{expires_at}|{hardware_hash}".encode("utf-8")
    signature = private_key.sign(data, ec.ECDSA(hashes.SHA256()))

    payload = {
        "customer_name": customer_name,
        "expires_at": expires_at,
        "hardware_hash": hardware_hash,
        "signature": base64.b64encode(signature).decode("utf-8"),
    }
    return base64.b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8")


def main() -> None:
    """Entry point for the license generator CLI."""
    parser = argparse.ArgumentParser(
        description="TileVision AI — License Key Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate a license for the current machine expiring 2027-12-31:
  python dev_tools/generate_license.py --customer "Tile Showroom Ltd" --expires 2027-12-31

  # Generate a wildcard license (any machine):
  python dev_tools/generate_license.py --customer "Demo User" --wildcard --expires 2027-12-31

  # Specify a hardware hash manually:
  python dev_tools/generate_license.py --customer "ABC" --hardware-hash ABCDEF1234
        """,
    )
    parser.add_argument(
        "--customer", default="TileVision Development License",
        help="Customer or company name to embed in the license."
    )
    parser.add_argument(
        "--expires", default=f"{date.today().year + 2}-12-31",
        help="Expiration date in YYYY-MM-DD format (default: 2 years from today)."
    )
    parser.add_argument(
        "--hardware-hash", default=None,
        help="Hardware fingerprint to lock to. Reads the current machine if not specified."
    )
    parser.add_argument(
        "--wildcard", action="store_true",
        help="Create a wildcard license that works on any machine (overrides --hardware-hash)."
    )
    parser.add_argument(
        "--private-key", default="dev_tools/dev_private_key.pem",
        help=(
            "Path to the private key PEM to sign with. Defaults to the committed "
            "dev key (dev_tools/dev_private_key.pem), whose matching public key is "
            "already embedded in src/licensing/validator.py — so a license "
            "generated with the default settings validates immediately, no "
            "copy-pasting a new public key required."
        )
    )
    parser.add_argument(
        "--new-keypair", action="store_true",
        help=(
            "Generate a brand-new keypair instead of reusing --private-key. "
            "You'll need to copy the printed public key into validator.py's "
            "EMBEDDED_PUBLIC_KEY_PEM for the resulting license to validate."
        )
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  TileVision AI — License Key Generator")
    print("=" * 60)

    private_key_path = Path(args.private_key)

    if args.new_keypair or not private_key_path.exists():
        print("\n[1/4] Generating a NEW ECDSA P-256 keypair...")
        private_key, public_key = generate_keypair()

        private_pem = private_key.private_bytes(
            encoding=Encoding.PEM,
            format=PrivateFormat.PKCS8,
            encryption_algorithm=NoEncryption(),
        )
        public_pem = public_key.public_bytes(
            encoding=Encoding.PEM,
            format=PublicFormat.SubjectPublicKeyInfo,
        )

        private_key_path.parent.mkdir(parents=True, exist_ok=True)
        private_key_path.write_bytes(private_pem)
        print(f"[2/4] Private key saved to: {private_key_path.resolve()}")
        print("      ⚠️  KEEP THIS FILE SECURE AND OFFLINE — NEVER COMMIT IT!")
        print("      ⚠️  You must copy the public key below into validator.py")
        print("          for licenses signed with this new key to validate.")
    else:
        print(f"\n[1/4] Reusing existing private key: {private_key_path.resolve()}")
        private_key, public_key = load_keypair(private_key_path)
        public_pem = public_key.public_bytes(
            encoding=Encoding.PEM,
            format=PublicFormat.SubjectPublicKeyInfo,
        )
        print("[2/4] (Skipped saving — reused the key above instead of generating one.)")

    # 4. Determine hardware hash
    if args.wildcard:
        hw_hash = "*"
        print("[3/4] Using WILDCARD hardware hash (valid on any machine).")
    elif args.hardware_hash:
        hw_hash = args.hardware_hash
        print(f"[3/4] Using custom hardware hash: {hw_hash}")
    else:
        print("[3/4] Reading current machine hardware fingerprint...")
        hw_hash = get_machine_fingerprint()
        print(f"      Current machine fingerprint: {hw_hash}")

    # 5. Sign license
    print(f"\n[4/4] Generating license for:")
    print(f"      Customer : {args.customer}")
    print(f"      Expires  : {args.expires}")
    print(f"      HW Hash  : {hw_hash if hw_hash != '*' else '* (wildcard)'}")
    license_key = sign_license(private_key, args.customer, args.expires, hw_hash)

    # 6. Output
    print("\n" + "=" * 60)
    if args.new_keypair:
        print("  PUBLIC KEY PEM (copy this into src/licensing/validator.py):")
    else:
        print("  PUBLIC KEY PEM (already embedded in src/licensing/validator.py,")
        print("  shown here for reference — no copy-paste needed):")
    print("=" * 60)
    print(public_pem.decode("utf-8"))

    print("=" * 60)
    print("  GENERATED LICENSE KEY (give to customer):")
    print("=" * 60)
    print(f"\n{license_key}\n")
    print("=" * 60)

    # Save license key to file for convenience
    license_file = Path("dev_tools/test_license.txt")
    license_file.write_text(license_key, encoding="utf-8")
    print(f"  License key also saved to: {license_file.resolve()}")
    print("=" * 60)
    print("\nDone. Remember:")
    if args.new_keypair:
        print("  1. Embed the public key PEM above into src/licensing/validator.py")
        print(f"  2. Delete or secure {private_key_path} from the build machine")
    else:
        print(f"  This license was signed with {private_key_path}, whose public")
        print("  key already matches what's embedded in validator.py — nothing to copy.")
    print("  3. Set TILEVISION_DEV_MODE unset (not '1') in production builds")
    print("     so signature verification and the wildcard bypass stay enforced.")


if __name__ == "__main__":
    main()
