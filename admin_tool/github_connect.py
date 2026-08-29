"""One-click GitHub connect for the vendor admin Pricing page."""

from __future__ import annotations

import re
import subprocess
import webbrowser
from typing import Optional

from github_pricing_publish import GitHubPublishError, verify_github_token
from vendor_settings import (
    DEFAULT_GITHUB_BRANCH,
    DEFAULT_GITHUB_REPO,
    get_github_login,
    get_github_token,
    save_vendor_settings,
)

# Classic PAT page with repo scope pre-selected for pricing JSON publish.
_GITHUB_TOKEN_URL = (
    "https://github.com/settings/tokens/new"
    "?description=TileVision+AI+Admin"
    "&scopes=repo"
)

_TOKEN_PATTERN = re.compile(r"^(gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})$")
_TOKEN_FIND = re.compile(r"(gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})")


def github_token_creation_url() -> str:
    return _GITHUB_TOKEN_URL


def open_github_token_page() -> None:
    webbrowser.open(_GITHUB_TOKEN_URL)


def try_github_cli_token() -> Optional[str]:
    """Use GitHub CLI token when `gh` is installed and logged in."""
    try:
        result = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True,
            text=True,
            timeout=12,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    token = (result.stdout or "").strip()
    if result.returncode != 0 or not _TOKEN_PATTERN.match(token):
        return None
    return token


def normalize_pasted_token(text: str) -> str:
    """Extract a single GitHub token from pasted text (ignores extra lines)."""
    raw = str(text or "").strip().strip('"').strip("'")
    if not raw:
        return ""
    if _TOKEN_PATTERN.match(raw):
        return raw
    match = _TOKEN_FIND.search(raw)
    if match and _TOKEN_PATTERN.match(match.group(1)):
        return match.group(1)
    return ""


def require_github_token(text: str) -> str:
    token = normalize_pasted_token(text)
    if not token:
        raise GitHubPublishError(
            "That does not look like a GitHub token.\n\n"
            "Copy only the token from GitHub (starts with ghp_ or github_pat_). "
            "Do not paste shell commands or other text."
        )
    return token


def save_github_connection(token: str) -> str:
    """Verify token, persist it, return GitHub login name."""
    clean = require_github_token(token)
    login = verify_github_token(token=clean)
    save_vendor_settings(github_token=clean, github_login=login)
    return login


def connect_github_automatically() -> tuple[str, str]:
    """
    Try GitHub CLI first, otherwise open browser for one-time token paste.

    Returns ``(login, source)`` where source is ``cli`` or ``browser``.
    """
    cli_token = try_github_cli_token()
    if cli_token:
        login = save_github_connection(cli_token)
        return login, "cli"

    open_github_token_page()
    raise GitHubPublishError(
        "GitHub opened in your browser.\n\n"
        "1. Click Generate token\n"
        "2. Copy the token\n"
        "3. Click Paste token here in TileVision Admin"
    )


def connection_status() -> tuple[bool, str]:
    token = get_github_token()
    login = get_github_login()
    clean = normalize_pasted_token(token)
    if token and not clean:
        return False, "Saved token is invalid — click Connect GitHub and paste again"
    if clean and login:
        return True, f"Connected as {login}"
    if clean:
        try:
            login = verify_github_token(token=clean)
            save_vendor_settings(github_token=clean, github_login=login)
            return True, f"Connected as {login}"
        except GitHubPublishError:
            return False, "Token saved but connection failed — click Connect GitHub"
    return False, "Not connected"


def publish_target_label() -> str:
    return f"{DEFAULT_GITHUB_REPO}  ·  branch {DEFAULT_GITHUB_BRANCH}  ·  pricing/prices.json"
