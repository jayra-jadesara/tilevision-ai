"""Publish pricing/prices.json to GitHub via the Contents API (Option A)."""

from __future__ import annotations

import base64
import json
import logging
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from pricing_manager import PUBLISH_PATHS, serialize_prices_json
from vendor_settings import (
    DEFAULT_GITHUB_BRANCH,
    DEFAULT_GITHUB_REPO,
    get_github_branch,
    get_github_repo,
    get_github_token,
)

logger = logging.getLogger("tilevision.admin.github_pricing")


class GitHubPublishError(RuntimeError):
    """Raised when GitHub API calls fail."""


@dataclass(frozen=True, slots=True)
class PublishResult:
    repo: str
    branch: str
    commit_message: str
    updated_paths: tuple[str, ...]


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _api_request(
    method: str,
    url: str,
    token: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "TileVisionAI-Admin",
    }
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(
            request, timeout=timeout, context=_ssl_context()
        ) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body.strip() else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise GitHubPublishError(
            f"GitHub API {exc.code} for {url}: {detail[:500]}"
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise GitHubPublishError(f"GitHub request failed: {exc}") from exc


def _split_repo(repo: str) -> tuple[str, str]:
    text = repo.strip().strip("/")
    if "/" not in text:
        raise GitHubPublishError(
            f"Invalid repo '{repo}'. Use owner/name (e.g. jayra-jadesara/tilevision-ai)."
        )
    owner, name = text.split("/", 1)
    return owner, name


def get_file_metadata(
    repo_path: str,
    *,
    repo: str | None = None,
    branch: str | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    owner, name = _split_repo(repo or get_github_repo() or DEFAULT_GITHUB_REPO)
    branch_name = branch or get_github_branch() or DEFAULT_GITHUB_BRANCH
    auth = (token or get_github_token()).strip()
    if not auth:
        raise GitHubPublishError(
            "GitHub token is not configured. Add a Personal Access Token in the Pricing tab."
        )
    url = (
        f"https://api.github.com/repos/{owner}/{name}/contents/{repo_path}"
        f"?ref={branch_name}"
    )
    return _api_request("GET", url, auth)


def update_repo_file(
    repo_path: str,
    content_text: str,
    *,
    message: str,
    repo: str | None = None,
    branch: str | None = None,
    token: str | None = None,
    sha: str | None = None,
) -> dict[str, Any]:
    owner, name = _split_repo(repo or get_github_repo() or DEFAULT_GITHUB_REPO)
    branch_name = branch or get_github_branch() or DEFAULT_GITHUB_BRANCH
    auth = (token or get_github_token()).strip()
    if not auth:
        raise GitHubPublishError("GitHub token is not configured.")

    if sha is None:
        try:
            meta = get_file_metadata(
                repo_path, repo=f"{owner}/{name}", branch=branch_name, token=auth
            )
            sha = str(meta.get("sha", ""))
        except GitHubPublishError:
            sha = None

    encoded = base64.b64encode(content_text.encode("utf-8")).decode("ascii")
    body: dict[str, Any] = {
        "message": message,
        "content": encoded,
        "branch": branch_name,
    }
    if sha:
        body["sha"] = sha

    url = f"https://api.github.com/repos/{owner}/{name}/contents/{repo_path}"
    return _api_request("PUT", url, auth, payload=body)


def verify_github_token(
    *,
    token: str | None = None,
    repo: str | None = None,
) -> str:
    """Return the authenticated GitHub login name."""
    auth = (token or get_github_token()).strip()
    if not auth:
        raise GitHubPublishError("Enter a GitHub Personal Access Token first.")
    data = _api_request("GET", "https://api.github.com/user", auth)
    login = str(data.get("login", "")).strip()
    if not login:
        raise GitHubPublishError("Token accepted but login name was missing.")
    repo_name = repo or get_github_repo() or DEFAULT_GITHUB_REPO
    owner, name = _split_repo(repo_name)
    _api_request("GET", f"https://api.github.com/repos/{owner}/{name}", auth)
    return login


def publish_prices_to_github(
    data: dict[str, Any],
    *,
    message: str | None = None,
    repo: str | None = None,
    branch: str | None = None,
    token: str | None = None,
    paths: tuple[str, ...] = PUBLISH_PATHS,
) -> PublishResult:
    """Update live pricing JSON on GitHub (and bundled copy in repo)."""
    validated_text = serialize_prices_json(data)
    repo_name = repo or get_github_repo() or DEFAULT_GITHUB_REPO
    branch_name = branch or get_github_branch() or DEFAULT_GITHUB_BRANCH
    commit_message = message or (
        f"chore(pricing): update rates via TileVision Admin ({data.get('updated_at', '')})"
    )

    updated: list[str] = []
    for repo_path in paths:
        logger.info("Publishing %s on %s@%s", repo_path, repo_name, branch_name)
        update_repo_file(
            repo_path,
            validated_text,
            message=commit_message,
            repo=repo_name,
            branch=branch_name,
            token=token,
        )
        updated.append(repo_path)

    return PublishResult(
        repo=repo_name,
        branch=branch_name,
        commit_message=commit_message,
        updated_paths=tuple(updated),
    )
