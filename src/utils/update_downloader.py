"""
Fast in-app update installer downloader.

GitHub Releases host large TileVision installers on Azure Blob behind
release-assets.githubusercontent.com. A single browser connection is often
throttled to a few KB/s in some regions (hours for a 1.6 GB Mac DMG).

This module downloads with:
  - One redirect resolve to the Azure CDN signed URL, then Range GETs there
  - Many parallel HTTP Range streams (default 16)
  - Per-chunk retries
  - Large read buffers (2 MiB)
  - Instantaneous + average speed for ETA
  - Single-stream fallback when Range is unavailable
"""

from __future__ import annotations

import logging
import ssl
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import unquote, urlparse

from src.version import APP_VERSION

logger = logging.getLogger("tilevision.update_downloader")

DEFAULT_CONNECTIONS = 16
MAX_CONNECTIONS = 24
READ_BUFFER_BYTES = 2 * 1024 * 1024  # 2 MiB
CONNECT_TIMEOUT_S = 30.0
READ_TIMEOUT_S = 180.0
MIN_PARALLEL_FILE_BYTES = 2 * 1024 * 1024  # parallel for anything multi-MB
RANGE_RETRIES = 3
USER_AGENT = f"TileVisionAI/{APP_VERSION} (update-downloader)"

ProgressCallback = Callable[[int, int, float], None]  # received, total, bytes_per_sec


class DownloadCancelled(Exception):
    """Raised when the user cancels an in-progress download."""


class DownloadError(RuntimeError):
    """Raised when the update installer cannot be downloaded."""


@dataclass(frozen=True, slots=True)
class RemoteFileInfo:
    url: str
    size: Optional[int]
    accept_ranges: bool
    filename: str
    # Final CDN URL after redirects (prefer this for parallel Range GETs).
    resolved_url: str = ""


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _request(
    url: str,
    *,
    method: str = "GET",
    headers: Optional[dict[str, str]] = None,
    timeout: float = CONNECT_TIMEOUT_S,
):
    merged = {
        "User-Agent": USER_AGENT,
        "Accept": "*/*",
        "Connection": "keep-alive",
    }
    if headers:
        merged.update(headers)
    req = urllib.request.Request(url, method=method, headers=merged)
    return urllib.request.urlopen(req, timeout=timeout, context=_ssl_context())


def filename_from_url(url: str) -> str:
    path = unquote(urlparse(url).path)
    name = Path(path).name
    if name:
        return name
    return "TileVisionAI-Update.bin"


def default_download_dir() -> Path:
    home = Path.home()
    downloads = home / "Downloads"
    if downloads.is_dir():
        return downloads
    return home


def probe_remote_file(url: str) -> RemoteFileInfo:
    """
    Discover Content-Length, Range support, and the final CDN URL.

    Prefer a 1-byte Range GET (works through GitHub→Azure redirects). Capture
    ``geturl()`` so parallel workers hit Azure Blob directly instead of
    re-walking the throttled GitHub redirect on every connection.
    """
    name = filename_from_url(url)
    try:
        with _request(
            url,
            headers={"Range": "bytes=0-0"},
            timeout=CONNECT_TIMEOUT_S,
        ) as response:
            disposition = response.headers.get("Content-Disposition") or ""
            if "filename=" in disposition:
                raw = disposition.split("filename=")[-1].strip().strip("\"'")
                if raw:
                    name = Path(unquote(raw)).name
            content_range = response.headers.get("Content-Range") or ""
            size: Optional[int] = None
            if "/" in content_range:
                total = content_range.rsplit("/", 1)[-1].strip()
                if total.isdigit():
                    size = int(total)
            accept = (
                response.status == 206
                or "bytes" in (response.headers.get("Accept-Ranges") or "").lower()
                or bool(content_range)
            )
            if size is None:
                length = response.headers.get("Content-Length")
                if length and length.isdigit() and response.status != 206:
                    size = int(length)
            response.read()  # drain the 1-byte body
            resolved = response.geturl() or url
            return RemoteFileInfo(
                url=url,
                size=size,
                accept_ranges=accept and size is not None and size > 0,
                filename=name,
                resolved_url=resolved,
            )
    except Exception as exc:
        logger.info("Range probe failed (%s) — trying HEAD", exc)

    try:
        with _request(url, method="HEAD", timeout=CONNECT_TIMEOUT_S) as response:
            length = response.headers.get("Content-Length")
            size = int(length) if length and length.isdigit() else None
            accept = "bytes" in (response.headers.get("Accept-Ranges") or "").lower()
            disposition = response.headers.get("Content-Disposition") or ""
            if "filename=" in disposition:
                raw = disposition.split("filename=")[-1].strip().strip("\"'")
                if raw:
                    name = Path(unquote(raw)).name
            resolved = response.geturl() or url
            return RemoteFileInfo(
                url=url,
                size=size,
                accept_ranges=accept and size is not None and size > 0,
                filename=name,
                resolved_url=resolved,
            )
    except Exception as exc:
        logger.warning("HEAD probe failed: %s", exc)
        return RemoteFileInfo(
            url=url,
            size=None,
            accept_ranges=False,
            filename=name,
            resolved_url=url,
        )


def _raise_if_cancelled(cancel_event: Optional[threading.Event]) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise DownloadCancelled("Download cancelled.")


def _download_single(
    url: str,
    dest: Path,
    *,
    expected_size: Optional[int],
    cancel_event: Optional[threading.Event],
    progress: Optional[ProgressCallback],
) -> Path:
    started = time.monotonic()
    received = 0
    with _request(url, timeout=READ_TIMEOUT_S) as response:
        length = response.headers.get("Content-Length")
        total = int(length) if length and length.isdigit() else (expected_size or 0)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as handle:
            while True:
                _raise_if_cancelled(cancel_event)
                chunk = response.read(READ_BUFFER_BYTES)
                if not chunk:
                    break
                handle.write(chunk)
                received += len(chunk)
                if progress is not None:
                    elapsed = max(time.monotonic() - started, 1e-3)
                    progress(received, total, received / elapsed)
    if expected_size and dest.stat().st_size != expected_size:
        raise DownloadError(
            f"Download incomplete: got {dest.stat().st_size} bytes, expected {expected_size}."
        )
    return dest


def _download_range_to_part(
    url: str,
    part_path: Path,
    start: int,
    end: int,
    *,
    cancel_event: Optional[threading.Event],
    progress_state: dict,
    progress: Optional[ProgressCallback],
) -> Path:
    headers = {"Range": f"bytes={start}-{end}"}
    expected = end - start + 1
    last_error: BaseException | None = None

    for attempt in range(1, RANGE_RETRIES + 1):
        _raise_if_cancelled(cancel_event)
        written = 0
        try:
            with _request(url, headers=headers, timeout=READ_TIMEOUT_S) as response:
                if response.status not in (200, 206):
                    raise DownloadError(
                        f"Range request failed with HTTP {response.status}."
                    )
                with part_path.open("wb") as handle:
                    while True:
                        _raise_if_cancelled(cancel_event)
                        chunk = response.read(READ_BUFFER_BYTES)
                        if not chunk:
                            break
                        handle.write(chunk)
                        written += len(chunk)
                        with progress_state["lock"]:
                            progress_state["received"] += len(chunk)
                            # Rolling window for snappier speed / ETA display.
                            now = time.monotonic()
                            samples = progress_state["samples"]
                            samples.append((now, progress_state["received"]))
                            while len(samples) > 1 and now - samples[0][0] > 2.5:
                                samples.pop(0)
                            received = progress_state["received"]
                            total = progress_state["total"]
                            if len(samples) >= 2:
                                dt = max(samples[-1][0] - samples[0][0], 1e-3)
                                db = max(samples[-1][1] - samples[0][1], 0)
                                speed = db / dt
                            else:
                                elapsed = max(now - progress_state["started"], 1e-3)
                                speed = received / elapsed
                        if progress is not None:
                            progress(received, total, speed)
            if written != expected:
                raise DownloadError(
                    f"Range {start}-{end} incomplete: got {written}, expected {expected}."
                )
            return part_path
        except DownloadCancelled:
            raise
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Range %s-%s attempt %s/%s failed: %s",
                start,
                end,
                attempt,
                RANGE_RETRIES,
                exc,
            )
            # Undo bytes counted from a partial failed attempt.
            if written > 0:
                with progress_state["lock"]:
                    progress_state["received"] = max(
                        0, progress_state["received"] - written
                    )
            try:
                if part_path.exists():
                    part_path.unlink()
            except OSError:
                pass
            if attempt < RANGE_RETRIES:
                time.sleep(0.4 * attempt)

    raise DownloadError(
        f"Range {start}-{end} failed after {RANGE_RETRIES} attempts: {last_error}"
    )


def installer_file_reusable(path: Path, expected_size: Optional[int]) -> bool:
    """True when ``path`` exists and matches the remote Content-Length."""
    if expected_size is None or int(expected_size) <= 0:
        return False
    try:
        candidate = Path(path)
        return candidate.is_file() and candidate.stat().st_size == int(expected_size)
    except OSError:
        return False


def resolve_cached_installer(
    url: str,
    dest_dir: Optional[Path] = None,
    *,
    preferred_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    Return a local installer that already matches the remote size, or None.

    Used so a completed download is not repeated when install/restart failed.
    """
    if not url or not str(url).strip():
        return None
    try:
        info = probe_remote_file(url)
    except Exception as exc:
        logger.warning("Could not probe update URL for cache reuse: %s", exc)
        return None

    out_name = info.filename or filename_from_url(url)
    out_dir = Path(dest_dir) if dest_dir is not None else default_download_dir()
    candidates: list[Path] = []
    if preferred_path is not None:
        candidates.append(Path(preferred_path))
    candidates.append(out_dir / out_name)

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if installer_file_reusable(candidate, info.size):
            logger.info(
                "Reusing cached update installer: %s (%s bytes)",
                candidate,
                info.size,
            )
            return candidate
    return None


def download_update_file(
    url: str,
    dest_dir: Optional[Path] = None,
    *,
    connections: int = DEFAULT_CONNECTIONS,
    cancel_event: Optional[threading.Event] = None,
    progress: Optional[ProgressCallback] = None,
    filename: Optional[str] = None,
) -> Path:
    """
    Download an update installer as fast as the network allows.

    Resolves GitHub→Azure CDN once, then uses parallel HTTP Range requests on
    the direct CDN URL. Returns the final file path.

    If the destination file already matches the remote size, it is reused
    (no second multi-GB download).
    """
    if not url or not str(url).strip():
        raise DownloadError("Missing download URL.")

    info = probe_remote_file(url)
    download_url = info.resolved_url or info.url or url
    out_dir = Path(dest_dir) if dest_dir is not None else default_download_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = filename or info.filename or filename_from_url(url)
    dest = out_dir / out_name
    partial = out_dir / f".{out_name}.partial"

    _raise_if_cancelled(cancel_event)

    if installer_file_reusable(dest, info.size):
        assert info.size is not None
        logger.info(
            "Skipping update download; reusable file already present: %s",
            dest,
        )
        if progress is not None:
            progress(info.size, info.size, float(info.size))
        return dest

    use_parallel = (
        info.accept_ranges
        and info.size is not None
        and info.size >= MIN_PARALLEL_FILE_BYTES
        and max(1, int(connections)) > 1
    )

    workers = max(2, min(int(connections), MAX_CONNECTIONS)) if use_parallel else 1
    logger.info(
        "Downloading update: url=%s cdn=%s size=%s parallel=%s connections=%s dest=%s",
        url,
        download_url != url,
        info.size,
        use_parallel,
        workers,
        dest,
    )

    part_paths: list[Path] = []
    try:
        if not use_parallel:
            result = _download_single(
                download_url,
                partial,
                expected_size=info.size,
                cancel_event=cancel_event,
                progress=progress,
            )
            result.replace(dest)
            return dest

        assert info.size is not None
        chunk = info.size // workers
        ranges: list[tuple[int, int, Path]] = []
        for i in range(workers):
            start = i * chunk
            end = info.size - 1 if i == workers - 1 else (i + 1) * chunk - 1
            if start <= end:
                part = out_dir / f".{out_name}.part{i}"
                part_paths.append(part)
                ranges.append((start, end, part))

        progress_state = {
            "received": 0,
            "total": info.size,
            "started": time.monotonic(),
            "lock": threading.Lock(),
            "samples": [],
        }

        errors: list[BaseException] = []
        with ThreadPoolExecutor(max_workers=len(ranges)) as pool:
            futures = [
                pool.submit(
                    _download_range_to_part,
                    download_url,
                    part,
                    start,
                    end,
                    cancel_event=cancel_event,
                    progress_state=progress_state,
                    progress=progress,
                )
                for start, end, part in ranges
            ]
            for fut in as_completed(futures):
                try:
                    fut.result()
                except DownloadCancelled:
                    for pending in futures:
                        pending.cancel()
                    raise
                except Exception as exc:
                    errors.append(exc)
                    if cancel_event is not None:
                        cancel_event.set()

        if errors:
            raise DownloadError(str(errors[0])) from errors[0]

        # Assemble parts in order into the final file.
        with partial.open("wb") as out:
            for _start, _end, part in ranges:
                with part.open("rb") as handle:
                    while True:
                        _raise_if_cancelled(cancel_event)
                        block = handle.read(READ_BUFFER_BYTES)
                        if not block:
                            break
                        out.write(block)

        if partial.stat().st_size != info.size:
            raise DownloadError(
                f"Download size mismatch: got {partial.stat().st_size}, expected {info.size}."
            )

        partial.replace(dest)
        if progress is not None:
            elapsed = max(time.monotonic() - progress_state["started"], 1e-3)
            progress(info.size, info.size, info.size / elapsed)
        logger.info("Update downloaded: %s (%.1f MB)", dest, info.size / 1e6)
        return dest
    except DownloadCancelled:
        _cleanup_paths([partial, *part_paths])
        raise
    except Exception:
        _cleanup_paths([partial, *part_paths])
        raise
    finally:
        _cleanup_paths(part_paths)


def _cleanup_paths(paths: list[Path]) -> None:
    for path in paths:
        try:
            if path.exists():
                path.unlink()
        except OSError:
            pass


def format_bytes(num: float) -> str:
    units = ["B", "KB", "MB", "GB"]
    value = float(num)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num:.0f} B"


def format_speed(bytes_per_sec: float) -> str:
    return f"{format_bytes(bytes_per_sec)}/s"


def eta_seconds(received: int, total: int, speed_bps: float) -> float | None:
    """Estimated seconds remaining at the current download speed."""
    if speed_bps <= 1.0 or total <= 0:
        return None
    remaining = max(0, int(total) - int(received))
    if remaining <= 0:
        return 0.0
    return remaining / float(speed_bps)


def format_eta(seconds: float | None) -> str:
    """Human-readable remaining time for update downloads."""
    if seconds is None:
        return "calculating…"
    total = max(0, int(round(seconds)))
    if total < 60:
        return f"~{total}s left"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"~{minutes}m {secs:02d}s left"
    hours, minutes = divmod(minutes, 60)
    return f"~{hours}h {minutes:02d}m left"
