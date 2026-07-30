"""Tests for fast multi-connection update downloader."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from src.utils import update_downloader as ud


class _RangeHandler(BaseHTTPRequestHandler):
    payload = b""

    def log_message(self, format, *args):  # noqa: A003
        return

    def do_HEAD(self):  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.payload)))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Disposition", 'attachment; filename="update.bin"')
        self.end_headers()

    def do_GET(self):  # noqa: N802
        data = self.payload
        range_header = self.headers.get("Range")
        if range_header and range_header.startswith("bytes="):
            spec = range_header.split("=", 1)[1]
            start_s, end_s = spec.split("-", 1)
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else len(data) - 1
            end = min(end, len(data) - 1)
            chunk = data[start : end + 1]
            self.send_response(206)
            self.send_header("Content-Length", str(len(chunk)))
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(chunk)
            return

        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture()
def range_server():
    payload = bytes((i * 17 + 3) % 256 for i in range(6 * 1024 * 1024))  # 6 MiB
    _RangeHandler.payload = payload
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    url = f"http://{host}:{port}/TileVisionAI-Setup-test.bin"
    try:
        yield url, payload
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_probe_remote_file_detects_ranges(range_server):
    url, payload = range_server
    info = ud.probe_remote_file(url)
    assert info.size == len(payload)
    assert info.accept_ranges is True
    assert info.filename in {"update.bin", "TileVisionAI-Setup-test.bin"}


def test_parallel_download_matches_payload(range_server, tmp_path):
    url, payload = range_server
    progress_events = []

    path = ud.download_update_file(
        url,
        tmp_path,
        connections=4,
        progress=lambda r, t, s: progress_events.append((r, t, s)),
    )
    assert path.exists()
    assert path.read_bytes() == payload
    assert progress_events
    assert progress_events[-1][0] == len(payload)


def test_single_stream_download(range_server, tmp_path, monkeypatch):
    url, payload = range_server
    monkeypatch.setattr(ud, "MIN_PARALLEL_FILE_BYTES", 10**12)
    path = ud.download_update_file(url, tmp_path, connections=8)
    assert path.read_bytes() == payload


def test_cancel_download(range_server, tmp_path):
    url, _payload = range_server
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(ud.DownloadCancelled):
        ud.download_update_file(url, tmp_path, cancel_event=cancel)


def test_format_helpers():
    assert "MB" in ud.format_bytes(5 * 1024 * 1024)
    assert "/s" in ud.format_speed(1500)


def test_eta_helpers_show_remaining_time():
    assert ud.eta_seconds(50, 100, 0) is None
    assert ud.eta_seconds(100, 100, 10.0) == 0.0
    assert ud.eta_seconds(25, 100, 25.0) == 3.0
    assert ud.format_eta(None) == "calculating…"
    assert ud.format_eta(12) == "~12s left"
    assert ud.format_eta(125) == "~2m 05s left"
    assert ud.format_eta(3725) == "~1h 02m left"


def test_default_connections_are_aggressive():
    assert ud.DEFAULT_CONNECTIONS >= 16
    assert ud.MAX_CONNECTIONS >= ud.DEFAULT_CONNECTIONS
    assert ud.READ_BUFFER_BYTES >= 1024 * 1024


def test_probe_captures_resolved_cdn_url(tmp_path):
    """Redirect target URL must be stored so workers skip GitHub throttle hops."""

    class _RedirectHandler(BaseHTTPRequestHandler):
        payload = bytes(range(256)) * 4096  # 1 MiB

        def log_message(self, format, *args):  # noqa: A003
            return

        def do_GET(self):  # noqa: N802
            if self.path.startswith("/release/"):
                self.send_response(302)
                self.send_header("Location", f"http://{self.headers.get('Host')}/cdn/file.dmg")
                self.end_headers()
                return
            range_header = self.headers.get("Range")
            data = self.payload
            if range_header and range_header.startswith("bytes="):
                spec = range_header.split("=", 1)[1]
                start_s, end_s = spec.split("-", 1)
                start = int(start_s) if start_s else 0
                end = int(end_s) if end_s else len(data) - 1
                chunk = data[start : end + 1]
                self.send_response(206)
                self.send_header("Content-Length", str(len(chunk)))
                self.send_header(
                    "Content-Range", f"bytes {start}-{end}/{len(data)}"
                )
                self.end_headers()
                self.wfile.write(chunk)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), _RedirectHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        info = ud.probe_remote_file(
            f"http://{host}:{port}/release/TileVisionAI-macOS-Intel.dmg"
        )
        assert info.accept_ranges is True
        assert info.size == len(_RedirectHandler.payload)
        assert "/cdn/file.dmg" in info.resolved_url
        assert "/release/" not in info.resolved_url

        path = ud.download_update_file(
            f"http://{host}:{port}/release/TileVisionAI-macOS-Intel.dmg",
            tmp_path,
            connections=4,
            filename="intel.dmg",
        )
        assert path.read_bytes() == _RedirectHandler.payload
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_range_retry_recovers_from_transient_failure(range_server, tmp_path, monkeypatch):
    url, payload = range_server
    original_request = ud._request
    fail_once = {"done": False}

    def request_with_one_fail(*args, **kwargs):
        if kwargs.get("headers") and "Range" in kwargs["headers"] and not fail_once["done"]:
            fail_once["done"] = True
            raise TimeoutError("simulated timeout")
        return original_request(*args, **kwargs)

    monkeypatch.setattr(ud, "_request", request_with_one_fail)
    path = ud.download_update_file(url, tmp_path, connections=2)
    assert path.read_bytes() == payload
    assert fail_once["done"] is True
