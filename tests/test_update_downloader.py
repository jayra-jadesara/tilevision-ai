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
