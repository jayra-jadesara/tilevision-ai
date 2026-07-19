"""Tests for DINOv2 model path resolution."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.ai.model_paths as model_paths


def test_resolve_uses_bundled_dir_when_present(tmp_path, monkeypatch):
    model_dir = tmp_path / "dinov2-large"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"x")

    monkeypatch.delenv("TILEVISION_MODEL_DIR", raising=False)
    monkeypatch.delenv("TILEVISION_OFFLINE_MODEL", raising=False)
    monkeypatch.setattr(model_paths, "bundled_model_dir", lambda: model_dir)

    source, local_only = model_paths.resolve_dinov2_model_source()
    assert source == str(model_dir)
    assert local_only is True


def test_offline_mode_requires_local_weights(monkeypatch):
    monkeypatch.setenv("TILEVISION_OFFLINE_MODEL", "1")
    monkeypatch.delenv("TILEVISION_MODEL_DIR", raising=False)
    monkeypatch.setattr(model_paths, "bundled_model_dir", lambda: Path("/nonexistent/model"))

    try:
        model_paths.resolve_dinov2_model_source()
        assert False, "expected FileNotFoundError"
    except FileNotFoundError as exc:
        assert "TILEVISION_OFFLINE_MODEL" in str(exc)


def test_hub_download_when_no_local_copy(monkeypatch):
    monkeypatch.delenv("TILEVISION_MODEL_DIR", raising=False)
    monkeypatch.delenv("TILEVISION_OFFLINE_MODEL", raising=False)
    monkeypatch.setattr(model_paths, "bundled_model_dir", lambda: Path("/nonexistent/model"))

    source, local_only = model_paths.resolve_dinov2_model_source()
    assert source == model_paths.DEFAULT_MODEL_ID
    assert local_only is False


def test_frozen_runtime_uses_meipass(tmp_path, monkeypatch):
    model_dir = tmp_path / "model_weights" / "dinov2-large"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"x")

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    monkeypatch.delenv("TILEVISION_MODEL_DIR", raising=False)
    monkeypatch.delenv("TILEVISION_OFFLINE_MODEL", raising=False)

    source, local_only = model_paths.resolve_dinov2_model_source()
    assert source == str(model_dir)
    assert local_only is True
