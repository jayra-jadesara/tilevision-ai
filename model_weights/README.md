# Bundled model weights for offline installs.
#
# DINOv2 (required for search — production):
#   python scripts/download_dinov2_model.py
#   → model_weights/dinov2-large/
#
# SAM 2 tiny (experimental Precise Crop only — lab):
#   python scripts/download_sam2_model.py
#   → model_weights/sam2.1-hiera-tiny/
#
#   python scripts/download_sam2_onnx_model.py
#   → model_weights/sam2.1-hiera-tiny-onnx/   (Windows + Mac Intel + Apple Silicon)
#
# Installer bundling (lab): TILEVISION_BUNDLE_SAM2=auto
#   → ONNX on Windows + Mac Intel + Apple Silicon (identical package)
#   → Transformers safetensors only if TILEVISION_BUNDLE_SAM2_TRANSFORMERS=1
#
# Weight file contents are gitignored — do not commit the binaries.
