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
# Installer bundling (lab): TILEVISION_BUNDLE_SAM2=auto
#   → Windows + Mac Apple Silicon include this folder; Mac Intel skips it.
#
# Weight file contents are gitignored — do not commit the binaries.
