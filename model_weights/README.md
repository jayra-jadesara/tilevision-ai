# Bundled model weights for offline installs.
#
# DINOv2 (required for visual search — production):
#   python scripts/download_dinov2_model.py
#   → model_weights/dinov2-large/
#
# CLIP ViT-B/32 (optional — description / text search; MIT license):
#   python scripts/download_clip_model.py
#   → model_weights/clip-vit-base-patch32/
#   Ship as a *separate* download (~580 MB). Lazy first-use also downloads
#   here so the DINOv2 installer stays under GitHub's 2 GiB asset limit.
#
# SAM 2 tiny (experimental Precise Crop only — lab):
#   python scripts/download_sam2_model.py
#   → model_weights/sam2.1-hiera-tiny/
#
#   python scripts/download_sam2_onnx_model.py
#   → model_weights/sam2.1-hiera-tiny-onnx/   (Mac Intel + Windows CPU)
#
# Installer bundling (lab): TILEVISION_BUNDLE_SAM2=auto
#   → ONNX on Windows + Mac Intel + Apple Silicon
#   → Transformers safetensors on Windows + Apple Silicon (not Mac Intel)
#
# Weight file contents are gitignored — do not commit the binaries.
