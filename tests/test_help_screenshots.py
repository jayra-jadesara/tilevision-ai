"""Help dialog screenshot assets."""

from pathlib import Path

_HELP_DIR = Path(__file__).resolve().parents[1] / "src" / "resources" / "help"

_EXPECTED = (
    "step1_choose_folder.png",
    "step2_index_images.png",
    "step3_upload_customer_image.png",
    "step4_view_similar_tiles.png",
    "step5_double_click_to_open.png",
)


def test_help_step_screenshots_exist_and_are_images():
    from PIL import Image

    for name in _EXPECTED:
        path = _HELP_DIR / name
        assert path.is_file(), f"Missing help screenshot: {path}"
        assert path.stat().st_size > 10_000, f"Screenshot too small: {path}"
        with Image.open(path) as img:
            assert img.size[0] >= 600
            assert img.size[1] >= 400
