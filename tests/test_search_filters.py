"""
Integration tests for Feature 8 (Filters) in SearchTilesUseCase.

Uses real SQLite + real FAISS (IndexFlatIP) with a fake embedder (no
torch/open_clip needed), verifying that filtering by brand/category/color
correctly narrows results without breaking the top_k contract or losing
the similarity ranking order.
"""

import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

faiss = pytest.importorskip("faiss")

from src.ai.vector_index import FaissIndexManager
from src.core.models import TileImage
from src.core.use_cases.search_tiles import SearchTilesUseCase
from src.data.db_context import DatabaseContext
from src.data.sqlite_repository import SQLiteImageRepository


class FakeEmbedder:
    """Deterministic embedder based on solid-color average, like other test files."""

    def load_model(self):
        pass

    def get_embedding(self, image_path: str):
        with Image.open(image_path) as img:
            img = img.convert("RGB").resize((4, 4))
            pixels = list(img.getdata())
            r = sum(p[0] for p in pixels) / (len(pixels) * 255.0)
            g = sum(p[1] for p in pixels) / (len(pixels) * 255.0)
            b = sum(p[2] for p in pixels) / (len(pixels) * 255.0)
            return [r, g, b, 1.0]


@pytest.fixture()
def env(tmp_path):
    db_context = DatabaseContext(str(tmp_path / "db" / "tiles.db"))
    repo = SQLiteImageRepository(db_context)
    embedder = FakeEmbedder()
    vector_index = FaissIndexManager(str(tmp_path / "index" / "tiles.index"), dimension=4)
    vector_index.load_index()

    use_case = SearchTilesUseCase(
        image_repository=repo,
        embedder=embedder,
        vector_index=vector_index,
        thumbnail_dir=str(tmp_path / "thumbs"),
    )

    return {"use_case": use_case, "repo": repo, "vector_index": vector_index, "tmp_path": tmp_path}


def _add_tile(env, name, color, brand, category, tile_color):
    path = env["tmp_path"] / name
    Image.new("RGB", (16, 16), color=color).save(path)

    tile = TileImage(
        file_path=str(path), file_name=name, file_size=1, dimensions="16x16",
        brand=brand, category=category, color=tile_color,
    )
    tile_id = env["repo"].add(tile)

    embedding = env["use_case"]._embedder.get_embedding(str(path))
    env["vector_index"].update_vectors([tile_id], [embedding], persist=False)
    env["repo"].mark_as_indexed(tile_id, True)
    return tile_id


def test_search_without_filters_returns_all_matches(env):
    _add_tile(env, "a.jpg", (200, 0, 0), "Kajaria", "Floor", "Red")
    _add_tile(env, "b.jpg", (0, 200, 0), "Somany", "Wall", "Green")

    query_path = env["tmp_path"] / "query.jpg"
    Image.new("RGB", (16, 16), color=(200, 0, 0)).save(query_path)

    results = env["use_case"].execute(str(query_path), top_k=20)
    assert len(results) == 2


def test_filter_by_brand_excludes_non_matching_tiles(env):
    _add_tile(env, "a.jpg", (200, 0, 0), "Kajaria", "Floor", "Red")
    _add_tile(env, "b.jpg", (190, 10, 10), "Somany", "Floor", "Red")  # visually similar, different brand

    query_path = env["tmp_path"] / "query.jpg"
    Image.new("RGB", (16, 16), color=(200, 0, 0)).save(query_path)

    results = env["use_case"].execute(str(query_path), top_k=20, filters={"brand": "Kajaria"})
    assert len(results) == 1
    assert results[0].tile.brand == "Kajaria"


def test_filter_by_multiple_fields_requires_all_to_match(env):
    _add_tile(env, "a.jpg", (200, 0, 0), "Kajaria", "Floor", "Red")
    _add_tile(env, "b.jpg", (200, 0, 0), "Kajaria", "Wall", "Red")  # same brand, different category

    query_path = env["tmp_path"] / "query.jpg"
    Image.new("RGB", (16, 16), color=(200, 0, 0)).save(query_path)

    results = env["use_case"].execute(
        str(query_path), top_k=20, filters={"brand": "Kajaria", "category": "Floor"}
    )
    assert len(results) == 1
    assert results[0].tile.category == "Floor"


def test_filter_is_case_insensitive(env):
    _add_tile(env, "a.jpg", (200, 0, 0), "Kajaria", "Floor", "Red")

    query_path = env["tmp_path"] / "query.jpg"
    Image.new("RGB", (16, 16), color=(200, 0, 0)).save(query_path)

    results = env["use_case"].execute(str(query_path), top_k=20, filters={"brand": "kajaria"})
    assert len(results) == 1


def test_no_matches_for_filter_returns_empty_list(env):
    _add_tile(env, "a.jpg", (200, 0, 0), "Kajaria", "Floor", "Red")

    query_path = env["tmp_path"] / "query.jpg"
    Image.new("RGB", (16, 16), color=(200, 0, 0)).save(query_path)

    results = env["use_case"].execute(str(query_path), top_k=20, filters={"brand": "NoSuchBrand"})
    assert results == []


def test_unknown_filter_keys_are_ignored_not_errors(env):
    _add_tile(env, "a.jpg", (200, 0, 0), "Kajaria", "Floor", "Red")

    query_path = env["tmp_path"] / "query.jpg"
    Image.new("RGB", (16, 16), color=(200, 0, 0)).save(query_path)

    # "material" isn't an allowed filter field — should be silently ignored.
    results = env["use_case"].execute(str(query_path), top_k=20, filters={"material": "Ceramic"})
    assert len(results) == 1


def test_get_filter_options_returns_distinct_sorted_values(env):
    _add_tile(env, "a.jpg", (200, 0, 0), "Kajaria", "Floor", "Red")
    _add_tile(env, "b.jpg", (0, 200, 0), "Somany", "Wall", "Green")
    _add_tile(env, "c.jpg", (0, 0, 200), "Kajaria", "Floor", "Blue")

    options = env["use_case"].get_filter_options()

    assert options["brand"] == ["Kajaria", "Somany"]
    assert options["category"] == ["Floor", "Wall"]
    assert set(options["color"]) == {"Red", "Green", "Blue"}
