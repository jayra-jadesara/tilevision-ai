"""
Integration tests for Feature 8 (Filters) in SearchTilesUseCase.
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
from src.utils.image_utils import compute_sha256, compute_dhash
from tests.fake_ai import FakeEmbedder, FakeFeatureExtractor, make_tile_features


@pytest.fixture()
def env(tmp_path):
    db_context = DatabaseContext(str(tmp_path / "db" / "tiles.db"))
    repo = SQLiteImageRepository(db_context)
    embedder = FakeEmbedder()
    feature_extractor = FakeFeatureExtractor(embedder=embedder)
    vector_index = FaissIndexManager(str(tmp_path / "index" / "tiles.index"), dimension=4)
    vector_index.load_index()

    use_case = SearchTilesUseCase(
        image_repository=repo,
        feature_extractor=feature_extractor,
        vector_index=vector_index,
        thumbnail_dir=str(tmp_path / "thumbs"),
    )

    return {
        "use_case": use_case,
        "repo": repo,
        "embedder": embedder,
        "vector_index": vector_index,
        "tmp_path": tmp_path,
    }


def _add_tile(env, name, color, brand, category, tile_color):
    path = env["tmp_path"] / name
    Image.new("RGB", (16, 16), color=color).save(path)

    embedding = env["embedder"].get_embedding(str(path))
    features = make_tile_features(embedding)

    tile = TileImage(
        file_path=str(path),
        file_name=name,
        file_size=1,
        dimensions="16x16",
        brand=brand,
        category=category,
        color=tile_color,
        sha256_hash=compute_sha256(path),
        perceptual_hash=compute_dhash(path),
        features=features,
    )
    tile_id = env["repo"].add(tile)

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
    _add_tile(env, "b.jpg", (190, 10, 10), "Somany", "Floor", "Red")

    query_path = env["tmp_path"] / "query.jpg"
    Image.new("RGB", (16, 16), color=(200, 0, 0)).save(query_path)

    results = env["use_case"].execute(str(query_path), top_k=20, filters={"brand": "Kajaria"})
    assert len(results) == 1
    assert results[0].tile.brand == "Kajaria"


def test_filter_by_multiple_fields_requires_all_to_match(env):
    _add_tile(env, "a.jpg", (200, 0, 0), "Kajaria", "Floor", "Red")
    _add_tile(env, "b.jpg", (200, 0, 0), "Kajaria", "Wall", "Red")

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
