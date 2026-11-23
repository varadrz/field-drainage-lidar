"""
Unit and Integration Tests for Phase 2 Terrain Analysis Engine
Validates slope (degrees & percent grade), aspect, hillshade, vector contours,
and morphological depression detection.
"""

from pathlib import Path
import numpy as np
import pytest
import rasterio

from modules.data_loader import load_raster, RasterMetadata
from modules.terrain_analysis import (
    compute_slope,
    compute_aspect,
    compute_hillshade,
    extract_contours_geojson,
    detect_depressions,
    compute_terrain_summary,
    TerrainSummary
)


@pytest.fixture(scope="module")
def sample_farm_path() -> str:
    """Fixture providing path to synthetic farm raster."""
    path = Path("data/sample/synthetic_indian_farm.tif")
    if not path.exists():
        from scripts.generate_synthetic_farm import generate_synthetic_farm
        generate_synthetic_farm(str(path))
    return str(path)


def test_slope_known_plane():
    """Verify slope calculations on an exact tilted plane."""
    # Create 50x50 grid at 1.0m resolution
    # Z drops by 1.5m over 100m in X direction -> 1.5% slope
    y, x = np.mgrid[0:50, 0:50]
    res_x = 1.0
    res_y = 1.0
    plane = 100.0 - 0.015 * x

    slope_deg, slope_pct = compute_slope(plane, res_x, res_y)

    # Check interior cells (avoid boundary convolution padding)
    interior_pct = slope_pct[5:45, 5:45]
    interior_deg = slope_deg[5:45, 5:45]

    assert np.allclose(interior_pct, 1.5, atol=1e-3)
    expected_deg = np.degrees(np.arctan(0.015))
    assert np.allclose(interior_deg, expected_deg, atol=1e-3)


def test_aspect_cardinal_directions():
    """Verify aspect matches 0° (N), 90° (E), 180° (S), 270° (W)."""
    y, x = np.mgrid[0:30, 0:30]
    res = 1.0

    # 1. Plane dipping East: Z decreases as x increases (faces East = 90°)
    dem_east = 100.0 - 0.02 * x
    aspect_east = compute_aspect(dem_east, res, res)
    assert np.allclose(aspect_east[5:25, 5:25], 90.0, atol=1e-2)

    # 2. Plane dipping South: Z decreases as y increases (row index increases down) -> faces South = 180°
    dem_south = 100.0 - 0.02 * y
    aspect_south = compute_aspect(dem_south, res, res)
    assert np.allclose(aspect_south[5:25, 5:25], 180.0, atol=1e-2)

    # 3. Plane dipping North: Z increases as y increases (row 0 is North) -> faces North = 0°
    dem_north = 100.0 + 0.02 * y
    aspect_north = compute_aspect(dem_north, res, res)
    # 0° or 360°
    north_vals = aspect_north[5:25, 5:25]
    assert np.all((np.abs(north_vals) < 1e-2) | (np.abs(north_vals - 360.0) < 1e-2))

    # 4. Plane dipping West: Z increases as x increases -> faces West = 270°
    dem_west = 100.0 + 0.02 * x
    aspect_west = compute_aspect(dem_west, res, res)
    assert np.allclose(aspect_west[5:25, 5:25], 270.0, atol=1e-2)


def test_hillshade_properties(sample_farm_path):
    """Verify hillshade produces 0-255 uint8 shaded relief."""
    elevation, meta = load_raster(sample_farm_path)
    hs = compute_hillshade(elevation, meta.res_x, meta.res_y)

    assert isinstance(hs, np.ndarray)
    assert hs.dtype == np.uint8
    assert hs.shape == elevation.shape
    assert np.min(hs) >= 0
    assert np.max(hs) <= 255
    # Shaded relief on gently sloping farm (~0.8%) should have visible texture variation
    assert np.std(hs) > 3.0
    assert np.max(hs) > np.min(hs)


def test_contour_extraction(sample_farm_path):
    """Verify vector contour extraction generates valid GeoJSON FeatureCollection."""
    elevation, meta = load_raster(sample_farm_path)
    interval = 0.5
    geojson = extract_contours_geojson(elevation, meta, interval=interval)

    assert isinstance(geojson, dict)
    assert geojson.get("type") == "FeatureCollection"
    features = geojson.get("features", [])
    assert len(features) > 0

    first_feat = features[0]
    assert first_feat["geometry"]["type"] == "LineString"
    coords = first_feat["geometry"]["coordinates"]
    assert len(coords) >= 4  # min_length_pts
    assert "elevation" in first_feat["properties"]
    assert "elevation_str" in first_feat["properties"]


def test_depression_detection_on_synthetic_farm(sample_farm_path):
    """Verify Priority-Flood sink detection identifies the simulated depression."""
    elevation, meta = load_raster(sample_farm_path)
    dep_mask, dep_depth, stats = detect_depressions(elevation, meta, min_depth_m=0.05)

    assert isinstance(dep_mask, np.ndarray)
    assert dep_mask.dtype == bool
    assert isinstance(dep_depth, np.ndarray)

    # Sink was placed at row 95, col 75 with max depth 0.45m
    assert dep_mask[95, 75] == True
    assert 0.35 <= dep_depth[95, 75] <= 0.45

    # Statistics validation
    assert stats["depression_count"] >= 1
    assert stats["max_depth_m"] >= 0.35
    assert stats["total_sq_m"] > 100.0  # ~14m radius sink is ~600 m²
    assert stats["est_volume_m3"] > 20.0
    assert stats["total_acres"] > 0.0


def test_full_terrain_summary(sample_farm_path):
    """Verify integrated terrain summary compilation."""
    elevation, meta = load_raster(sample_farm_path)
    summary, slope_deg, slope_pct, aspect, hillshade, dep_depth = compute_terrain_summary(
        elevation, meta, min_depth_m=0.05
    )

    assert isinstance(summary, TerrainSummary)
    assert 0.5 < summary.mean_slope_pct < 5.0
    assert summary.max_slope_pct > summary.mean_slope_pct
    assert 0.0 <= summary.flat_area_pct <= 100.0
    assert summary.depression_count >= 1

    summary_dict = summary.to_dict()
    assert "Mean Slope (%)" in summary_dict
    assert "Max Slope (%)" in summary_dict
    assert "Total Depression Area" in summary_dict
    assert "Est. Stored Water Volume" in summary_dict
