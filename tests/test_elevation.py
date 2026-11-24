"""
Unit and Integration Tests for Phase 1 Elevation Processing
Validates synthetic farm generation, raster ingestion, NoData handling, and statistics calculations.
"""

import io
from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from scripts.generate_synthetic_farm import generate_synthetic_farm
from modules.data_loader import load_raster, RasterMetadata
from modules.elevation_analysis import compute_elevation_stats, ElevationStats
from components.maps import create_elevation_map


@pytest.fixture(scope="module")
def synthetic_farm_path(tmp_path_factory) -> str:
    """Fixture providing a generated synthetic farm GeoTIFF."""
    temp_dir = tmp_path_factory.mktemp("sample_data")
    out_file = temp_dir / "test_synthetic_farm.tif"
    generate_synthetic_farm(str(out_file))
    return str(out_file)


def test_synthetic_farm_generation(synthetic_farm_path):
    """Verify that the synthetic farm script creates a valid GeoTIFF with expected metadata."""
    assert Path(synthetic_farm_path).exists()
    
    with rasterio.open(synthetic_farm_path) as src:
        assert src.count == 1
        assert src.width == 140
        assert src.height == 150
        assert src.crs.to_epsg() == 32644
        assert abs(src.res[0] - 1.0) < 1e-4
        assert abs(src.res[1] - 1.0) < 1e-4
        assert src.nodata == -9999.0


def test_raster_loading(synthetic_farm_path):
    """Test data_loader on the synthetic farm raster."""
    elevation, meta = load_raster(synthetic_farm_path)
    
    assert isinstance(elevation, np.ndarray)
    assert elevation.shape == (150, 140)
    assert elevation.dtype == np.float32
    assert isinstance(meta, RasterMetadata)
    assert meta.crs_epsg == 32644
    assert meta.is_projected is True
    assert meta.is_geographic is False
    assert meta.width == 140
    assert meta.height == 150


def test_elevation_statistics(synthetic_farm_path):
    """Test elevation statistics calculation."""
    elevation, meta = load_raster(synthetic_farm_path)
    stats = compute_elevation_stats(elevation, meta)
    
    assert isinstance(stats, ElevationStats)
    # The base elevation is ~285.5m, with micro-depressions dipping to ~283.5m and bunds rising to ~286m
    assert 280.0 < stats.min_elevation < 285.0
    assert 285.0 < stats.max_elevation < 290.0
    assert stats.min_elevation < stats.mean_elevation < stats.max_elevation
    assert stats.elevation_range > 1.0
    assert stats.valid_pixels == 150 * 140
    assert stats.nodata_pixels == 0
    # Area should be approx 21,000 m² (~5.18 acres)
    assert 4.5 < stats.area_acres < 5.5
    assert 1.8 < stats.area_ha < 2.3
    assert stats.spatial_resolution_m == pytest.approx(1.0, 0.05)


def test_nodata_and_outlier_handling(tmp_path):
    """Test that NoData values and extreme outliers are masked to np.nan."""
    test_tif = tmp_path / "test_nodata.tif"
    data = np.array([
        [100.0, 102.0, -9999.0],
        [101.0, -9999.0, 103.0],
        [-32768.0, 104.0, 105.0]
    ], dtype=np.float32)

    transform = Affine(1.0, 0.0, 340000.0, 0.0, -1.0, 2050000.0)
    with rasterio.open(
        test_tif, "w",
        driver="GTiff", height=3, width=3, count=1,
        dtype=rasterio.float32, crs="EPSG:32644",
        transform=transform, nodata=-9999.0
    ) as dst:
        dst.write(data, 1)

    elevation, meta = load_raster(str(test_tif))
    assert np.isnan(elevation[0, 2])
    assert np.isnan(elevation[1, 1])
    assert np.isnan(elevation[2, 0])  # outlier < -1000 also masked

    stats = compute_elevation_stats(elevation, meta)
    assert stats.valid_pixels == 6
    assert stats.nodata_pixels == 3
    assert stats.min_elevation == 100.0
    assert stats.max_elevation == 105.0
    assert stats.mean_elevation == pytest.approx(102.5, 0.1)


def test_empty_raster_raises():
    """Verify that an all-NaN raster raises a meaningful ValueError."""
    dummy_meta = RasterMetadata(
        width=10, height=10, count=1, crs_wkt="", crs_epsg=32644,
        is_projected=True, is_geographic=False, units="meters",
        res_x=1.0, res_y=1.0, bounds_native=(0, 0, 10, 10),
        bounds_wgs84=(78.0, 18.0, 78.1, 18.1), nodata_value=-9999.0,
        transform=(0, 1, 0, 0, 0, -1)
    )
    all_nan = np.full((10, 10), np.nan, dtype=np.float32)
    with pytest.raises(ValueError, match="all raster cells are NoData/NaN"):
        compute_elevation_stats(all_nan, dummy_meta)


def test_bytes_io_loading(synthetic_farm_path):
    """Test loading raster from raw bytes / in-memory file buffer."""
    with open(synthetic_farm_path, "rb") as f:
        raw_bytes = f.read()

    # Test raw bytes
    elev_bytes, meta_bytes = load_raster(raw_bytes)
    assert elev_bytes.shape == (150, 140)

    # Test file-like BytesIO
    buf = io.BytesIO(raw_bytes)
    elev_buf, meta_buf = load_raster(buf)
    assert elev_buf.shape == (150, 140)
    assert meta_buf.crs_epsg == 32644


def test_create_elevation_map(synthetic_farm_path):
    """Test generating interactive Folium map."""
    elevation, meta = load_raster(synthetic_farm_path)
    stats = compute_elevation_stats(elevation, meta)
    
    fmap = create_elevation_map(elevation, meta, stats, colormap_name="terrain", opacity=0.8)
    assert fmap is not None
    html_repr = fmap.get_root().render()
    assert "Elevation" in html_repr and "DTM" in html_repr
    assert "Field Boundary" in html_repr


def test_stats_to_dict(synthetic_farm_path):
    """Test dictionary serialization of ElevationStats."""
    elevation, meta = load_raster(synthetic_farm_path)
    stats = compute_elevation_stats(elevation, meta)
    stats_dict = stats.to_dict()
    assert "Min Elevation (m)" in stats_dict
    assert "Max Elevation (m)" in stats_dict
    assert "Field Area (Acres)" in stats_dict
    assert "Spatial Resolution" in stats_dict
