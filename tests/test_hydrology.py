"""
Unit and Integration Tests for Phase 3 WhiteboxTools Hydrological Engine
Validates depression filling, true depression depth calculation, D8 flow direction,
D8 flow accumulation, stream extraction, sub-basin delineation, and CRS preservation.
"""

from pathlib import Path
import numpy as np
import pytest
import rasterio

from modules.data_loader import load_raster
from modules.hydrology import run_hydrological_analysis, HydrologyResult


@pytest.fixture(scope="module")
def sample_farm_path() -> str:
    """Fixture providing path to synthetic farm raster."""
    path = Path("data/sample/synthetic_indian_farm.tif")
    if not path.exists():
        from scripts.generate_synthetic_farm import generate_synthetic_farm
        generate_synthetic_farm(str(path))
    return str(path)


@pytest.fixture(scope="module")
def hydrology_output(sample_farm_path) -> HydrologyResult:
    """Fixture running the WhiteboxTools hydrological engine once for module tests."""
    temp_dir = Path("data/processed/pytest_hydro")
    temp_dir.mkdir(parents=True, exist_ok=True)
    result = run_hydrological_analysis(
        dem_source=sample_farm_path,
        output_dir=str(temp_dir),
        flow_accum_threshold=80.0,
        min_waterlogging_depth_m=0.05
    )
    return result


def test_hydrological_pipeline_execution(hydrology_output):
    """Verify that all WhiteboxTools intermediate and product files exist on disk."""
    res = hydrology_output
    assert isinstance(res, HydrologyResult)

    assert Path(res.filled_dem_path).exists()
    assert Path(res.d8_pointer_path).exists()
    assert Path(res.flow_accum_path).exists()
    assert Path(res.streams_path).exists()
    assert Path(res.basins_path).exists()
    assert Path(res.depression_depth_path).exists()


def test_flow_accumulation_properties(hydrology_output):
    """Verify flow accumulation grid dimensions, non-emptiness, and concentration paths."""
    res = hydrology_output
    accum = res.flow_accum_array

    assert isinstance(accum, np.ndarray)
    assert accum.shape == (150, 140)
    assert np.nanmax(accum) > 100.0  # concentrated runoff in channels
    assert np.nanmin(accum) >= 1.0   # minimum contributing cell is 1 (self)


def test_simulated_depression_depth(hydrology_output):
    """Verify that the WhiteboxTools filled-raw differential isolates the simulated sink."""
    res = hydrology_output
    dep_depth = res.depression_depth_array

    assert isinstance(dep_depth, np.ndarray)
    assert dep_depth.shape == (150, 140)

    # Sink was placed at row 95, col 75 with max depth 0.45m
    assert dep_depth[95, 75] > 0.35
    assert 0.35 <= res.max_waterlogging_depth_m <= 0.60
    assert res.total_waterlogging_volume_m3 > 50.0
    assert res.waterlogging_area_sq_m > 100.0
    assert res.waterlogging_area_acres > 0.0


def test_stream_vector_geojson(hydrology_output):
    """Verify natural stream extraction produces valid GeoJSON LineString vectors."""
    res = hydrology_output
    geojson = res.streams_geojson

    assert isinstance(geojson, dict)
    assert geojson.get("type") == "FeatureCollection"
    features = geojson.get("features", [])
    assert len(features) > 0
    assert res.total_stream_length_m > 500.0
    assert res.stream_segments_count == len(features)

    first_feat = features[0]
    assert first_feat["geometry"]["type"] == "LineString"
    coords = first_feat["geometry"]["coordinates"]
    assert len(coords) >= 2


def test_crs_and_resolution_preservation(hydrology_output, sample_farm_path):
    """Verify that all WhiteboxTools outputs strictly preserve native EPSG:32644 CRS."""
    with rasterio.open(sample_farm_path) as src_raw:
        expected_crs = src_raw.crs
        expected_res = src_raw.res

    for path in [
        hydrology_output.filled_dem_path,
        hydrology_output.depression_depth_path,
        hydrology_output.flow_accum_path,
        hydrology_output.basins_path
    ]:
        with rasterio.open(path) as ds:
            assert ds.crs == expected_crs
            assert abs(ds.res[0] - expected_res[0]) < 1e-4
            assert abs(ds.res[1] - expected_res[1]) < 1e-4


def test_sub_basin_delineation(hydrology_output):
    """Verify sub-basin delineation yields multiple distinct contributing catchments."""
    res = hydrology_output
    assert res.sub_basin_count > 1
    assert isinstance(res.basins_geojson, dict)
    assert res.basins_geojson.get("type") == "FeatureCollection"


def test_hydrology_to_dict(hydrology_output):
    """Verify dictionary serialization of hydrological summary metrics."""
    res_dict = hydrology_output.to_dict()
    assert "Total Natural Runoff Channel Length" in res_dict
    assert "Deepest Waterlogging Point" in res_dict
    assert "Estimated Trapped Ponding Volume" in res_dict
    assert "Field Drainage Sub-Catchments" in res_dict
