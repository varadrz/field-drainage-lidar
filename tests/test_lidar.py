"""
Unit Tests for Phase 5: Real LiDAR Processing & Roelens Ditch Extraction
Validates point cloud ingestion, ground return filtering, 1m DTM rasterization,
REA + Otsu ditch extraction, vector snapping, and culvert / driveway gap reconnection.
"""

from pathlib import Path
import numpy as np
import pytest

from scripts.generate_synthetic_lidar import generate_synthetic_lidar
from modules.lidar_processor import (
    process_lidar_point_cloud,
    read_lidar_data,
    parse_lidar_metadata,
    filter_ground_returns,
    rasterize_ground_points_to_dtm,
    LidarProcessingResult
)
from modules.drainage_extraction import (
    run_ditch_extraction,
    compute_relative_elevation_attribute,
    extract_candidate_ditch_mask,
    bridge_culvert_disconnections,
    DitchExtractionResult
)

SAMPLE_LAS = "data/sample/synthetic_indian_farm.las"


@pytest.fixture(scope="module")
def synthetic_las_file():
    """Ensures synthetic LAS point cloud exists."""
    p = Path(SAMPLE_LAS)
    if not p.exists():
        generate_synthetic_lidar(str(p))
    return str(p)


def test_synthetic_lidar_generation(synthetic_las_file):
    """Verifies that synthetic LiDAR file is created with accurate classes and CRS."""
    p = Path(synthetic_las_file)
    assert p.exists(), "Synthetic LAS file must exist"
    assert p.stat().st_size > 100_000, "LAS file size should be substantial"

    las_data, name = read_lidar_data(synthetic_las_file)
    assert len(las_data.x) >= 90_000, "Expected at least 90k points"
    classes = np.unique(np.asarray(las_data.classification))
    assert 2 in classes, "Bare-earth ground returns (Class 2) must be present"
    assert 5 in classes, "Vegetation returns (Class 5) must be present"


def test_lidar_metadata_parsing(synthetic_las_file):
    """Verifies metadata extraction, point density, and CRS display."""
    las_data, name = read_lidar_data(synthetic_las_file)
    meta = parse_lidar_metadata(las_data, name)

    assert meta.total_points >= 90_000
    assert meta.ground_points_count > 80_000
    assert meta.ground_fraction_pct > 80.0
    assert meta.point_density_sq_m > 3.0, f"Expected density > 3 pts/m², got {meta.point_density_sq_m}"
    assert meta.crs_epsg == 32644
    assert meta.has_ground_classification is True
    assert "Simulated Farm LiDAR" in meta.data_source_badge


def test_process_lidar_to_dtm(synthetic_las_file):
    """Verifies ground return filtering and 1m GeoTIFF DTM rasterization."""
    res = process_lidar_point_cloud(synthetic_las_file, output_dir="data/processed")

    assert isinstance(res, LidarProcessingResult)
    assert Path(res.dtm_tif_path).exists()
    assert res.elevation_array.shape == (150, 140)
    assert not np.any(np.isnan(res.elevation_array)), "Interpolated DTM must have zero NaNs"
    assert res.elevation_array.min() > 280.0
    assert res.elevation_array.max() < 290.0
    assert res.metadata.res_x == 1.0
    assert res.metadata.res_y == 1.0


def test_roelens_ditch_extraction(synthetic_las_file):
    """Verifies REA calculation, Otsu thresholding, thinning, and vectorization."""
    res = process_lidar_point_cloud(synthetic_las_file, output_dir="data/processed")
    ditch_res = run_ditch_extraction(
        elevation=res.elevation_array,
        metadata=res.metadata,
        ground_points_xyz=res.ground_points_xyz,
        rea_radius_m=3.0
    )

    assert isinstance(ditch_res, DitchExtractionResult)
    assert len(ditch_res.ditches) >= 2, "Expected at least 2 fragmented open ditch segments"
    assert ditch_res.total_ditch_length_m > 30.0, "Expected meaningful total ditch length"

    # GeoJSON validation
    assert ditch_res.ditches_geojson["type"] == "FeatureCollection"
    assert len(ditch_res.ditches_geojson["features"]) == len(ditch_res.ditches)
    for feat in ditch_res.ditches_geojson["features"]:
        assert feat["geometry"]["type"] == "LineString"
        assert len(feat["geometry"]["coordinates"]) >= 2


def test_culvert_driveway_reconnection(synthetic_las_file):
    """Verifies that the simulated driveway crossing gap is detected and reconnected."""
    res = process_lidar_point_cloud(synthetic_las_file, output_dir="data/processed")
    ditch_res = run_ditch_extraction(
        elevation=res.elevation_array,
        metadata=res.metadata,
        ground_points_xyz=res.ground_points_xyz,
        max_culvert_gap_m=50.0
    )

    assert len(ditch_res.culverts) >= 1, "At least one culvert bridge should be detected"
    
    # Check that a culvert bridges the driveway gap (around 5 to 12 meters)
    driveway_culvert = None
    for c in ditch_res.culverts:
        if 4.0 <= c.gap_length_m <= 15.0:
            driveway_culvert = c
            break

    assert driveway_culvert is not None, "Failed to identify the driveway culvert crossing gap"
    assert driveway_culvert.confidence_score > 0.80, f"Expected high confidence, got {driveway_culvert.confidence_score}"
    assert len(driveway_culvert.coords_native) == 2
    assert driveway_culvert.geojson_feature["geometry"]["type"] == "LineString"


def test_unclassified_fallback():
    """Verifies that point clouds without Class 2 ground returns trigger fallback gracefully."""
    import laspy
    header = laspy.LasHeader(point_format=2, version="1.2")
    header.offsets = [340000.0, 2050000.0, 280.0]
    header.scales = [0.001, 0.001, 0.001]

    las_data = laspy.LasData(header)
    las_data.x = np.array([340010.0, 340020.0, 340030.0])
    las_data.y = np.array([2050010.0, 2050020.0, 2050030.0])
    las_data.z = np.array([284.0, 284.5, 285.0])
    las_data.classification = np.array([0, 0, 0], dtype=np.uint8)  # Unclassified

    xyz, is_classified = filter_ground_returns(las_data)
    assert is_classified is False, "Should report unclassified fallback"
    assert len(xyz) == 3, "All points should be retained in unclassified mode"
