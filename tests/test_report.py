"""
Unit and Integration Tests for Phase 6 PDF Report Generator & GIS Exporters.
Verifies PDF generation, layout compilation, non-empty binary output,
and validity of GeoJSON vectors for external QGIS ingestion.
"""

from pathlib import Path
import json
import pytest
import numpy as np

from modules.data_loader import load_raster
from modules.elevation_analysis import compute_elevation_stats
from modules.terrain_analysis import compute_terrain_summary
from modules.hydrology import run_hydrological_analysis
from modules.drainage_recommendation import generate_drainage_plan
from modules.drainage_extraction import run_ditch_extraction
from modules.report_generator import generate_drainage_pdf_report


@pytest.fixture(scope="module")
def sample_analysis_pipeline():
    """Runs end-to-end analytical pipeline on synthetic farm to feed report generator."""
    sample_dem = Path("data/sample/synthetic_indian_farm.tif").resolve()
    if not sample_dem.exists():
        from scripts.generate_synthetic_farm import generate_synthetic_farm_geotiff
        generate_synthetic_farm_geotiff(sample_dem)

    dem_arr, meta = load_raster(sample_dem)
    elev_stats = compute_elevation_stats(dem_arr, meta)
    terrain_summary, *_ = compute_terrain_summary(dem_arr, meta)
    hydro_result = run_hydrological_analysis(
        sample_dem,
        output_dir="data/processed/test_report_hydro",
        flow_accum_threshold=80.0,
        min_waterlogging_depth_m=0.05
    )
    drainage_plan = generate_drainage_plan(
        dem_source=sample_dem,
        hydrology_result=hydro_result,
        output_dir="data/processed/test_report_hydro",
        min_waterlogging_depth_m=0.05,
        min_waterlogging_area_m2=10.0,
        min_bed_slope_pct=0.20
    )
    ditch_result = run_ditch_extraction(
        dem_arr,
        meta,
        rea_radius_m=3.0
    )

    return {
        "dem_arr": dem_arr,
        "meta": meta,
        "elev_stats": elev_stats,
        "terrain_summary": terrain_summary,
        "hydro_result": hydro_result,
        "drainage_plan": drainage_plan,
        "ditch_result": ditch_result
    }


def test_pdf_report_generation(sample_analysis_pipeline, tmp_path):
    """Verify that generate_drainage_pdf_report compiles a valid, non-empty PDF file."""
    data = sample_analysis_pipeline
    pdf_out = tmp_path / "test_drainage_plan_report.pdf"

    generated_path = generate_drainage_pdf_report(
        output_pdf_path=pdf_out,
        dem_array=data["dem_arr"],
        metadata=data["meta"],
        elev_stats=data["elev_stats"],
        terrain_summary=data["terrain_summary"],
        hydro_result=data["hydro_result"],
        drainage_plan=data["drainage_plan"],
        ditch_result=data["ditch_result"],
        data_source_type="Synthetic Indian Farm Benchmark",
        project_name="Phase 6 Automated Drainage Plan",
        field_plot_id="Plot #TEST-2026"
    )

    assert generated_path.exists(), "PDF report was not created on disk."
    assert generated_path.stat().st_size > 30000, "PDF report is unexpectedly small or incomplete."

    # Validate standard PDF file header
    with open(generated_path, "rb") as f:
        header = f.read(5)
        assert header == b"%PDF-", f"Invalid PDF signature: {header}"


def test_geojson_export_validity(sample_analysis_pipeline):
    """Verify that exportable GeoJSON structures have standard compliant specifications."""
    data = sample_analysis_pipeline
    drainage_plan = data["drainage_plan"]
    ditch_result = data["ditch_result"]

    # 1. Routes GeoJSON
    routes_json = drainage_plan.routes_geojson
    assert routes_json["type"] == "FeatureCollection"
    assert len(routes_json["features"]) > 0
    first_route = routes_json["features"][0]
    assert first_route["geometry"]["type"] == "LineString"
    assert "length_m" in first_route["properties"]
    assert "bed_slope_pct" in first_route["properties"]

    # 2. Sinks GeoJSON
    sinks_json = drainage_plan.sinks_geojson
    assert sinks_json["type"] == "FeatureCollection"
    assert len(sinks_json["features"]) > 0
    first_sink = sinks_json["features"][0]
    assert first_sink["geometry"]["type"] == "Point"
    assert "max_depth_cm" in first_sink["properties"]
    assert "volume_m3" in first_sink["properties"]

    # 3. Ditches GeoJSON
    ditches_json = ditch_result.ditches_geojson
    assert ditches_json["type"] == "FeatureCollection"
    assert "features" in ditches_json


def test_processed_geotiff_export_validity(sample_analysis_pipeline):
    """Verify that depression depth GeoTIFF exists and has valid geospatial metadata."""
    import rasterio
    data = sample_analysis_pipeline
    dep_path = data["hydro_result"].depression_depth_path

    assert Path(dep_path).exists(), "Depression depth GeoTIFF does not exist."
    with rasterio.open(dep_path) as src:
        assert src.count == 1
        assert src.crs is not None
        assert src.transform is not None
        data_arr = src.read(1)
        assert data_arr.shape == data["dem_arr"].shape
