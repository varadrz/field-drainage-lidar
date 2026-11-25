"""
Unit Tests for Phase 4: Drainage Recommendation Engine
Validates sink identification, candidate outlet detection, cost surface generation,
least-cost gravity route routing, bed invert monotonicity, earthwork volume calculation,
and GeoJSON vector export.
"""

import os
from pathlib import Path
import numpy as np
import pytest

from modules.data_loader import load_raster
from modules.hydrology import run_hydrological_analysis
from modules.drainage_recommendation import (
    identify_priority_sinks,
    identify_candidate_outlets,
    build_cost_surface,
    analyze_route_engineering,
    generate_drainage_plan,
    WaterloggingSink,
    CandidateOutlet,
    DrainageRoute,
    DrainagePlanResult
)

SAMPLE_TIF = "data/sample/synthetic_indian_farm.tif"


@pytest.fixture(scope="module")
def sample_data():
    """Loads elevation and runs hydrology once for the test suite."""
    dem, meta = load_raster(SAMPLE_TIF)
    hydro = run_hydrological_analysis(SAMPLE_TIF, output_dir="data/processed")
    return dem, meta, hydro


def test_identify_priority_sinks(sample_data):
    """Verifies that the primary waterlogging depression is identified with accurate dimensions."""
    dem, meta, hydro = sample_data
    sinks = identify_priority_sinks(
        depression_depth=hydro.depression_depth_array,
        metadata=meta,
        min_depth_m=0.05,
        min_area_m2=10.0,
        min_volume_m3=0.5
    )

    assert len(sinks) >= 1, "At least one priority sink should be detected"
    primary_sink = sinks[0]

    # Check sink attributes
    assert primary_sink.sink_id == 1
    assert primary_sink.area_m2 > 500.0, f"Expected large sink area, got {primary_sink.area_m2}"
    assert primary_sink.volume_m3 > 50.0, f"Expected large trapped volume, got {primary_sink.volume_m3}"
    assert primary_sink.max_depth_m >= 0.40, f"Expected deep sink >= 0.40m, got {primary_sink.max_depth_m}"
    assert primary_sink.severity == "Severe Ponding"

    # Check pit coordinates (synthetic farm depression is centered at row 95, col 75)
    pit_r, pit_c = primary_sink.pit_rc
    assert abs(pit_r - 95) <= 5, f"Expected pit row near 95, got {pit_r}"
    assert abs(pit_c - 75) <= 5, f"Expected pit col near 75, got {pit_c}"

    # Check GeoJSON structure
    geom = primary_sink.geojson_feature["geometry"]
    assert geom["type"] == "Point"
    assert len(geom["coordinates"]) == 2


def test_identify_candidate_outlets(sample_data):
    """Verifies that candidate outlets are found along perimeter with lowest elevation."""
    dem, meta, hydro = sample_data
    outlets = identify_candidate_outlets(
        elevation=dem,
        metadata=meta,
        streams_geojson=hydro.streams_geojson,
        max_outlets=4
    )

    assert len(outlets) >= 1, "At least one candidate outlet should be detected"
    
    # Check that outlets have valid elevations and coordinates
    for out in outlets:
        assert isinstance(out, CandidateOutlet)
        assert out.elevation_m > 280.0
        assert out.coord_wgs84[0] != 0.0
        assert out.coord_wgs84[1] != 0.0
        assert out.geojson_feature["geometry"]["type"] == "Point"

    # Verify that the eastern boundary opening (col ~138-139, row ~108-112) is found
    found_east_gap = any(abs(out.rc[1] - 139) <= 2 and abs(out.rc[0] - 110) <= 5 for out in outlets)
    assert found_east_gap, "Candidate outlets should include the eastern boundary drainage gap"


def test_build_cost_surface(sample_data):
    """Verifies that the anisotropic cost surface is strictly positive and finite for valid pixels."""
    dem, meta, hydro = sample_data
    cost = build_cost_surface(dem, flow_accum=hydro.flow_accum_array)

    valid_mask = ~np.isnan(dem)
    assert np.all(cost[valid_mask] > 0.0), "All valid terrain cells must have positive cost"
    assert np.all(np.isfinite(cost[valid_mask])), "All valid terrain cells must have finite cost"
    assert np.all(np.isinf(cost[~valid_mask])), "NoData cells must have infinite cost"


def test_route_engineering_monotonic_grade(sample_data):
    """Verifies that engineered ditch invert strictly decreases monotonically with zero uphill backwater."""
    dem, meta, hydro = sample_data
    sinks = identify_priority_sinks(hydro.depression_depth_array, meta)
    outlets = identify_candidate_outlets(dem, meta)
    
    # Path from primary sink to first outlet
    cost = build_cost_surface(dem, flow_accum=hydro.flow_accum_array)
    import skimage.graph
    indices, _ = skimage.graph.route_through_array(
        cost, sinks[0].pit_rc, outlets[0].rc, fully_connected=True, geometric=True
    )

    route = analyze_route_engineering(
        path_rc=indices,
        elevation=dem,
        metadata=meta,
        route_id="Test-Route",
        sink_id=1,
        outlet_id=1,
        min_bed_slope_pct=0.2
    )

    # 1. Monotonic bed invert: all steps along the bed must be downhill
    bed = np.array(route.bed_elev_profile_m)
    bed_diffs = np.diff(bed)
    assert np.all(bed_diffs <= 1e-6), "Ditch bed invert must decrease monotonically (no uphill flow)"

    # 2. Minimum grade enforced
    assert route.bed_slope_pct >= 0.20, f"Expected bed grade >= 0.2%, got {route.bed_slope_pct}"

    # 3. Positive length and reasonable excavation depth
    assert route.length_m > 10.0, f"Expected meaningful route length, got {route.length_m}"
    assert route.max_cut_depth_m > 0.0, "Expected non-zero excavation depth"
    assert route.max_cut_depth_m < 2.0, f"Expected realistic trench cut < 2.0m, got {route.max_cut_depth_m}"
    assert route.earthwork_vol_m3 > 0.0, "Expected positive excavation earthwork volume"

    # 4. GeoJSON structure
    geom = route.geojson_feature["geometry"]
    assert geom["type"] == "LineString"
    assert len(geom["coordinates"]) >= 2


def test_generate_drainage_plan_pipeline(sample_data):
    """End-to-end integration test of the Phase 4 drainage recommendation pipeline."""
    dem, meta, hydro = sample_data
    plan = generate_drainage_plan(
        dem_source=SAMPLE_TIF,
        hydrology_result=hydro,
        min_waterlogging_depth_m=0.05,
        min_waterlogging_area_m2=10.0,
        min_bed_slope_pct=0.2
    )

    assert isinstance(plan, DrainagePlanResult)
    assert len(plan.priority_sinks) >= 1
    assert len(plan.candidate_outlets) >= 1
    assert len(plan.routes) >= 1
    assert plan.best_route is not None
    assert plan.best_route.is_recommended is True
    assert plan.best_route.feasibility_rank in ["Recommended (Optimal)", "Moderate Excavation"]

    # Verify GeoJSON bundles
    assert plan.routes_geojson["type"] == "FeatureCollection"
    assert len(plan.routes_geojson["features"]) == len(plan.routes)
    assert plan.sinks_geojson["type"] == "FeatureCollection"
    assert plan.outlets_geojson["type"] == "FeatureCollection"

    # Verify summary dict
    summary = plan.to_dict()
    assert "Evaluated Routes" in summary
    assert "Best Route Length" in summary
    assert "Total Excavation Volume" in summary
