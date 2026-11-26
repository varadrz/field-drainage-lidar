"""
Field Drainage Using LiDAR - Phase 3 Web Application
Intelligent LiDAR-Based Agricultural Field Drainage Planning System
Interactive Streamlit Dashboard & WhiteboxTools Hydrological Engine
Optimized for high-performance caching and zero-rerun interactive mapping.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import streamlit as st
from streamlit_folium import st_folium

from modules.data_loader import load_raster, RasterMetadata
from modules.elevation_analysis import compute_elevation_stats, ElevationStats
from modules.terrain_analysis import compute_terrain_summary, extract_contours_geojson, TerrainSummary
from modules.hydrology import run_hydrological_analysis, HydrologyResult
from modules.drainage_recommendation import generate_drainage_plan, DrainagePlanResult, DrainageRoute
from modules.lidar_processor import process_lidar_point_cloud, LidarProcessingResult, LidarMetadata
from modules.drainage_extraction import run_ditch_extraction, DitchExtractionResult
from components.maps import create_terrain_map
from components.elevation_profile import create_trench_profile_figure
from components.charts import create_3d_terrain_figure

# Configure Streamlit Page
st.set_page_config(
    page_title="Field Drainage Using LiDAR | LiDAR Processing & Ditch Extraction",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Minimalist Dark Agronomic Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #2ecc71;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-size: 1.02rem;
        color: #94a3b8;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #14181f;
        border-radius: 8px;
        padding: 12px 14px;
        border: 1px solid #1f2937;
        border-left: 5px solid #2ecc71;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 8px;
    }
    .metric-card-alert {
        background-color: #1a1510;
        border-radius: 8px;
        padding: 12px 14px;
        border: 1px solid #372810;
        border-left: 5px solid #f59e0b;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 8px;
    }
    .metric-card-sink {
        background-color: #0c1829;
        border-radius: 8px;
        padding: 12px 14px;
        border: 1px solid #143254;
        border-left: 5px solid #38bdf8;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 8px;
    }
    .metric-card-stream {
        background-color: #0d1f1d;
        border-radius: 8px;
        padding: 12px 14px;
        border: 1px solid #134e48;
        border-left: 5px solid #00f5d4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 8px;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #f0f4f8;
    }
    .metric-value-alert {
        font-size: 1.35rem;
        font-weight: 700;
        color: #fbbf24;
    }
    .metric-value-sink {
        font-size: 1.35rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-value-stream {
        font-size: 1.35rem;
        font-weight: 700;
        color: #00f5d4;
    }
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# --- CACHED COMPUTATIONAL PIPELINES ---

@st.cache_data(show_spinner=False)
def get_cached_elevation_and_stats(
    source_key: str,
    file_bytes: Optional[bytes] = None
) -> Tuple[np.ndarray, RasterMetadata, ElevationStats]:
    """Caches raster loading and basic elevation statistics."""
    source = file_bytes if file_bytes is not None else source_key
    elevation, metadata = load_raster(source)
    stats = compute_elevation_stats(elevation, metadata)
    return elevation, metadata, stats


@st.cache_data(show_spinner=False)
def get_cached_terrain_analysis(
    source_key: str,
    min_depth_m: float,
    contour_interval: float,
    file_bytes: Optional[bytes] = None
) -> Tuple[TerrainSummary, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Caches slope, aspect, hillshade, and vector contour line extraction."""
    elevation, metadata, _ = get_cached_elevation_and_stats(source_key, file_bytes)
    (
        terrain_summary,
        slope_deg,
        slope_pct,
        aspect,
        hillshade,
        _
    ) = compute_terrain_summary(elevation, metadata, min_depth_m=min_depth_m)

    contour_geojson = extract_contours_geojson(
        elevation,
        metadata,
        interval=contour_interval
    )
    return terrain_summary, slope_deg, slope_pct, aspect, hillshade, contour_geojson


@st.cache_data(show_spinner=False)
def get_cached_hydrological_analysis(
    source_key: str,
    flow_accum_threshold: float,
    min_waterlogging_depth_m: float,
    file_bytes: Optional[bytes] = None
) -> HydrologyResult:
    """Caches WhiteboxTools depression filling, flow accumulation, streams, and sub-basins."""
    if file_bytes is not None:
        dem_disk_path = Path("data/processed/uploaded_field.tif")
        dem_disk_path.parent.mkdir(parents=True, exist_ok=True)
        if not dem_disk_path.exists() or dem_disk_path.stat().st_size != len(file_bytes):
            with open(dem_disk_path, "wb") as f:
                f.write(file_bytes)
        dem_source = str(dem_disk_path)
    else:
        dem_source = source_key

    return run_hydrological_analysis(
        dem_source=dem_source,
        output_dir="data/processed",
        flow_accum_threshold=flow_accum_threshold,
        min_waterlogging_depth_m=min_waterlogging_depth_m
    )


@st.cache_data(show_spinner=False)
def get_cached_drainage_plan(
    source_key: str,
    flow_accum_threshold: float,
    min_waterlogging_depth_m: float,
    min_waterlogging_area_m2: float,
    min_bed_slope_pct: float,
    ditch_bottom_width_m: float,
    file_bytes: Optional[bytes] = None
) -> DrainagePlanResult:
    """Caches candidate outlet detection, least-cost routing, and earthwork calculations."""
    hydro_res = get_cached_hydrological_analysis(
        source_key=source_key,
        flow_accum_threshold=flow_accum_threshold,
        min_waterlogging_depth_m=min_waterlogging_depth_m,
        file_bytes=file_bytes
    )
    if file_bytes is not None:
        dem_disk_path = Path("data/processed/uploaded_field.tif")
        dem_source = str(dem_disk_path)
    else:
        dem_source = source_key

    return generate_drainage_plan(
        dem_source=dem_source,
        hydrology_result=hydro_res,
        output_dir="data/processed",
        min_waterlogging_depth_m=min_waterlogging_depth_m,
        min_waterlogging_area_m2=min_waterlogging_area_m2,
        min_bed_slope_pct=min_bed_slope_pct,
        ditch_bottom_width_m=ditch_bottom_width_m
    )


@st.cache_data(show_spinner=False)
def get_cached_lidar_processing(
    source_key: str,
    file_bytes: Optional[bytes] = None
) -> LidarProcessingResult:
    """Caches point cloud ingestion, classification filtering, and 1m DTM rasterization."""
    if file_bytes is not None:
        lidar_disk_path = Path(f"data/processed/uploaded_cloud.las")
        lidar_disk_path.parent.mkdir(parents=True, exist_ok=True)
        if not lidar_disk_path.exists() or lidar_disk_path.stat().st_size != len(file_bytes):
            with open(lidar_disk_path, "wb") as f:
                f.write(file_bytes)
        source = str(lidar_disk_path)
    else:
        source = source_key
    return process_lidar_point_cloud(source=source, output_dir="data/processed")


@st.cache_data(show_spinner=False)
def get_cached_ditch_extraction(
    source_key: str,
    rea_radius_m: float = 3.0,
    file_bytes: Optional[bytes] = None,
    is_lidar: bool = False
) -> DitchExtractionResult:
    """Caches Roelens et al. REA calculation, Otsu thresholding, and culvert bridging."""
    if is_lidar:
        lidar_res = get_cached_lidar_processing(source_key, file_bytes)
        elev = lidar_res.elevation_array
        meta = lidar_res.metadata
        ground_pts = lidar_res.ground_points_xyz
    else:
        elev, meta, _ = get_cached_elevation_and_stats(source_key, file_bytes)
        ground_pts = None

    return run_ditch_extraction(
        elevation=elev,
        metadata=meta,
        ground_points_xyz=ground_pts,
        rea_radius_m=rea_radius_m
    )


def main():
    # Application Title & Subtitle
    st.markdown('<div class="main-header">🌾 Field Drainage Using LiDAR</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Smart Farm Drainage Assistant &bull; Phase 5: Real LiDAR (.las/.laz) Processing &amp; Research-Backed Ditch Extraction</div>',
        unsafe_allow_html=True
    )

    # --- SIDEBAR CONTROLS ---
    with st.sidebar:
        st.header("🌱 Field Data & Map Setup")
        
        sample_tif_path = Path("data/sample/synthetic_indian_farm.tif")
        sample_las_path = Path("data/sample/synthetic_indian_farm.las")

        col_b1, col_b2 = st.columns(2)
        load_sample_dem = col_b1.button("🌱 5-Ac TIFF", use_container_width=True, help="Load synthetic Indian farm elevation raster.")
        load_sample_lidar = col_b2.button("📡 5-Ac LiDAR", use_container_width=True, help="Load 100k-point synthetic LiDAR point cloud with culvert gap.")

        uploaded_file = st.file_uploader(
            "Upload Field Map / LiDAR (.las, .laz, .tif, .tiff)",
            type=["tif", "tiff", "las", "laz"],
            help="Upload bare-earth GeoTIFF elevation (.tif) or raw airborne/drone LiDAR point clouds (.las, .laz)."
        )

        st.markdown("---")
        st.subheader("🗺️ Map View & Layers")
        
        default_layer_choice = st.selectbox(
            "Main Map Layer",
            options=[
                "Extracted Ditches & Culverts (LiDAR)",
                "Proposed Surface Drainage Routes",
                "Land Elevation Map",
                "Natural Runoff Paths (Streams)",
                "Water Flow Concentration",
                "Waterlogging Depth Zones",
                "Land Slope (% Grade)",
                "3D Land Shape (Hillshade)",
                "Slope Direction (Aspect)",
                "Field Catchments (Sub-Basins)"
            ],
            index=0,
            help="Choose which layer appears first on the map. You can also turn layers on and off anytime using the top-right map menu."
        )

        colormap_choice = st.selectbox(
            "Elevation Color Theme",
            options=["terrain", "viridis", "plasma", "YlGnBu", "gist_earth"],
            index=0,
            help="Select the color palette for elevation gradients."
        )

        opacity_slider = st.slider(
            "Map Transparency",
            min_value=0.2,
            max_value=1.0,
            value=0.75,
            step=0.05,
            help="Adjust transparency to blend elevation data over satellite imagery."
        )

        animate_flow_toggle = st.toggle(
            "🌊 Animated Water Flow on Map",
            value=True,
            help="Show animated water pulses flowing along channels down toward outlets. Speed corresponds to route slope."
        )

        st.markdown("---")
        st.subheader("🚜 Drainage Trench Design")

        min_ditch_slope_pct = st.slider(
            "Minimum Trench Bed Grade (%)",
            min_value=0.1,
            max_value=1.0,
            value=0.2,
            step=0.05,
            help="Minimum downhill hydraulic gradient along the ditch invert bed to guarantee self-cleansing gravity drainage without uphill backwater."
        )

        ditch_bottom_width = st.slider(
            "Ditch Bed Width (m)",
            min_value=0.3,
            max_value=1.5,
            value=0.5,
            step=0.1,
            help="Excavated base width of trapezoidal drainage channel (standard farm mini-excavator bucket is ~0.5m)."
        )

        min_sink_area = st.slider(
            "Ponding Hotspot Min Area (m²)",
            min_value=5,
            max_value=50,
            value=10,
            step=5,
            help="Minimum surface puddle area required to generate an automated engineered drainage route."
        )

        st.markdown("---")
        st.subheader("💧 Water Flow & Hydrology")
        
        flow_accum_threshold = st.slider(
            "Channel Sensitivity (Flow Threshold)",
            min_value=30,
            max_value=300,
            value=80,
            step=10,
            help="Accumulated upstream cell threshold to extract natural runoff channels. Lower values show small surface trickles; higher values isolate major drainage gullies."
        )

        sink_threshold_m = st.slider(
            "Minimum Waterlogging Depth (m)",
            min_value=0.02,
            max_value=0.20,
            value=0.05,
            step=0.01,
            help="Minimum puddle depth to flag as a genuine waterlogging depression (filters minor tillage roughness)."
        )

        contour_interval = st.selectbox(
            "Elevation Line Gap (m)",
            options=[0.1, 0.25, 0.5, 1.0],
            index=1,
            help="Height step between contour lines (e.g., 0.25 m shows fine micro-slope details across flat plots)."
        )

        st.markdown("---")
        st.caption("🛡️ **Smart Agri-Drainage:** Powered by WhiteboxTools Open Core & 100% free open-source technology.")
        st.caption("© 2026 Varad Patil. All Rights Reserved.")

    # Determine which source to process
    source_key = None
    file_bytes = None
    source_name = ""
    is_lidar = False

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        source_key = f"upload_{uploaded_file.name}_{len(file_bytes)}"
        source_name = uploaded_file.name
        is_lidar = uploaded_file.name.lower().endswith((".las", ".laz"))
    elif load_sample_lidar or st.session_state.get("loaded_sample_lidar"):
        st.session_state["loaded_sample_lidar"] = True
        st.session_state.pop("loaded_sample_dem", None)
        if sample_las_path.exists():
            source_key = str(sample_las_path)
            source_name = "Sample Indian Farm Airborne LiDAR (.las)"
            is_lidar = True
        else:
            st.error(f"Sample LiDAR file not found at `{sample_las_path}`. Run `python scripts/generate_synthetic_lidar.py` first.")
            return
    elif load_sample_dem or st.session_state.get("loaded_sample_dem", True):
        st.session_state["loaded_sample_dem"] = True
        st.session_state.pop("loaded_sample_lidar", None)
        if sample_tif_path.exists():
            source_key = str(sample_tif_path)
            source_name = "Sample 5-Acre Indian Farm (UTM Zone 44N)"
            is_lidar = False
        else:
            st.error(f"Sample data file not found at `{sample_tif_path}`. Run `python scripts/generate_synthetic_farm.py` first.")
            return

    # If no data is selected yet, display initial instructions
    if source_key is None:
        st.info("👈 **Get Started:** Click **'📡 5-Ac LiDAR'** or **'🌱 5-Ac TIFF'** in the sidebar to see an instant demo, or upload your own field elevation / LiDAR file.")
        return

    # --- PROCESS RASTER / LIDAR, TERRAIN, HYDROLOGY & DITCH EXTRACTION ENGINE (CACHED) ---
    try:
        with st.spinner("Processing point clouds, bare-earth filtering, slope, and drainage network..."):
            lidar_meta: Optional[LidarMetadata] = None
            if is_lidar:
                # 1A. Cached LiDAR Processing (LAS/LAZ -> 1m GeoTIFF DTM)
                lidar_res = get_cached_lidar_processing(
                    source_key=source_key,
                    file_bytes=file_bytes
                )
                elevation = lidar_res.elevation_array
                metadata = lidar_res.metadata
                stats = compute_elevation_stats(elevation, metadata)
                lidar_meta = lidar_res.lidar_metadata
                dtm_source_key = lidar_res.dtm_tif_path
                dtm_bytes = None
            else:
                # 1B. Cached Raster Elevation & Metadata
                elevation, metadata, stats = get_cached_elevation_and_stats(
                    source_key=source_key,
                    file_bytes=file_bytes
                )
                dtm_source_key = source_key
                dtm_bytes = file_bytes

            # 2. Cached Terrain Metrics (Slope, Aspect, Hillshade, Contours)
            (
                terrain_summary,
                slope_deg,
                slope_pct,
                aspect,
                hillshade,
                contour_geojson
            ) = get_cached_terrain_analysis(
                source_key=dtm_source_key,
                min_depth_m=sink_threshold_m,
                contour_interval=contour_interval,
                file_bytes=dtm_bytes
            )

            # 3. Cached WhiteboxTools Hydrology (D8 Flow, Accumulation, Streams, Basins)
            hydro_result = get_cached_hydrological_analysis(
                source_key=dtm_source_key,
                flow_accum_threshold=float(flow_accum_threshold),
                min_waterlogging_depth_m=sink_threshold_m,
                file_bytes=dtm_bytes
            )

            # 4. Cached Roelens et al. (2018) Ditch Extraction & Culvert Bridging (Phase 5)
            ditch_result = get_cached_ditch_extraction(
                source_key=source_key,
                rea_radius_m=3.0,
                file_bytes=file_bytes,
                is_lidar=is_lidar
            )

            # 5. Cached Drainage Recommendation & Surface Trench Routes (Phase 4)
            drainage_plan = get_cached_drainage_plan(
                source_key=dtm_source_key,
                flow_accum_threshold=float(flow_accum_threshold),
                min_waterlogging_depth_m=sink_threshold_m,
                min_waterlogging_area_m2=float(min_sink_area),
                min_bed_slope_pct=float(min_ditch_slope_pct),
                ditch_bottom_width_m=float(ditch_bottom_width),
                file_bytes=dtm_bytes
            )
    except Exception as e:
        st.error(f"❌ Error during point cloud and hydrological processing: {str(e)}")
        return

    # --- POINT CLOUD SUMMARY BADGE & METRICS (PHASE 5) ---
    if is_lidar and lidar_meta is not None:
        st.markdown(f"""
        <div style="background-color: #14181F; border: 1px solid #00F5D4; border-radius: 8px; padding: 12px 16px; margin-bottom: 14px;">
            <span style="background-color: #00F5D4; color: #0B0D10; font-weight: bold; font-size: 11px; padding: 3px 8px; border-radius: 4px; text-transform: uppercase;">
                {lidar_meta.data_source_badge}
            </span>
            <span style="color: #F0F4F8; font-weight: 600; font-size: 13px; margin-left: 10px;">
                Raw Laser Return Density: <b style="color: #00F5D4;">{lidar_meta.point_density_sq_m:.1f} pts/m²</b> &bull; Bare-Earth Classification: <b style="color: #2ECC71;">ASPRS Class 2 ({lidar_meta.ground_fraction_pct:.1f}%)</b>
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        lp1, lp2, lp3, lp4 = st.columns(4)
        with lp1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Laser Pulses</div>
                <div class="metric-value">{lidar_meta.total_points:,}</div>
            </div>
            """, unsafe_allow_html=True)
        with lp2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Bare-Earth Returns</div>
                <div class="metric-value">{lidar_meta.ground_points_count:,} ({lidar_meta.ground_fraction_pct:.1f}%)</div>
            </div>
            """, unsafe_allow_html=True)
        with lp3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Point Density</div>
                <div class="metric-value">{lidar_meta.point_density_sq_m:.1f} pts/m²</div>
            </div>
            """, unsafe_allow_html=True)
        with lp4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Tree / Bund Returns</div>
                <div class="metric-value">{lidar_meta.vegetation_points_count:,} pts</div>
            </div>
            """, unsafe_allow_html=True)

    # --- CRS ALERT CHECKS ---
    if metadata.is_geographic:
        st.warning(
            "⚠️ **Latitude/Longitude Coordinates Detected:** This file is measured in degrees rather than meters. "
            "Distance and area values are approximate. For best drainage precision, export your map in an Indian UTM Zone (e.g., EPSG:32643 or EPSG:32644)."
        )
    else:
        st.success(f"✅ **Accurate Metric Map:** Measured in **meters** ({stats.crs_display}) &bull; Ground detail: **{stats.spatial_resolution_m:.2f} m per pixel**")

    # --- KPI METRIC CARDS (ELEVATION & TERRAIN) ---
    st.markdown("### 📊 Field Elevation & Land Relief")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Lowest Point</div>
            <div class="metric-value">{stats.min_elevation:.2f} m</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Highest Point</div>
            <div class="metric-value">{stats.max_elevation:.2f} m</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Height Difference</div>
            <div class="metric-value">{stats.elevation_range:.2f} m</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Average Height</div>
            <div class="metric-value">{stats.mean_elevation:.2f} m</div>
        </div>
        """, unsafe_allow_html=True)

    with m5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Field Area</div>
            <div class="metric-value">{stats.area_acres:.2f} ac</div>
        </div>
        """, unsafe_allow_html=True)

    with m6:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Map Detail</div>
            <div class="metric-value">{stats.spatial_resolution_m:.1f} m/px</div>
        </div>
        """, unsafe_allow_html=True)

    # --- PHASE 3 HYDROLOGY & RUNOFF SUMMARY PANEL ---
    st.markdown("### 💧 Hydrological Runoff & Waterlogging Vulnerability Panel")
    h1, h2, h3, h4, h5 = st.columns(5)

    with h1:
        st.markdown(f"""
        <div class="metric-card-stream">
            <div class="metric-label">Natural Runoff Channels</div>
            <div class="metric-value-stream">{hydro_result.total_stream_length_m:,.0f} m</div>
        </div>
        """, unsafe_allow_html=True)

    with h2:
        st.markdown(f"""
        <div class="metric-card-sink">
            <div class="metric-label">Max Ponding Depth</div>
            <div class="metric-value-sink">{hydro_result.max_waterlogging_depth_cm:.1f} cm</div>
        </div>
        """, unsafe_allow_html=True)

    with h3:
        st.markdown(f"""
        <div class="metric-card-sink">
            <div class="metric-label">Ponding Surface Area</div>
            <div class="metric-value-sink">{hydro_result.waterlogging_area_acres:.2f} ac</div>
        </div>
        """, unsafe_allow_html=True)

    with h4:
        st.markdown(f"""
        <div class="metric-card-sink">
            <div class="metric-label">Trapped Water Volume</div>
            <div class="metric-value-sink">{hydro_result.total_waterlogging_volume_m3:,.0f} m³</div>
        </div>
        """, unsafe_allow_html=True)

    with h5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Field Sub-Catchments</div>
            <div class="metric-value">{hydro_result.sub_basin_count} basins</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- INTERACTIVE MAP VIEWPORT (OPTIMIZED WITH RETURNED_OBJECTS=[]) ---
    st.markdown(f"### 🗺️ Interactive Field Drainage & Water Flow Map &bull; *{source_name}*")
    st.caption("💡 Tip: Click the layer icon in the top-right of the map to toggle between Proposed Drainage Routes, Runoff Paths, Waterlogging Zones, DTM, Slope, and Sub-Basins.")
    
    with st.spinner("Rendering interactive field map..."):
        fmap = create_terrain_map(
            elevation=elevation,
            metadata=metadata,
            elevation_stats=stats,
            terrain_summary=terrain_summary,
            slope_pct=slope_pct,
            aspect=aspect,
            hillshade=hillshade,
            depression_depth=hydro_result.depression_depth_array,
            contour_geojson=contour_geojson,
            flow_accum=hydro_result.flow_accum_array,
            streams_geojson=hydro_result.streams_geojson,
            basins_geojson=hydro_result.basins_geojson,
            drainage_routes=drainage_plan.routes,
            priority_sinks=drainage_plan.priority_sinks,
            candidate_outlets=drainage_plan.candidate_outlets,
            extracted_ditches_geojson=ditch_result.ditches_geojson,
            reconnected_culverts_geojson=ditch_result.culverts_geojson,
            colormap_name=colormap_choice,
            layer_opacity=opacity_slider,
            default_layer=default_layer_choice,
            animate_flow=animate_flow_toggle
        )
        st_folium(
            fmap,
            width="100%",
            height=570,
            returned_objects=[],
            key="field_drainage_folium_map"
        )

    # --- PHASE 4 DRAINAGE RECOMMENDATIONS & TRENCH PLANNER ---
    st.markdown("### 🚜 Recommended Surface Drainage Channels (Phase 4)")
    
    if drainage_plan.best_route:
        best_r = drainage_plan.best_route
        p1, p2, p3, p4, p5 = st.columns(5)
        
        with p1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Recommended Channel</div>
                <div class="metric-value">{best_r.route_id}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with p2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Channel Length</div>
                <div class="metric-value">{best_r.length_m:.1f} m</div>
            </div>
            """, unsafe_allow_html=True)

        with p3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Bed Invert Grade</div>
                <div class="metric-value">{best_r.bed_slope_pct:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with p4:
            st.markdown(f"""
            <div class="metric-card-alert">
                <div class="metric-label">Max Trench Depth</div>
                <div class="metric-value-alert">{best_r.max_cut_depth_m:.2f} m</div>
            </div>
            """, unsafe_allow_html=True)

        with p5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Excavation Earthwork</div>
                <div class="metric-value">{best_r.earthwork_vol_m3:,.1f} m³</div>
            </div>
            """, unsafe_allow_html=True)

        # Tabbed interface for 3D View, Cross-Section Profile, Ditches & Route Comparison
        tab_3d, tab_profile, tab_ditches, tab_compare, tab_sinks = st.tabs([
            "🌐 Interactive 3D Field Terrain",
            "📈 Longitudinal Trench Elevation Cross-Section",
            "📡 Extracted Ditches & Culverts (Roelens LiDAR)",
            "📋 Candidate Route Comparison & Earthwork Table",
            "🌊 Identified Waterlogging Depressions (Sinks)"
        ])

        route_options = {r.route_id: r for r in drainage_plan.routes}

        with tab_3d:
            st.markdown("#### 🌐 Interactive 3D Bare-Earth Terrain Model")
            c_3d_1, c_3d_2, c_3d_3, c_3d_4 = st.columns(4)
            with c_3d_1:
                selected_route_id_3d = st.selectbox(
                    "3D Route Overlay:",
                    options=list(route_options.keys()),
                    index=0,
                    key="select_route_3d",
                    format_func=lambda rid: f"{rid} {'⭐ [Recommended]' if route_options[rid].is_recommended else ''}"
                )
            with c_3d_2:
                vertical_exagg = st.slider(
                    "Vertical Relief Exaggeration",
                    min_value=1.0,
                    max_value=25.0,
                    value=10.0,
                    step=1.0,
                    help="Magnifies vertical relief to make micro-topography, bunds, and waterlogging hollows clearly visible in 3D."
                )
            with c_3d_3:
                flood_depth_sim = st.slider(
                    "Simulate Rainfall / Flooding (m)",
                    min_value=0.0,
                    max_value=1.5,
                    value=0.0,
                    step=0.05,
                    help="Simulates water accumulation by raising a translucent water plane across the field."
                )
            with c_3d_4:
                colormap_3d = st.selectbox(
                    "3D Color Palette",
                    options=["Viridis", "Earth", "Turbo", "YlGnBu", "Plasma"],
                    index=0
                )

            selected_route_3d = route_options[selected_route_id_3d]
            fig_3d = create_3d_terrain_figure(
                elevation=elevation,
                metadata=metadata,
                selected_route=selected_route_3d,
                vertical_exaggeration=vertical_exagg,
                flood_depth_m=flood_depth_sim,
                colormap=colormap_3d
            )
            st.plotly_chart(fig_3d, use_container_width=True)
            st.caption(
                "💡 **3D Navigation:** Left-click & drag to rotate; right-click & drag or scroll to zoom; "
                "double-click to reset view. The selected drainage channel is highlighted in bright cyan."
            )

        with tab_profile:
            # Route selector if multiple candidate routes exist
            selected_route_id = st.selectbox(
                "Select Drainage Route for Longitudinal Cross-Section:",
                options=list(route_options.keys()),
                index=0,
                key="select_route_profile",
                format_func=lambda rid: f"{rid} {'⭐ [Recommended / Optimal]' if route_options[rid].is_recommended else ''} (Length: {route_options[rid].length_m:.1f} m, Max Cut: {route_options[rid].max_cut_depth_m:.2f} m)"
            )
            selected_route = route_options[selected_route_id]
            fig = create_trench_profile_figure(selected_route)
            st.pyplot(fig, use_container_width=True)
            st.caption(
                f"ℹ️ **Profile Notes for {selected_route.route_id}:** Starting at waterlogging pit invert ({selected_route.start_elev_m:.2f} m), "
                f"the trench maintains a minimum hydraulic gravity bed grade of {selected_route.bed_slope_pct:.2f}% down to the field outlet "
                f"(outlet bed: {selected_route.outlet_bed_elev_m:.2f} m, ground: {selected_route.end_elev_m:.2f} m). "
                f"Maximum excavation cut depth is {selected_route.max_cut_depth_m:.2f} m with an average depth of {selected_route.avg_cut_depth_m:.2f} m."
            )

        with tab_ditches:
            st.markdown("#### 📡 Research-Backed Ditch Extraction & Culvert Reconnection (Roelens et al. 2018)")
            d_col1, d_col2, d_col3 = st.columns(3)
            with d_col1:
                st.markdown(f"""
                <div class="metric-card-stream">
                    <div class="metric-label">Open Ditch Length</div>
                    <div class="metric-value-stream">{ditch_result.total_ditch_length_m:.1f} m</div>
                </div>
                """, unsafe_allow_html=True)
            with d_col2:
                st.markdown(f"""
                <div class="metric-card-alert">
                    <div class="metric-label">Reconnected Culverts</div>
                    <div class="metric-value-alert">{len(ditch_result.culverts)} crossings ({ditch_result.total_culvert_length_m:.1f} m)</div>
                </div>
                """, unsafe_allow_html=True)
            with d_col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Total Connected Network</div>
                    <div class="metric-value">{ditch_result.total_ditch_length_m + ditch_result.total_culvert_length_m:.1f} m</div>
                </div>
                """, unsafe_allow_html=True)

            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.markdown("##### Extracted Open Ditches")
                if ditch_result.ditches:
                    ditch_table = []
                    for d in ditch_result.ditches:
                        ditch_table.append({
                            "Segment ID": f"Segment #{d.segment_id}",
                            "Length (m)": f"{d.length_m:.1f}",
                            "Start Invert (m)": f"{d.start_elev_m:.2f}",
                            "End Invert (m)": f"{d.end_elev_m:.2f}",
                            "Type": "Open Ditch"
                        })
                    st.table(ditch_table)
                else:
                    st.info("No open ditch centerlines detected at current REA threshold.")

            with col_t2:
                st.markdown("##### Reconnected Culvert / Driveway Crossings")
                if ditch_result.culverts:
                    culvert_table = []
                    for c in ditch_result.culverts:
                        culvert_table.append({
                            "Culvert ID": f"Culvert #{c.culvert_id}",
                            "Gap Distance (m)": f"{c.gap_length_m:.1f}",
                            "Confidence": f"{c.confidence_score * 100:.1f}%",
                            "Connected Segments": f"Seg {c.upstream_segment_id} → Seg {c.downstream_segment_id}",
                            "Elevation Drop (m)": f"{c.elevation_drop_m:.2f}"
                        })
                    st.table(culvert_table)
                else:
                    st.info("No fragmented ditch gaps or driveway crossings detected.")

        with tab_compare:
            st.markdown("#### Candidate Surface Drainage Routes Comparison")
            table_rows = []
            for r in drainage_plan.routes:
                table_rows.append({
                    "Route ID": r.route_id,
                    "Target Outlet": f"Outlet #{r.outlet_id}",
                    "Status": "⭐ Recommended" if r.is_recommended else "Alternative",
                    "Total Length (m)": f"{r.length_m:,.1f}",
                    "Ground Slope (%)": f"{r.avg_slope_pct:.2f}%",
                    "Bed Invert Grade (%)": f"{r.bed_slope_pct:.2f}%",
                    "Max Cut Depth (m)": f"{r.max_cut_depth_m:.2f}",
                    "Avg Cut Depth (m)": f"{r.avg_cut_depth_m:.2f}",
                    "Excavation Volume (m³)": f"{r.earthwork_vol_m3:,.1f}",
                    "Feasibility": r.feasibility_rank
                })
            st.table(table_rows)

        with tab_sinks:
            st.markdown("#### Waterlogging Sinks & Pockets Prioritization")
            sink_rows = []
            for s in drainage_plan.priority_sinks:
                sink_rows.append({
                    "Sink Priority": f"Priority #{s.sink_id}",
                    "Ponding Severity": s.severity,
                    "Surface Area (m²)": f"{s.area_m2:,.0f}",
                    "Surface Area (Acres)": f"{s.area_m2 / 4046.8564:,.2f}",
                    "Trapped Water Volume (m³)": f"{s.volume_m3:,.1f}",
                    "Max Water Depth (cm)": f"{s.max_depth_cm:.1f}",
                    "Avg Water Depth (cm)": f"{s.mean_depth_cm:.1f}",
                    "GPS Coordinates": f"{s.pit_coord_wgs84[0]:.6f}, {s.pit_coord_wgs84[1]:.6f}"
                })
            st.table(sink_rows)
    else:
        st.info("ℹ️ No critical waterlogging depressions requiring artificial surface drainage were detected at the current thresholds.")

    # --- DETAILED METADATA & DATA TABLE EXPANDER ---
    with st.expander("🔍 Technical Geospatial & Hydrological Metrics (Advanced)"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.write("**Grid Dimensions & Data Quality:**")
            st.json({
                "Grid Width": metadata.width,
                "Grid Height": metadata.height,
                "Valid Data Pixels": stats.valid_pixels,
                "Cleaned NoData Pixels": stats.nodata_pixels,
                "Field Coverage": f"{stats.valid_fraction*100:.2f}%",
                "Pixel Dimensions": f"{metadata.res_x:.4f} x {metadata.res_y:.4f} ({metadata.units})"
            })
        with c2:
            st.write("**Boundary Coordinates & Area:**")
            st.json({
                "Coordinate System (CRS)": stats.crs_display,
                "Metric Units": metadata.is_projected,
                "Native Bounds": [round(b, 2) for b in metadata.bounds_native],
                "GPS Bounds (WGS84)": [round(b, 6) for b in metadata.bounds_wgs84],
                "Calculated Area (m²)": f"{stats.area_sq_m:,.1f} m²",
                "Calculated Area (Hectares)": f"{stats.area_ha:.3f} ha"
            })
        with c3:
            st.write("**Slope & Relief Profile:**")
            st.json(terrain_summary.to_dict())
            st.write("**Roelens Ditch Extraction:**")
            st.json(ditch_result.to_dict())
        with c4:
            if is_lidar and lidar_meta:
                st.write("**Airborne LiDAR Metrics:**")
                st.json(lidar_meta.to_dict())
            else:
                st.write("**Drainage Plan Metrics:**")
                st.json(drainage_plan.to_dict())


if __name__ == "__main__":
    main()
