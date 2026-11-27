"""
Field Drainage Using LiDAR - Phase 3 Web Application
Intelligent LiDAR-Based Agricultural Field Drainage Planning System
Interactive Streamlit Dashboard & WhiteboxTools Hydrological Engine
Optimized for high-performance caching and zero-rerun interactive mapping.
"""

import json
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
from modules.report_generator import generate_drainage_pdf_report
try:
    from components.maps import create_terrain_map, create_focused_map, create_side_by_side_swipe_map
except ImportError:
    import importlib
    import components.maps
    importlib.reload(components.maps)
    from components.maps import create_terrain_map, create_focused_map, create_side_by_side_swipe_map

from components.elevation_profile import create_trench_profile_figure
from components.charts import create_3d_terrain_figure

# Configure Streamlit Page
st.set_page_config(
    page_title="Field Drainage Using LiDAR | LiDAR Processing & Ditch Extraction",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for High-Precision Minimalist Dark-Mode Desktop Web Application UI/UX
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
    
    /* Global Canvas & Typography */
    html, body, [class*="css"], .stApp {
        background-color: #0B0D10 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        color: #F0F4F8 !important;
    }

    /* Monospace for GIS Numbers, Coordinates, and Engineering Metrics */
    .mono-font, .metric-value, .metric-value-alert, .metric-value-sink, .metric-value-stream,
    .hud-item b, .legend-labels, code, pre, [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        letter-spacing: -0.3px;
    }

    /* Streamlit Top Header & Main Grid Layout */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
        height: 2.5rem !important;
        z-index: 10 !important;
    }
    .block-container {
        padding-top: 3.6rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
        max-width: 1440px !important;
    }

    /* Left Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #14181F !important;
        border-right: 1px solid #232A35 !important;
        width: 320px !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding: 1.2rem 1.0rem 2.0rem 1.0rem !important;
    }

    /* Top Navigation Bar (56px) */
    .top-nav-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        min-height: 56px;
        padding: 8px 18px;
        margin-top: 4px;
        margin-bottom: 16px;
        background: #14181F;
        border: 1px solid #232A35;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
        position: relative;
        z-index: 20;
    }
    .top-nav-left {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .top-nav-brand {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 1.15rem;
        font-weight: 700;
        color: #F0F4F8;
    }
    .top-nav-brand span {
        color: #2ECC71;
    }
    .top-nav-right {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .pill-btn {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #1C222B;
        color: #8B949E;
        border: 1px solid #232A35;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.78rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .pill-btn:hover {
        color: #F0F4F8;
        border-color: #2ECC71;
        background: #1E2530;
    }
    .pill-btn.active {
        color: #2ECC71;
        border-color: #2ECC71;
        background: rgba(46, 204, 113, 0.12);
    }

    /* Status Pill Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #1C222B;
        color: #F0F4F8;
        border: 1px solid #232A35;
        border-radius: 9999px;
        padding: 4px 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .status-badge-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #2ECC71;
        box-shadow: 0 0 8px #2ECC71;
        animation: pulse-glow 2s infinite ease-in-out;
    }

    @keyframes pulse-glow {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.45; transform: scale(0.85); }
    }

    /* Uploader Box */
    [data-testid="stFileUploader"] {
        border: 1px dashed #232A35 !important;
        border-radius: 10px !important;
        background: #14181F !important;
        padding: 8px !important;
        transition: border-color 0.2s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #2ECC71 !important;
    }

    /* KPI Metric Strip (96px height, 4 Equal Cards, 12px radius) */
    .kpi-strip {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 16px;
    }
    .kpi-card {
        background: #14181F;
        border: 1px solid #232A35;
        border-radius: 12px;
        padding: 14px 16px;
        min-height: 96px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 4px 16px -2px rgba(0, 0, 0, 0.45);
        transition: transform 0.18s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.18s ease, box-shadow 0.18s ease;
        position: relative;
        overflow: hidden;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 255, 255, 0.15);
        box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.65);
    }
    .kpi-card.accent-green { border-left: 4px solid #2ECC71; }
    .kpi-card.accent-amber { border-left: 4px solid #F59E0B; }
    .kpi-card.accent-cyan  { border-left: 4px solid #00E5FF; }
    .kpi-card.accent-slate { border-left: 4px solid #8B949E; }

    .kpi-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .kpi-title {
        font-size: 0.75rem;
        color: #8B949E;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .kpi-badge {
        font-size: 0.68rem;
        font-weight: 700;
        padding: 2px 7px;
        border-radius: 6px;
    }
    .kpi-badge-amber {
        background: rgba(245, 158, 11, 0.16);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.35);
    }
    .kpi-badge-green {
        background: rgba(46, 204, 113, 0.16);
        color: #2ECC71;
        border: 1px solid rgba(46, 204, 113, 0.35);
    }
    .kpi-badge-cyan {
        background: rgba(0, 229, 255, 0.16);
        color: #00E5FF;
        border: 1px solid rgba(0, 229, 255, 0.35);
    }
    .kpi-value {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.55rem;
        font-weight: 700;
        color: #F0F4F8;
        line-height: 1.1;
        margin: 4px 0 2px 0;
    }
    .kpi-subtext {
        font-size: 0.74rem;
        color: #8B949E;
    }

    /* Floating Heads-Up Display (HUD) */
    .stitch-hud {
        display: flex;
        align-items: center;
        justify-content: flex-start;
        flex-wrap: wrap;
        gap: 12px;
        background: rgba(20, 24, 31, 0.88);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid #232A35;
        border-radius: 10px;
        padding: 8px 14px;
        margin-bottom: 12px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
    }
    .hud-item {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.78rem;
        color: #F0F4F8;
    }
    .hud-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
    }
    .dot-emerald { background: #2ECC71; box-shadow: 0 0 8px rgba(46, 204, 113, 0.8); }
    .dot-cyan { background: #00E5FF; box-shadow: 0 0 8px rgba(0, 229, 255, 0.8); }
    .dot-blue { background: #38BDF8; box-shadow: 0 0 8px rgba(56, 189, 248, 0.8); }
    .dot-amber { background: #F59E0B; box-shadow: 0 0 8px rgba(245, 158, 11, 0.8); }
    .dot-violet { background: #A855F7; box-shadow: 0 0 8px rgba(168, 85, 247, 0.8); }

    /* Interactive Buttons */
    .stButton>button {
        background: #14181F !important;
        color: #2ECC71 !important;
        border: 1px solid #2ECC71 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        min-height: 42px !important;
        padding: 8px 16px !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    .stButton>button:hover {
        background: #2ECC71 !important;
        color: #0B0D10 !important;
        box-shadow: 0 0 16px rgba(46, 204, 113, 0.45) !important;
        transform: translateY(-1px);
    }
    .stDownloadButton>button {
        background: #2ECC71 !important;
        color: #0B0D10 !important;
        border: 1px solid #2ECC71 !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        min-height: 44px !important;
        padding: 10px 18px !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    .stDownloadButton>button:hover {
        background: #27ae60 !important;
        color: #0B0D10 !important;
        box-shadow: 0 0 20px rgba(46, 204, 113, 0.6) !important;
        transform: translateY(-1px);
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #232A35;
        padding-bottom: 4px;
        margin-bottom: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 18px;
        color: #8B949E;
        font-weight: 600;
        font-size: 0.9rem;
        background: transparent;
        transition: color 0.15s ease, background 0.15s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #F0F4F8;
        background: rgba(255, 255, 255, 0.04);
    }
    .stTabs [aria-selected="true"] {
        color: #2ECC71 !important;
        background: #1C222B !important;
        border-bottom: 2px solid #2ECC71 !important;
    }

    /* Bottom Analytics Split Panel Cards */
    .analytics-card {
        background: #14181F;
        border: 1px solid #232A35;
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
    }
    .analytics-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid #232A35;
    }
    .analytics-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #F0F4F8;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Export Cards */
    .export-card {
        background: #14181F;
        border: 1px solid #232A35;
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 14px;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .export-card:hover {
        border-color: rgba(255, 255, 255, 0.16);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35);
    }

    /* Responsive Grid */
    @media (max-width: 1024px) {
        .kpi-strip {
            grid-template-columns: repeat(2, 1fr) !important;
        }
    }
    @media (max-width: 640px) {
        .kpi-strip {
            grid-template-columns: 1fr !important;
        }
        .top-nav-bar {
            flex-direction: column !important;
            align-items: flex-start !important;
            gap: 8px !important;
        }
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


@st.cache_data(show_spinner=False)
def get_cached_pdf_report(
    source_key: str,
    data_source_badge: str,
    field_plot_name: str,
    flow_accum_threshold: float = 80.0,
    min_waterlogging_depth_m: float = 0.05,
    min_waterlogging_area_m2: float = 10.0,
    min_bed_slope_pct: float = 0.20,
    file_bytes: Optional[bytes] = None,
    is_lidar: bool = False
) -> bytes:
    """Caches compilation of the publication-grade PDF drainage feasibility report."""
    if is_lidar:
        lidar_res = get_cached_lidar_processing(source_key, file_bytes)
        elevation = lidar_res.elevation_array
        metadata = lidar_res.metadata
        stats = compute_elevation_stats(elevation, metadata)
    else:
        elevation, metadata, stats = get_cached_elevation_and_stats(source_key, file_bytes)

    terrain_summary, *_ = get_cached_terrain_analysis(source_key, min_waterlogging_depth_m, 0.5, file_bytes)
    hydro_result = get_cached_hydrological_analysis(source_key, flow_accum_threshold, min_waterlogging_depth_m, file_bytes)
    drainage_plan = get_cached_drainage_plan(
        source_key, flow_accum_threshold, min_waterlogging_depth_m, min_waterlogging_area_m2, min_bed_slope_pct, 0.5, file_bytes
    )
    ditch_result = get_cached_ditch_extraction(source_key, 3.0, file_bytes, is_lidar)

    out_dir = Path("outputs/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_out = out_dir / f"{Path(source_key).stem}_drainage_plan.pdf"

    generate_drainage_pdf_report(
        output_pdf_path=pdf_out,
        dem_array=elevation,
        metadata=metadata,
        elev_stats=stats,
        terrain_summary=terrain_summary,
        hydro_result=hydro_result,
        drainage_plan=drainage_plan,
        ditch_result=ditch_result,
        data_source_type=data_source_badge,
        project_name="Agricultural Surface Drainage Feasibility Plan",
        field_plot_id=field_plot_name
    )
    with open(pdf_out, "rb") as f:
        return f.read()


def main():
    # 1. Top Navigation Bar (56px)
    st.markdown("""
    <div class="top-nav-bar">
        <div class="top-nav-left">
            <div class="top-nav-brand">
                <span style="color:#2ECC71; font-size:1.4rem;">∿</span>
                <span>Field Drainage Using LiDAR</span>
            </div>
            <div class="status-badge">
                <span class="status-badge-dot"></span>
                <span>Simulated LiDAR &bull; EPSG:32644 (Metric)</span>
            </div>
        </div>
        <div class="top-nav-right">
            <div class="pill-btn">🌐 EN | हिं | मर</div>
            <div class="pill-btn" title="Precision Geospatial Agri-Drainage Engine">❓ Help / Glossary</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- SIDEBAR CONTROLS ---
    with st.sidebar:
        # 1. Header Block
        st.markdown("""
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
            <span style="font-size:1.4rem; color:#2ECC71;">∿</span>
            <span style="font-size:1.1rem; font-weight:700; color:#F0F4F8; letter-spacing:-0.3px;">Field Drainage Using LiDAR</span>
        </div>
        <div style="margin-bottom:14px;">
            <div class="status-badge">
                <span class="status-badge-dot"></span>
                <span>Active CRS &bull; EPSG:32644 (Metric)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        sample_tif_path = Path("data/sample/synthetic_indian_farm.tif")
        sample_las_path = Path("data/sample/synthetic_indian_farm.las")

        # 2. Data Input Section
        st.markdown("#### 📂 Field Data Input")
        uploaded_file = st.file_uploader(
            "Drop .LAS, .LAZ, or .TIF field file",
            type=["tif", "tiff", "las", "laz"],
            help="Upload bare-earth GeoTIFF elevation (.tif) or raw airborne/drone LiDAR point clouds (.las, .laz)."
        )

        col_b1, col_b2 = st.columns(2)
        load_sample_dem = col_b1.button("🌱 5-Ac Farm DEM", use_container_width=True, help="Load 5-acre synthetic Indian farm bare-earth elevation raster.")
        load_sample_lidar = col_b2.button("📡 5-Ac LiDAR (.las)", use_container_width=True, help="Load 100k-point synthetic airborne LiDAR point cloud with culvert gap.")

        st.markdown("---")

        # 3. Map Layers Section
        st.markdown("#### 🗺️ Map Layers & Overlays")
        default_layer_choice = st.selectbox(
            "Default Active Layer",
            options=[
                "Extracted Ditches & Culverts (LiDAR)",
                "Proposed Surface Drainage Routes",
                "Land Elevation Map (DTM)",
                "Natural Runoff Paths (Streams)",
                "Water Flow Concentration",
                "Waterlogging Depth Zones",
                "Land Slope (% Grade)",
                "3D Land Shape (Hillshade)",
                "Slope Direction (Aspect)",
                "Field Catchments (Sub-Basins)"
            ],
            index=0,
            help="Select the default primary layer. You can toggle all layers dynamically via the top-right map menu."
        )

        colormap_choice = st.selectbox(
            "Elevation Colormap",
            options=["terrain", "viridis", "plasma", "YlGnBu", "gist_earth"],
            index=0,
            help="Color ramp for elevation terrain visualization."
        )

        opacity_slider = st.slider(
            "Layer Opacity",
            min_value=0.2,
            max_value=1.0,
            value=0.75,
            step=0.05,
            help="Adjust layer transparency over the basemap."
        )

        st.markdown("---")

        # 4. Analysis Parameters Section
        st.markdown("#### ⚙️ Engineering & Hydrology Parameters")
        contour_interval = st.radio(
            "Contour Interval",
            options=[0.1, 0.25, 0.5, 1.0],
            index=1,
            horizontal=True,
            format_func=lambda x: f"{x}m",
            help="Step between contour lines (0.25m reveals subtle micro-topography on flat farm plots)."
        )

        sink_threshold_m = st.slider(
            "Min. Waterlogging Depth (m)",
            min_value=0.02,
            max_value=0.30,
            value=0.05,
            step=0.01,
            help="Minimum puddle depth required to flag an actionable ponding depression."
        )

        flow_accum_threshold = st.slider(
            "Channel Sensitivity (Flow Accumulation)",
            min_value=30,
            max_value=300,
            value=80,
            step=10,
            help="Upstream accumulation threshold (Low = captures minor surface trickles, High = isolates major drainage paths)."
        )

        min_ditch_slope_pct = st.slider(
            "Minimum Trench Bed Grade (%)",
            min_value=0.1,
            max_value=1.0,
            value=0.2,
            step=0.05,
            help="Guarantees self-cleansing monotonic gravity slope without uphill backwater."
        )

        ditch_bottom_width = st.slider(
            "Ditch Bed Width (m)",
            min_value=0.3,
            max_value=1.5,
            value=0.5,
            step=0.1,
            help="Base width of excavated trapezoidal channel."
        )

        min_sink_area = st.slider(
            "Ponding Hotspot Min Area (m²)",
            min_value=5,
            max_value=50,
            value=10,
            step=5,
            help="Minimum surface puddle area required to generate an automated engineered drainage route."
        )

        vertical_exaggeration_sidebar = st.slider(
            "Vertical Exaggeration (3D)",
            min_value=1.0,
            max_value=25.0,
            value=10.0,
            step=1.0,
            help="Vertical relief multiplier for 3D exploration."
        )

        animate_flow_toggle = st.toggle(
            "🌊 Animated Water Flow",
            value=True,
            help="Animate directional water flow along channels toward field outlets."
        )

        st.markdown("---")
        st.caption("🛡️ **Precision Agri-Drainage Engine:** 100% free open-source stack powered by WhiteboxTools Open Core & laspy.")
        st.caption("© 2026 Varad Patil. All Rights Reserved.")

    # Determine which source to process
    source_key = None
    file_bytes = None
    source_name = ""
    is_lidar = False

    is_sample = (uploaded_file is None)
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

    # Data source provenance badge
    data_source_badge = (
        lidar_meta.data_source_badge
        if (is_lidar and lidar_meta)
        else ("Synthetic Indian Farm Benchmark" if is_sample else "Real Farm DEM / Satellite GeoTIFF")
    )

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

    # --- TOP ROW: 4 CLEAN AGRONOMIC & ENGINEERING KPI METRIC CARDS (96px height) ---
    best_r = drainage_plan.best_route
    best_len_str = f"{best_r.length_m:,.0f} m" if best_r else "No Sink"
    slope_desc = "Nearly Level" if terrain_summary.mean_slope_pct < 1.0 else ("Gentle Slope" if terrain_summary.mean_slope_pct < 2.5 else "Moderate Slope")
    risk_level = "High" if hydro_result.waterlogging_area_pct > 15.0 else ("Moderate" if hydro_result.waterlogging_area_pct > 5.0 else "Low")
    pockets_count = len(drainage_plan.priority_sinks)
    optimal_count = len(drainage_plan.routes)
    outlets_count = len(drainage_plan.candidate_outlets)

    st.markdown(f"""
    <div class="kpi-strip">
        <div class="kpi-card accent-green">
            <div class="kpi-header">
                <span class="kpi-title">Field Area</span>
                <span class="kpi-badge kpi-badge-green">Metric Area</span>
            </div>
            <div class="kpi-value">{stats.area_acres:.1f} Acres</div>
            <div class="kpi-subtext">{stats.area_ha:.2f} ha &bull; {stats.spatial_resolution_m:.0f}m Resolution</div>
        </div>
        <div class="kpi-card accent-amber">
            <div class="kpi-header">
                <span class="kpi-title">Waterlogging Risk</span>
                <span class="kpi-badge kpi-badge-amber">{risk_level} &bull; {hydro_result.waterlogging_area_pct:.1f}%</span>
            </div>
            <div class="kpi-value" style="color: #F59E0B;">{hydro_result.waterlogging_area_pct:.1f}%</div>
            <div class="kpi-subtext">{pockets_count} Priority Pockets &bull; {hydro_result.total_waterlogging_volume_m3:,.0f} m³ trapped</div>
        </div>
        <div class="kpi-card accent-cyan">
            <div class="kpi-header">
                <span class="kpi-title">Recommended Channels</span>
                <span class="kpi-badge kpi-badge-cyan">{optimal_count} Optimal</span>
            </div>
            <div class="kpi-value" style="color: #2ECC71;">{best_len_str}</div>
            <div class="kpi-subtext">{optimal_count} Optimal Routes &bull; {outlets_count} Field Outlet(s)</div>
        </div>
        <div class="kpi-card accent-slate">
            <div class="kpi-header">
                <span class="kpi-title">Average Field Slope</span>
                <span class="kpi-badge" style="background: rgba(139,148,158,0.15); color: #8B949E; border: 1px solid #232A35;">Grade</span>
            </div>
            <div class="kpi-value">{terrain_summary.mean_slope_pct:.2f}%</div>
            <div class="kpi-subtext">{slope_desc} &bull; {'Poor Natural Drainage' if terrain_summary.mean_slope_pct < 1.0 else 'Moderate Drainage'}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Auxiliary Elevation & Hydrology Metrics (Collapsible)
    with st.expander("ℹ️ View Additional Relief, Sub-Basin & Stream Metrics", expanded=False):
        ax1, ax2, ax3, ax4, ax5, ax6 = st.columns(6)
        with ax1:
            st.metric("Lowest Point", f"{stats.min_elevation:.2f} m")
        with ax2:
            st.metric("Highest Point", f"{stats.max_elevation:.2f} m")
        with ax3:
            st.metric("Relief Range", f"{stats.elevation_range:.2f} m")
        with ax4:
            st.metric("Average Height", f"{stats.mean_elevation:.2f} m")
        with ax5:
            st.metric("Natural Runoff Channels", f"{hydro_result.total_stream_length_m:,.0f} m")
        with ax6:
            st.metric("Field Sub-Catchments", f"{hydro_result.sub_basin_count} basins")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- MAIN VIEWPORT: 4 MASTER NAVIGATION TABS ---
    tab_master, tab_3d_view, tab_compare_view, tab_exports = st.tabs([
        "🗺️ Map Analysis",
        "🌐 3D Field View",
        "⚖️ Compare Mode",
        "📑 Reports & Exports"
    ])

    with tab_master:
        st.markdown(f"#### 🗺️ Interactive GIS Map Workspace &bull; *{source_name}*")
        
        # Stitch Floating Heads-Up Display (HUD)
        st.markdown(f"""
        <div class="stitch-hud">
            <div class="hud-item"><span class="hud-dot dot-emerald"></span><b>CRS:</b> {stats.crs_display}</div>
            <div class="hud-item"><span class="hud-dot dot-cyan"></span><b>Spatial Res:</b> {stats.spatial_resolution_m:.2f} m/px</div>
            <div class="hud-item"><span class="hud-dot dot-blue"></span><b>Core:</b> WhiteboxTools + Roelens (2018)</div>
            <div class="hud-item"><span class="hud-dot dot-amber"></span><b>Min Gravity Grade:</b> &ge; {min_ditch_slope_pct:.2f}%</div>
            <div class="hud-item"><span class="hud-dot dot-violet"></span><b>Data Mode:</b> {data_source_badge}</div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("💡 Click the layer icon (top-right of the map) to toggle between Proposed Drainage Routes, Runoff Paths, Waterlogging Zones, DTM, Slope, and Sub-Basins.")
        
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
                height=560,
                returned_objects=[],
                key="field_drainage_master_map"
            )

        # Bottom Analytics Split Panel (30% Viewport Height, 65% / 35% Split)
        st.markdown("<br>", unsafe_allow_html=True)
        col_split_left, col_split_right = st.columns([0.65, 0.35])

        if drainage_plan.routes and drainage_plan.best_route:
            route_options = {r.route_id: r for r in drainage_plan.routes}

            with col_split_left:
                st.markdown("""
                <div class="analytics-card-header">
                    <div class="analytics-title">
                        <span style="color:#2ECC71;">📈</span> Proposed Channel Elevation Profile
                    </div>
                </div>
                """, unsafe_allow_html=True)

                sub_sel1, sub_sel2 = st.columns([0.7, 0.3])
                with sub_sel1:
                    selected_route_id = st.selectbox(
                        "Select Drainage Route",
                        options=list(route_options.keys()),
                        index=0,
                        key="select_route_profile_main",
                        label_visibility="collapsed",
                        format_func=lambda rid: f"{rid} {'— Optimal • ' if route_options[rid].is_recommended else '— Candidate • '}{route_options[rid].bed_slope_pct:.2f}% Grade (Cut: {route_options[rid].max_cut_depth_m:.2f}m)"
                    )
                with sub_sel2:
                    view_all_toggle = st.toggle("View All Routes", value=False, key="toggle_view_all_routes")

                selected_route = route_options[selected_route_id]
                fig_profile = create_trench_profile_figure(selected_route)
                st.pyplot(fig_profile, use_container_width=True)
                st.caption(
                    f"ℹ️ **Invert Profile for {selected_route.route_id}:** Starting at waterlogging pit ({selected_route.start_elev_m:.2f} m), "
                    f"maintains continuous gravity grade of {selected_route.bed_slope_pct:.2f}% to outlet "
                    f"(outlet bed: {selected_route.outlet_bed_elev_m:.2f} m). Max cut: **{selected_route.max_cut_depth_m:.2f} m** &bull; Earthwork: **{selected_route.earthwork_vol_m3:,.1f} m³**."
                )

            with col_split_right:
                st.markdown("""
                <div class="analytics-card-header">
                    <div class="analytics-title">
                        <span style="color:#00E5FF;">🌐</span> 3D Terrain Mesh
                    </div>
                    <div style="font-size:0.75rem; color:#8B949E;">Isometric 10×</div>
                </div>
                """, unsafe_allow_html=True)

                c_rain1, c_rain2 = st.columns([0.65, 0.35])
                with c_rain1:
                    sim_rain_btn = st.toggle("▶ Simulate Rainfall", value=False, key="toggle_simulate_rainfall_preview")
                with c_rain2:
                    st.caption("Flooding Plane")

                flood_m = 0.35 if sim_rain_btn else 0.0
                fig_3d_preview = create_3d_terrain_figure(
                    elevation=elevation,
                    metadata=metadata,
                    selected_route=selected_route,
                    vertical_exaggeration=float(vertical_exaggeration_sidebar),
                    flood_depth_m=flood_m,
                    max_grid=160
                )
                st.plotly_chart(fig_3d_preview, use_container_width=True, key="plotly_3d_preview_mesh")
                st.caption("💡 **3D Preview:** Left-click & drag to rotate; scroll to zoom. Cyan line = trench.")
        else:
            with col_split_left:
                st.info("ℹ️ No critical waterlogging depressions detected at current threshold settings.")
            with col_split_right:
                fig_3d_preview = create_3d_terrain_figure(
                    elevation=elevation,
                    metadata=metadata,
                    selected_route=None,
                    vertical_exaggeration=float(vertical_exaggeration_sidebar),
                    flood_depth_m=0.0,
                    max_grid=160
                )
                st.plotly_chart(fig_3d_preview, use_container_width=True, key="plotly_3d_preview_empty")

    with tab_3d_view:
        st.markdown(f"#### 🌐 Full-Field 3D Topographic Exploration &bull; *{source_name}*")
        st.caption("Inspect micro-topographic depressions, field bunds, and waterlogging hollows in high-resolution 3D perspective.")

        c3_ctrl1, c3_ctrl2 = st.columns(2)
        with c3_ctrl1:
            full_vert_exagg = st.slider(
                "Vertical Relief Exaggeration",
                min_value=1.0,
                max_value=25.0,
                value=float(vertical_exaggeration_sidebar),
                step=1.0,
                key="slider_full_vert_exagg"
            )
        with c3_ctrl2:
            full_flood_sim = st.slider(
                "Simulate Flooding Water Plane (m)",
                min_value=0.0,
                max_value=1.5,
                value=0.0,
                step=0.05,
                key="slider_full_flood_sim"
            )

        active_route_for_3d = drainage_plan.best_route if drainage_plan.routes else None
        fig_3d_full = create_3d_terrain_figure(
            elevation=elevation,
            metadata=metadata,
            selected_route=active_route_for_3d,
            vertical_exaggeration=full_vert_exagg,
            flood_depth_m=full_flood_sim,
            max_grid=220
        )
        st.plotly_chart(fig_3d_full, use_container_width=True, key="plotly_3d_full_field_model")
        st.caption("💡 **3D Interaction:** Left-click & drag to rotate; scroll to zoom; double-click to reset camera. Cyan line = proposed trench.")

    with tab_compare_view:
        st.markdown("#### ⚖️ Visual Comparison: Original Terrain vs. Waterlogging Risk vs. Drainage Alignment")
        
        st.markdown("""
        <div class="stitch-hud">
            <div class="hud-item"><span class="hud-dot dot-emerald"></span><b>Mode:</b> Split-Screen Hydrological Verification</div>
            <div class="hud-item"><span class="hud-dot dot-amber"></span><b>Left View:</b> Bare-Earth Relief & Micro-Contours</div>
            <div class="hud-item"><span class="hud-dot dot-cyan"></span><b>Right View:</b> Trapped Water Pits & Gravity Cut Path</div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Inspect how the engineered gravity drainage trench routes water out of low-lying depressions towards the field boundary.")

        compare_style = st.radio(
            "Comparison Layout Mode:",
            options=["Dual Synchronized View (Side-by-Side)", "Interactive Swipe Split-Screen (Folium)"],
            horizontal=True,
            key="compare_mode_selector"
        )

        if compare_style == "Dual Synchronized View (Side-by-Side)":
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.markdown("##### 1. Original Bare-Earth Terrain & Contours")
                map_terrain = create_focused_map(
                    mode="terrain",
                    elevation=elevation,
                    metadata=metadata,
                    elevation_stats=stats,
                    terrain_summary=terrain_summary,
                    hillshade=hillshade,
                    contour_geojson=contour_geojson,
                    layer_opacity=opacity_slider
                )
                st_folium(
                    map_terrain,
                    width="100%",
                    height=510,
                    returned_objects=[],
                    key="compare_map_terrain"
                )
            with col_c2:
                st.markdown("##### 2. Waterlogging Vulnerability & Proposed Trench")
                map_drainage = create_focused_map(
                    mode="drainage",
                    elevation=elevation,
                    metadata=metadata,
                    elevation_stats=stats,
                    terrain_summary=terrain_summary,
                    hillshade=hillshade,
                    depression_depth=hydro_result.depression_depth_array,
                    drainage_routes=drainage_plan.routes,
                    candidate_outlets=drainage_plan.candidate_outlets,
                    extracted_ditches_geojson=ditch_result.ditches_geojson,
                    layer_opacity=opacity_slider
                )
                st_folium(
                    map_drainage,
                    width="100%",
                    height=510,
                    returned_objects=[],
                    key="compare_map_drainage"
                )
        else:
            st.caption("Drag the emerald vertical divider left and right to peel between Raw Terrain (Left) and Waterlogging Risk & Channels (Right).")
            swipe_map = create_side_by_side_swipe_map(
                elevation=elevation,
                metadata=metadata,
                elevation_stats=stats,
                hillshade=hillshade,
                depression_depth=hydro_result.depression_depth_array,
                drainage_routes=drainage_plan.routes,
                layer_opacity=opacity_slider
            )
            st_folium(
                swipe_map,
                width="100%",
                height=560,
                returned_objects=[],
                key="compare_map_swipe"
            )

    with tab_exports:
        st.markdown("#### 📑 Automated Engineering Reports & GIS Data Exports")
        st.caption(
            "All exported geospatial datasets strictly preserve native metric coordinate systems "
            f"({stats.crs_display}) for immediate drag-and-drop verification in QGIS, ArcGIS Pro, or CAD."
        )

        badge_name = data_source_badge
        plot_name = Path(source_name).stem.replace("_", " ").title()

        exp_col1, exp_col2 = st.columns(2)

        with exp_col1:
            st.markdown(f"""
            <div class="export-card">
                <div style="font-size: 1.05rem; font-weight: 700; color: #2ECC71; margin-bottom: 4px;">
                    📄 Complete Drainage Feasibility Report (PDF)
                </div>
                <div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 12px;">
                    Publication-grade multi-page engineering document including executive summary, landholding KPIs,
                    high-DPI cartographic maps, ranked ditch specifications table, and agronomic guidelines.
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.spinner("Compiling publication-grade PDF report..."):
                pdf_bytes = get_cached_pdf_report(
                    source_key=source_key,
                    data_source_badge=badge_name,
                    field_plot_name=plot_name,
                    flow_accum_threshold=float(flow_accum_threshold),
                    min_waterlogging_depth_m=float(sink_threshold_m),
                    min_waterlogging_area_m2=float(min_sink_area),
                    min_bed_slope_pct=float(min_ditch_slope_pct),
                    file_bytes=file_bytes,
                    is_lidar=is_lidar
                )

            st.download_button(
                label="📥 Download Drainage Plan Report (PDF)",
                data=pdf_bytes,
                file_name=f"{Path(source_name).stem}_drainage_plan_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )

            st.markdown(f"""
            <div class="export-card" style="margin-top: 14px;">
                <div style="font-size: 1.05rem; font-weight: 700; color: #00E5FF; margin-bottom: 4px;">
                    🗺️ Proposed Drainage Routes (GeoJSON)
                </div>
                <div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 12px;">
                    Vector LineStrings with engineered attributes: length (m), natural slope (%),
                    monotonic bed slope (%), max cut depth (m), and estimated excavation volume (m³).
                </div>
            </div>
            """, unsafe_allow_html=True)

            routes_geojson_str = json.dumps(drainage_plan.routes_geojson, indent=2)
            st.download_button(
                label="📥 Download Proposed Routes (GeoJSON)",
                data=routes_geojson_str,
                file_name="recommended_routes.geojson",
                mime="application/geo+json",
                use_container_width=True
            )

        with exp_col2:
            st.markdown(f"""
            <div class="export-card">
                <div style="font-size: 1.05rem; font-weight: 700; color: #F59E0B; margin-bottom: 4px;">
                    📡 Extracted Ditches & Culverts (GeoJSON)
                </div>
                <div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 12px;">
                    Research-backed ditch centerlines (Roelens et al. 2018) and bridged driveway/culvert gaps
                    ready for overlay against farm boundary shapefiles.
                </div>
            </div>
            """, unsafe_allow_html=True)

            ditches_geojson_str = json.dumps(ditch_result.ditches_geojson, indent=2)
            st.download_button(
                label="📥 Download Extracted Ditches & Culverts (GeoJSON)",
                data=ditches_geojson_str,
                file_name="extracted_ditches.geojson",
                mime="application/geo+json",
                use_container_width=True
            )

            st.markdown(f"""
            <div class="export-card" style="margin-top: 14px;">
                <div style="font-size: 1.05rem; font-weight: 700; color: #38BDF8; margin-bottom: 4px;">
                    🌊 Waterlogging Depth Raster (GeoTIFF)
                </div>
                <div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 12px;">
                    Processed depression depth raster (Filled DTM − Raw DTM) in 32-bit floating point,
                    providing exact puddle water depth in meters per pixel.
                </div>
            </div>
            """, unsafe_allow_html=True)

            try:
                with open(hydro_result.depression_depth_path, "rb") as f_dep:
                    dep_raster_bytes = f_dep.read()
                st.download_button(
                    label="📥 Download Waterlogging Depth (GeoTIFF)",
                    data=dep_raster_bytes,
                    file_name="waterlogging_depth.tif",
                    mime="image/tiff",
                    use_container_width=True
                )
            except Exception as e:
                st.warning(f"Could not read depression depth raster: {e}")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- TABBED ENGINEERING SPECIFICATIONS TABLES ---
    st.markdown("### 📋 Engineering Specification & Field Extraction Tables")
    tab_ditches_tbl, tab_routes_tbl, tab_sinks_tbl = st.tabs([
        "📡 Extracted Ditches & Culverts (Roelens LiDAR)",
        "📋 Candidate Drainage Routes Comparison Table",
        "🌊 Identified Waterlogging Depressions (Sinks)"
    ])

    with tab_ditches_tbl:
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

    with tab_routes_tbl:
        st.markdown("##### Candidate Surface Drainage Routes Comparison")
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

    with tab_sinks_tbl:
        st.markdown("##### Waterlogging Sinks & Pockets Prioritization")
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
