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
from components.maps import create_terrain_map

# Configure Streamlit Page
st.set_page_config(
    page_title="Field Drainage Using LiDAR | Hydrological Modeling",
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


def main():
    # Application Title & Subtitle
    st.markdown('<div class="main-header">🌾 Field Drainage Using LiDAR</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Smart Farm Drainage Assistant &bull; Phase 3: Hydrological Modeling &amp; Water Flow Concentration</div>',
        unsafe_allow_html=True
    )

    # --- SIDEBAR CONTROLS ---
    with st.sidebar:
        st.header("🌱 Field Data & Map Setup")
        
        sample_path = Path("data/sample/synthetic_indian_farm.tif")
        load_sample = st.button("🌱 Load Sample 5-Acre Farm Demo", use_container_width=True)

        uploaded_file = st.file_uploader(
            "Upload Your Field Elevation Map (.tif, .tiff)",
            type=["tif", "tiff"],
            help="Upload your drone LiDAR or satellite elevation map (preferably in metric UTM meters)."
        )

        st.markdown("---")
        st.subheader("🗺️ Map View & Layers")
        
        default_layer_choice = st.selectbox(
            "Main Map Layer",
            options=[
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

    # Determine which raster source to process
    source_key = None
    file_bytes = None
    source_name = ""

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        source_key = f"upload_{uploaded_file.name}_{len(file_bytes)}"
        source_name = uploaded_file.name
    elif load_sample or "loaded_sample" in st.session_state:
        st.session_state["loaded_sample"] = True
        if sample_path.exists():
            source_key = str(sample_path)
            source_name = "Sample 5-Acre Indian Farm (UTM Zone 44N)"
        else:
            st.error(f"Sample data file not found at `{sample_path}`. Run `python scripts/generate_synthetic_farm.py` first.")
            return

    # If no data is selected yet, display initial instructions
    if source_key is None:
        st.info("👈 **Get Started:** Click **'Load Sample 5-Acre Farm Demo'** in the sidebar to see an instant demo, or upload your own field elevation file.")
        
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.markdown("""
            ### 🎯 What This Tool Does For You
            - **Easy Field Map Upload:** Ingests elevation maps from drone surveys or open satellite data.
            - **Accurate Distance & Area Checks:** Validates real metric units (meters, acres, hectares) so channel lengths are precise.
            - **D8 Flow Direction & Accumulation:** Tracks upstream contributing area cell-by-cell to map runoff concentration.
            - **Natural Runoff Extraction:** Automatically traces natural gullies and surface streams with precise lengths.
            - **Waterlogging Risk Spots:** Detects trapped hollows and estimates how much water could pool in them after heavy rains.
            """)
        with col_info2:
            st.markdown("""
            ### 🚜 Why Drainage Matters for Indian Soils
            - **Black Cotton Soils (Vertisols):** Swell when wet, stopping water soak-in and suffocating crop roots. Areas flatter than 0.5% slope need planned trenches.
            - **Field Bunds (*Medh / Daul*):** Earthen boundary ridges frequently trap rainwater inside plots if no drainage outlets exist.
            - **Smart Drainage Planning:** Pinpoints exact puddle zones so you can dig direct gravity-fed channels to field outlets.
            """)
        return

    # --- PROCESS RASTER, TERRAIN & HYDROLOGY ENGINE (CACHED) ---
    try:
        with st.spinner("Analyzing field topography, slope, and WhiteboxTools hydrology..."):
            # 1. Cached Elevation & Metadata
            elevation, metadata, stats = get_cached_elevation_and_stats(
                source_key=source_key,
                file_bytes=file_bytes
            )

            # 2. Cached Terrain Metrics (Slope, Aspect, Hillshade, Contours)
            (
                terrain_summary,
                slope_deg,
                slope_pct,
                aspect,
                hillshade,
                contour_geojson
            ) = get_cached_terrain_analysis(
                source_key=source_key,
                min_depth_m=sink_threshold_m,
                contour_interval=contour_interval,
                file_bytes=file_bytes
            )

            # 3. Cached WhiteboxTools Hydrology (D8 Flow, Accumulation, Streams, Basins)
            hydro_result = get_cached_hydrological_analysis(
                source_key=source_key,
                flow_accum_threshold=float(flow_accum_threshold),
                min_waterlogging_depth_m=sink_threshold_m,
                file_bytes=file_bytes
            )
    except Exception as e:
        st.error(f"❌ Error during hydrological processing: {str(e)}")
        return

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
    st.caption("💡 Tip: Click the layer icon in the top-right of the map to toggle between Runoff Paths, Flow Accumulation, Waterlogging Zones, DTM, Slope, and Sub-Basins.")
    
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
            colormap_name=colormap_choice,
            layer_opacity=opacity_slider,
            default_layer=default_layer_choice
        )
        st_folium(
            fmap,
            width="100%",
            height=570,
            returned_objects=[],
            key="field_drainage_folium_map"
        )

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
        with c4:
            st.write("**WhiteboxTools Hydrology Profile:**")
            st.json(hydro_result.to_dict())


if __name__ == "__main__":
    main()
