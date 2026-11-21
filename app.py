"""
Field Drainage Using LiDAR - Phase 1 Web Application
Intelligent LiDAR-Based Agricultural Field Drainage Planning System
Interactive Streamlit Dashboard & Elevation Viewer
"""

import os
from pathlib import Path
import streamlit as st
from streamlit_folium import st_folium

from modules.data_loader import load_raster
from modules.elevation_analysis import compute_elevation_stats
from components.maps import create_elevation_map

# Configure Streamlit Page
st.set_page_config(
    page_title="Field Drainage Using LiDAR | Elevation Viewer",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Agronomic Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1b4332;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #40916c;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 5px solid #2d6a4f;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .metric-label {
        font-size: 0.85rem;
        color: #6c757d;
        font-weight: 600;
        text-transform: uppercase;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1b4332;
    }
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Application Title & Subtitle
    st.markdown('<div class="main-header">🌾 Field Drainage Using LiDAR</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Intelligent Agricultural Field Drainage Planning System &bull; Phase 1: Foundation & Elevation Viewer</div>',
        unsafe_allow_html=True
    )

    # --- SIDEBAR CONTROLS ---
    with st.sidebar:
        st.header("⚙️ Data & Field Settings")
        
        sample_path = Path("data/sample/synthetic_indian_farm.tif")
        load_sample = st.button("🌱 Load Sample Indian Farm (5-Acre Demo)", use_container_width=True)

        uploaded_file = st.file_uploader(
            "Upload Elevation GeoTIFF (.tif, .tiff)",
            type=["tif", "tiff"],
            help="Upload a bare-earth DTM or elevation raster (preferably in projected metric UTM coordinates)."
        )

        st.markdown("---")
        st.subheader("🎨 Map Visualization")
        colormap_choice = st.selectbox(
            "Colormap Palette",
            options=["terrain", "viridis", "plasma", "YlGnBu", "gist_earth"],
            index=0,
            help="Select color ramp for visualizing elevation gradients."
        )

        opacity_slider = st.slider(
            "Layer Opacity",
            min_value=0.2,
            max_value=1.0,
            value=0.75,
            step=0.05
        )

        st.markdown("---")
        st.caption("🛡️ **System Architecture:** 100% Free Open-Source Stack (Python, Rasterio, Folium, Streamlit).")
        st.caption("© 2026 Varad Patil. All Rights Reserved.")

    # Determine which raster source to process
    raster_source = None
    source_name = ""

    if uploaded_file is not None:
        raster_source = uploaded_file
        source_name = uploaded_file.name
    elif load_sample or "loaded_sample" in st.session_state:
        st.session_state["loaded_sample"] = True
        if sample_path.exists():
            raster_source = str(sample_path)
            source_name = "Synthetic Indian Farm (5-Acre Demo, EPSG:32644)"
        else:
            st.error(f"Sample data file not found at `{sample_path}`. Run `python scripts/generate_synthetic_farm.py` first.")
            return

    # If no data is selected yet, display initial instructions
    if raster_source is None:
        st.info("👈 **Get Started:** Click **'Load Sample Indian Farm'** in the sidebar for an instant demonstration, or upload your own field GeoTIFF.")
        
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.markdown("""
            ### 🎯 Phase 1 Capabilities
            - **Robust GeoTIFF Ingestion:** Validates raster integrity, affine bounds, and spatial projection.
            - **Metric CRS Preservation:** Audits Indian UTM zones (e.g., EPSG:32643, EPSG:32644) and warns if unprojected degrees are used.
            - **NoData & Outlier Masking:** Automatically handles negative invalid nodata values (`-9999`, `-32768`).
            - **High-Precision Statistics:** Computes true field acreage, relief range, and statistical dispersion.
            """)
        with col_info2:
            st.markdown("""
            ### 🚜 Why Surface Drainage Matters in India
            - **Black Cotton Soils (Vertisols):** Swell on wetting, dropping hydraulic conductivity to near-zero and suffocating crop roots.
            - **Field Bund Obstacles:** Perimeter ridges (*medh/daul*) trap storm runoff in micro-depressions.
            - **Phase 1 Foundation:** Sets up the certified coordinate reference pipeline required for downstream Whitebox hydrological routing and Roelens ditch extraction.
            """)
        return

    # --- PROCESS RASTER ---
    try:
        with st.spinner("Analyzing elevation raster and calculating terrain statistics..."):
            elevation, metadata = load_raster(raster_source)
            stats = compute_elevation_stats(elevation, metadata)
    except Exception as e:
        st.error(f"❌ Error loading raster: {str(e)}")
        return

    # --- CRS ALERT CHECKS ---
    if metadata.is_geographic:
        st.warning(
            "⚠️ **Geographic CRS Detected (Degrees):** This raster uses latitude/longitude degrees rather than projected meters. "
            "Surface area and distance calculations are approximated. For sub-meter drainage accuracy, reproject to an Indian UTM Zone (e.g., EPSG:32643 or EPSG:32644)."
        )
    else:
        st.success(f"✅ **Validated Metric Coordinate System:** {stats.crs_display} &bull; Native units: **{metadata.units}**")

    # --- KPI METRIC CARDS ---
    st.markdown("### 📊 Field Elevation & Terrain Metrics")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Min Elevation</div>
            <div class="metric-value">{stats.min_elevation:.2f} m</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Max Elevation</div>
            <div class="metric-value">{stats.max_elevation:.2f} m</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Relief Range</div>
            <div class="metric-value">{stats.elevation_range:.2f} m</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Mean Elevation</div>
            <div class="metric-value">{stats.mean_elevation:.2f} m</div>
        </div>
        """, unsafe_allow_html=True)

    with m5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Field Area</div>
            <div class="metric-value">{stats.area_acres:.2f} ac</div>
        </div>
        """, unsafe_allow_html=True)

    with m6:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Resolution</div>
            <div class="metric-value">{stats.spatial_resolution_m:.1f} m/px</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- INTERACTIVE MAP VIEWPORT ---
    st.markdown(f"### 🗺️ Interactive Elevation Viewer &bull; *{source_name}*")
    
    with st.spinner("Rendering interactive geospatial Leaflet overlay..."):
        fmap = create_elevation_map(
            elevation=elevation,
            metadata=metadata,
            stats=stats,
            colormap_name=colormap_choice,
            opacity=opacity_slider
        )
        st_folium(fmap, width="100%", height=550)

    # --- DETAILED METADATA & DATA TABLE EXPANDER ---
    with st.expander("🔍 View Technical Geospatial Metadata"):
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Raster Dimensions & Pixels:**")
            st.json({
                "Grid Width": metadata.width,
                "Grid Height": metadata.height,
                "Valid Data Pixels": stats.valid_pixels,
                "NoData Pixels": stats.nodata_pixels,
                "Coverage": f"{stats.valid_fraction*100:.2f}%",
                "Pixel Dimensions": f"{metadata.res_x:.4f} x {metadata.res_y:.4f} ({metadata.units})"
            })
        with c2:
            st.write("**Spatial Coordinate Extents:**")
            st.json({
                "CRS": stats.crs_display,
                "Projected Metric": metadata.is_projected,
                "Native Bounds (Left, Bottom, Right, Top)": [round(b, 2) for b in metadata.bounds_native],
                "WGS84 Bounds (Lon Min, Lat Min, Lon Max, Lat Max)": [round(b, 6) for b in metadata.bounds_wgs84],
                "Calculated Area (m²)": f"{stats.area_sq_m:,.1f} m²",
                "Calculated Area (Hectares)": f"{stats.area_ha:.3f} ha"
            })


if __name__ == "__main__":
    main()
