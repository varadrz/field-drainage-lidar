"""
Field Drainage Using LiDAR - Phase 2 Web Application
Intelligent LiDAR-Based Agricultural Field Drainage Planning System
Interactive Streamlit Dashboard & Terrain Analysis Engine
"""

import os
from pathlib import Path
import streamlit as st
from streamlit_folium import st_folium

from modules.data_loader import load_raster
from modules.elevation_analysis import compute_elevation_stats
from modules.terrain_analysis import compute_terrain_summary, extract_contours_geojson
from components.maps import create_terrain_map

# Configure Streamlit Page
st.set_page_config(
    page_title="Field Drainage Using LiDAR | Land & Waterlogging Viewer",
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
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Application Title & Subtitle
    st.markdown('<div class="main-header">🌾 Field Drainage Using LiDAR</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Smart Farm Drainage Assistant &bull; Land Slope, Elevation &amp; Waterlogging Risk Viewer</div>',
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
                "Land Slope (% Grade)",
                "3D Land Shape (Hillshade)",
                "Slope Direction (Aspect)"
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
        st.subheader("⚙️ Drainage Settings")
        
        contour_interval = st.selectbox(
            "Elevation Line Gap (m)",
            options=[0.1, 0.25, 0.5, 1.0],
            index=1,
            help="Height step between contour lines (e.g., 0.25 m shows fine micro-slope details across flat plots)."
        )

        sink_threshold_m = st.slider(
            "Minimum Puddle Depth to Flag (m)",
            min_value=0.02,
            max_value=0.20,
            value=0.05,
            step=0.01,
            help="Ignores small plow furrows and soil clods below this depth so only real waterlogging spots are highlighted."
        )

        st.markdown("---")
        st.caption("🛡️ **Smart Agri-Drainage:** Built with 100% free, open-source technology for farmers and agronomists.")
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
            source_name = "Sample 5-Acre Indian Farm (UTM Zone 44N)"
        else:
            st.error(f"Sample data file not found at `{sample_path}`. Run `python scripts/generate_synthetic_farm.py` first.")
            return

    # If no data is selected yet, display initial instructions
    if raster_source is None:
        st.info("👈 **Get Started:** Click **'Load Sample 5-Acre Farm Demo'** in the sidebar to see an instant demo, or upload your own field elevation file.")
        
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.markdown("""
            ### 🎯 What This Tool Does For You
            - **Easy Field Map Upload:** Ingests elevation maps from drone surveys or open satellite data.
            - **Accurate Distance & Area Checks:** Validates real metric units (meters, acres, hectares) so channel lengths are precise.
            - **Automatic Data Cleanup:** Cleans missing boundary values and bad sensor readings automatically.
            - **Slope & Natural Drainage Flow:** Calculates exact land slope (%) and shows where runoff naturally heads.
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

    # --- PROCESS RASTER & TERRAIN ENGINE ---
    try:
        with st.spinner("Analyzing field topography, slope, and waterlogging risks..."):
            elevation, metadata = load_raster(raster_source)
            stats = compute_elevation_stats(elevation, metadata)
            (
                terrain_summary,
                slope_deg,
                slope_pct,
                aspect,
                hillshade,
                dep_depth
            ) = compute_terrain_summary(elevation, metadata, min_depth_m=sink_threshold_m)

            contour_geojson = extract_contours_geojson(
                elevation,
                metadata,
                interval=contour_interval
            )
    except Exception as e:
        st.error(f"❌ Error during terrain processing: {str(e)}")
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

    # --- TERRAIN & DRAINAGE SUMMARY PANEL ---
    st.markdown("### 🚜 Land Slope & Waterlogging Risk Summary")
    t1, t2, t3, t4, t5 = st.columns(5)

    with t1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Average Land Slope</div>
            <div class="metric-value">{terrain_summary.mean_slope_pct:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with t2:
        st.markdown(f"""
        <div class="metric-card-alert">
            <div class="metric-label">Flat / Poor Drainage (&lt;0.5%)</div>
            <div class="metric-value-alert">{terrain_summary.flat_area_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with t3:
        st.markdown(f"""
        <div class="metric-card-sink">
            <div class="metric-label">Waterlogging Spots</div>
            <div class="metric-value-sink">{terrain_summary.depression_count} spots</div>
        </div>
        """, unsafe_allow_html=True)

    with t4:
        st.markdown(f"""
        <div class="metric-card-sink">
            <div class="metric-label">Total Ponding Area</div>
            <div class="metric-value-sink">{terrain_summary.depression_acres:.2f} ac</div>
        </div>
        """, unsafe_allow_html=True)

    with t5:
        st.markdown(f"""
        <div class="metric-card-sink">
            <div class="metric-label">Est. Trapped Water</div>
            <div class="metric-value-sink">{terrain_summary.est_depression_volume_m3:,.0f} m³</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- INTERACTIVE MAP VIEWPORT ---
    st.markdown(f"### 🗺️ Interactive Field Drainage Map &bull; *{source_name}*")
    st.caption("💡 Tip: Click the layer icon in the top-right of the map to turn individual map views on or off.")
    
    with st.spinner("Rendering interactive field map..."):
        fmap = create_terrain_map(
            elevation=elevation,
            metadata=metadata,
            elevation_stats=stats,
            terrain_summary=terrain_summary,
            slope_pct=slope_pct,
            aspect=aspect,
            hillshade=hillshade,
            depression_depth=dep_depth,
            contour_geojson=contour_geojson,
            colormap_name=colormap_choice,
            layer_opacity=opacity_slider,
            default_layer=default_layer_choice
        )
        st_folium(fmap, width="100%", height=560)

    # --- DETAILED METADATA & DATA TABLE EXPANDER ---
    with st.expander("🔍 Technical Map & Measurement Details (Advanced)"):
        c1, c2, c3 = st.columns(3)
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
            st.write("**Slope & Waterlogging Breakdown:**")
            st.json(terrain_summary.to_dict())


if __name__ == "__main__":
    main()
