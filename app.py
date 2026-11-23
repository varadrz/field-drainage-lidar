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
    page_title="Field Drainage Using LiDAR | Terrain Analysis",
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
        '<div class="sub-header">Intelligent Agricultural Field Drainage Planning System &bull; Phase 2: Terrain Analysis & Surface Viewer</div>',
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
        st.subheader("🎨 Map & Layer Controls")
        
        default_layer_choice = st.selectbox(
            "Primary Viewport Focus",
            options=[
                "Elevation Surface (DTM)",
                "Slope Map (% Grade)",
                "Hillshade (Shaded Relief)",
                "Aspect Map (Direction)"
            ],
            index=0,
            help="Select the initial active layer overlay on the map. You can also toggle layers dynamically on the top-right map menu."
        )

        colormap_choice = st.selectbox(
            "Elevation Colormap Palette",
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
        st.subheader("📐 Terrain Parameters")
        
        contour_interval = st.selectbox(
            "Contour Interval (m)",
            options=[0.1, 0.25, 0.5, 1.0],
            index=1,
            help="Vertical contour interval for vector contour extraction."
        )

        sink_threshold_m = st.slider(
            "Min Depression Depth (m)",
            min_value=0.02,
            max_value=0.20,
            value=0.05,
            step=0.01,
            help="Minimum sink depth to isolate genuine waterlogging depressions from tillage soil roughness."
        )

        st.markdown("---")
        st.caption("🛡️ **System Architecture:** 100% Free Open-Source Stack (Python, Rasterio, SciPy, Folium, Streamlit).")
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
            ### 🎯 Phase 2: Terrain Analysis Capabilities
            - **Horn 3×3 Slope & Aspect:** Calculates continuous slope in degrees and agronomic percent grade (`%`), plus 360° compass aspect.
            - **Lambertian Shaded Relief (Hillshade):** Renders micro-topography, bund ridges, and ditches in clear relief.
            - **Vector Contour Extraction:** Computes sub-meter vector contour lines (GeoJSON) dynamically mapped onto Leaflet.
            - **Priority-Flood Sink Detection:** Pinpoints natural waterlogging depressions and estimates stored water volumes.
            """)
        with col_info2:
            st.markdown("""
            ### 🚜 Agronomic Context: Indian Farmland
            - **Drainage Thresholds:** Lands with $<0.5\%$ slope require engineered surface channels to prevent ponding in Vertisols.
            - **Earthen Perimeter Bunds (*Medh/Daul*):** Frequently create artificial catchment basins by blocking natural overland runoff.
            - **Automated Hotspot Sizing:** Calculates depression area and required excavation volume for subsequent gravity alignment.
            """)
        return

    # --- PROCESS RASTER & TERRAIN ENGINE ---
    try:
        with st.spinner("Executing terrain analysis engine (Slope, Aspect, Hillshade, Contours, Sinks)..."):
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
            "⚠️ **Geographic CRS Detected (Degrees):** This raster uses latitude/longitude degrees rather than projected meters. "
            "Surface area and distance calculations are approximated. Reproject to an Indian UTM Zone (e.g., EPSG:32643 or EPSG:32644) for millimeter-grade accuracy."
        )
    else:
        st.success(f"✅ **Validated Metric Coordinate System:** {stats.crs_display} &bull; Native units: **{metadata.units}** &bull; Resolution: **{stats.spatial_resolution_m:.2f} m/pixel**")

    # --- KPI METRIC CARDS (ELEVATION & TERRAIN) ---
    st.markdown("### 📊 Field Elevation & Macro-Relief Metrics")
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
            <div class="metric-label">Grid Resolution</div>
            <div class="metric-value">{stats.spatial_resolution_m:.1f} m/px</div>
        </div>
        """, unsafe_allow_html=True)

    # --- TERRAIN & DRAINAGE SUMMARY PANEL ---
    st.markdown("### 🚜 Terrain Slope & Waterlogging Vulnerability Panel")
    t1, t2, t3, t4, t5 = st.columns(5)

    with t1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Average Slope</div>
            <div class="metric-value">{terrain_summary.mean_slope_pct:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with t2:
        st.markdown(f"""
        <div class="metric-card-alert">
            <div class="metric-label">Poorly Drained (&lt;0.5%)</div>
            <div class="metric-value-alert">{terrain_summary.flat_area_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with t3:
        st.markdown(f"""
        <div class="metric-card-sink">
            <div class="metric-label">Depression Hotspots</div>
            <div class="metric-value-sink">{terrain_summary.depression_count} clusters</div>
        </div>
        """, unsafe_allow_html=True)

    with t4:
        st.markdown(f"""
        <div class="metric-card-sink">
            <div class="metric-label">Total Sink Area</div>
            <div class="metric-value-sink">{terrain_summary.depression_acres:.2f} ac</div>
        </div>
        """, unsafe_allow_html=True)

    with t5:
        st.markdown(f"""
        <div class="metric-card-sink">
            <div class="metric-label">Est. Stored Water</div>
            <div class="metric-value-sink">{terrain_summary.est_depression_volume_m3:,.0f} m³</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- INTERACTIVE MAP VIEWPORT ---
    st.markdown(f"### 🗺️ Interactive Multi-Layer GIS Viewer &bull; *{source_name}*")
    st.caption("💡 Use the top-right layer switch menu on the map to toggle between DTM, Slope, Aspect, Hillshade, Contours, and Waterlogging Depressions.")
    
    with st.spinner("Rendering interactive geospatial Leaflet overlay..."):
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
    with st.expander("🔍 View Technical Geospatial & Terrain Metrics"):
        c1, c2, c3 = st.columns(3)
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
                "Native Bounds": [round(b, 2) for b in metadata.bounds_native],
                "WGS84 Bounds": [round(b, 6) for b in metadata.bounds_wgs84],
                "Calculated Area (m²)": f"{stats.area_sq_m:,.1f} m²",
                "Calculated Area (Hectares)": f"{stats.area_ha:.3f} ha"
            })
        with c3:
            st.write("**Terrain Slope & Depression Profile:**")
            st.json(terrain_summary.to_dict())


if __name__ == "__main__":
    main()
