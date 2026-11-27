"""
Interactive Web Map Component
Builds responsive Folium / Leaflet maps with styled colored raster overlays,
multi-layer controls (Elevation, Slope, Aspect, Hillshade, Contours, Depressions,
Natural Runoff Paths, Flow Accumulation, Sub-Catchments),
WGS84 coordinate handling, and dynamic colorbar legends.
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

# Configure temporary Matplotlib cache directory to avoid permission issues in restricted environments
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))

import numpy as np
import folium
from folium.raster_layers import ImageOverlay
from folium.plugins import AntPath
import branca.colormap as cm
import matplotlib.pyplot as plt

from modules.data_loader import RasterMetadata
from modules.elevation_analysis import ElevationStats
from modules.terrain_analysis import TerrainSummary


def matplotlib_color_to_hex(rgba_tuple: Tuple[float, float, float, float]) -> str:
    """Converts a matplotlib RGBA float tuple (0-1) to an HTML hex color string."""
    r, g, b = int(rgba_tuple[0] * 255), int(rgba_tuple[1] * 255), int(rgba_tuple[2] * 255)
    return f"#{r:02x}{g:02x}{b:02x}"


def array_to_rgba(
    data: np.ndarray,
    colormap_name: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    alpha: float = 1.0,
    nan_transparent: bool = True
) -> np.ndarray:
    """
    Transforms a 2D float/int array into an RGBA (0-255 uint8) image using a Matplotlib colormap.
    
    Args:
        data: 2D numpy array with NaNs.
        colormap_name: Name of Matplotlib colormap (e.g. 'terrain', 'YlOrRd', 'twilight', 'Blues').
        vmin: Minimum value for normalization (defaults to np.nanmin(data)).
        vmax: Maximum value for normalization (defaults to np.nanmax(data)).
        alpha: Master opacity multiplier (0.0 - 1.0).
        nan_transparent: If True, NaN or masked pixels are completely transparent.
        
    Returns:
        2D uint8 numpy array of shape (height, width, 4).
    """
    valid_mask = ~np.isnan(data)
    if not np.any(valid_mask):
        return np.zeros((data.shape[0], data.shape[1], 4), dtype=np.uint8)

    min_val = vmin if vmin is not None else float(np.nanmin(data))
    max_val = vmax if vmax is not None else float(np.nanmax(data))
    val_range = max_val - min_val if (max_val - min_val) > 1e-5 else 1.0

    normalized = np.zeros_like(data, dtype=np.float32)
    normalized[valid_mask] = (data[valid_mask] - min_val) / val_range
    normalized = np.clip(normalized, 0.0, 1.0)

    try:
        cmap = plt.colormaps[colormap_name]
    except (AttributeError, KeyError):
        cmap = plt.cm.get_cmap(colormap_name)

    rgba_img = (cmap(normalized) * 255).astype(np.uint8)
    if alpha < 1.0:
        rgba_img[..., 3] = (rgba_img[..., 3] * alpha).astype(np.uint8)

    if nan_transparent:
        rgba_img[~valid_mask] = [0, 0, 0, 0]

    return rgba_img


def create_terrain_map(
    elevation: np.ndarray,
    metadata: RasterMetadata,
    elevation_stats: ElevationStats,
    terrain_summary: Optional[TerrainSummary] = None,
    slope_pct: Optional[np.ndarray] = None,
    aspect: Optional[np.ndarray] = None,
    hillshade: Optional[np.ndarray] = None,
    depression_depth: Optional[np.ndarray] = None,
    contour_geojson: Optional[Dict[str, Any]] = None,
    flow_accum: Optional[np.ndarray] = None,
    streams_geojson: Optional[Dict[str, Any]] = None,
    basins_geojson: Optional[Dict[str, Any]] = None,
    drainage_routes: Optional[List[Any]] = None,
    priority_sinks: Optional[List[Any]] = None,
    candidate_outlets: Optional[List[Any]] = None,
    extracted_ditches_geojson: Optional[Dict[str, Any]] = None,
    reconnected_culverts_geojson: Optional[Dict[str, Any]] = None,
    colormap_name: str = "terrain",
    layer_opacity: float = 0.75,
    default_layer: str = "Land Elevation Map",
    animate_flow: bool = True
) -> folium.Map:
    """
    Renders an interactive multi-layer GIS map with Elevation, Slope, Aspect,
    Hillshade, Vector Contours, Waterlogging Depressions, Natural Runoff Streams,
    Flow Accumulation, and Sub-Basins.
    
    Returns:
        folium.Map object ready for Streamlit display.
    """
    min_lon, min_lat, max_lon, max_lat = metadata.bounds_wgs84
    center_lat = (min_lat + max_lat) / 2.0
    center_lon = (min_lon + max_lon) / 2.0
    leaflet_bounds = [[min_lat, min_lon], [max_lat, max_lon]]

    # Initialize Base Folium Map
    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=18,
        max_zoom=22,
        tiles=None
    )

    # 1. Base OpenStreetMap (Alternative)
    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr="OpenStreetMap",
        name="OpenStreetMap",
        overlay=False,
        control=True
    ).add_to(fmap)

    # 2. Base Satellite Imagery (Esri)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite Imagery (Esri)",
        overlay=False,
        control=True
    ).add_to(fmap)

    # 3. Base Dark Minimal (CartoDB) - Active default dark basemap
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr="&copy; OpenStreetMap contributors &copy; CARTO",
        subdomains="abcd",
        name="Dark Minimal (CartoDB)",
        overlay=False,
        control=True
    ).add_to(fmap)

    # Determine which layer to activate initially
    is_elev_active = default_layer in [
        "Land Elevation Map",
        "Elevation Surface (DTM)",
        "Land Elevation Map (DTM)",
        "Extracted Ditches & Culverts (LiDAR)",
        "Proposed Surface Drainage Routes"
    ]
    is_slope_active = default_layer in ["Land Slope (% Grade)", "Slope Map (% Grade)"]
    is_hs_active = default_layer in ["3D Land Shape (Hillshade)", "Hillshade (Shaded Relief)"]
    is_aspect_active = default_layer in ["Slope Direction (Aspect)", "Aspect Map (Direction)"]
    is_flow_active = default_layer in ["Water Flow Concentration", "Water Flow Concentration (Accumulation)"]

    # 4. Layer: Hillshade (Shaded Relief)
    if hillshade is not None:
        valid_mask = ~np.isnan(elevation)
        hs_rgba = np.zeros((hillshade.shape[0], hillshade.shape[1], 4), dtype=np.uint8)
        hs_rgba[..., 0] = hillshade
        hs_rgba[..., 1] = hillshade
        hs_rgba[..., 2] = hillshade
        hs_rgba[..., 3] = int(255 * layer_opacity)
        hs_rgba[~valid_mask] = [0, 0, 0, 0]

        ImageOverlay(
            image=hs_rgba,
            bounds=leaflet_bounds,
            opacity=layer_opacity,
            name="3D Land Shape (Hillshade)",
            show=is_hs_active,
            interactive=True,
            zindex=2
        ).add_to(fmap)

    # 5. Layer: Elevation Surface (DTM)
    elev_rgba = array_to_rgba(
        elevation,
        colormap_name=colormap_name,
        vmin=elevation_stats.min_elevation,
        vmax=elevation_stats.max_elevation,
        alpha=layer_opacity
    )
    ImageOverlay(
        image=elev_rgba,
        bounds=leaflet_bounds,
        opacity=layer_opacity,
        name="Land Elevation Map (DTM)",
        show=is_elev_active,
        interactive=True,
        zindex=3
    ).add_to(fmap)

    # 6. Layer: Slope Map (% Grade)
    if slope_pct is not None:
        valid_slopes = slope_pct[~np.isnan(slope_pct)]
        vmax_slope = float(np.percentile(valid_slopes, 98)) if len(valid_slopes) > 0 else 5.0
        vmax_slope = max(vmax_slope, 2.0)

        slope_rgba = array_to_rgba(
            slope_pct,
            colormap_name="YlOrRd",
            vmin=0.0,
            vmax=vmax_slope,
            alpha=layer_opacity
        )
        ImageOverlay(
            image=slope_rgba,
            bounds=leaflet_bounds,
            opacity=layer_opacity,
            name="Land Slope Map (% Grade)",
            show=is_slope_active,
            interactive=True,
            zindex=4
        ).add_to(fmap)

    # 7. Layer: Water Flow Concentration (Accumulation)
    if flow_accum is not None:
        valid_acc = flow_accum[~np.isnan(flow_accum) & (flow_accum > 0)]
        if len(valid_acc) > 0:
            log_accum = np.zeros_like(flow_accum)
            log_accum[flow_accum > 0] = np.log10(flow_accum[flow_accum > 0] + 1.0)
            log_accum[np.isnan(flow_accum)] = np.nan
            accum_rgba = array_to_rgba(
                log_accum,
                colormap_name="Blues",
                vmin=0.0,
                vmax=float(np.nanmax(log_accum)),
                alpha=layer_opacity
            )
            ImageOverlay(
                image=accum_rgba,
                bounds=leaflet_bounds,
                opacity=layer_opacity,
                name="Water Flow Concentration (Accumulation)",
                show=is_flow_active,
                interactive=True,
                zindex=5
            ).add_to(fmap)

    # 8. Layer: Aspect Map (Compass Direction)
    if aspect is not None:
        aspect_display = np.copy(aspect)
        aspect_display[aspect < 0] = np.nan
        aspect_rgba = array_to_rgba(
            aspect_display,
            colormap_name="twilight",
            vmin=0.0,
            vmax=360.0,
            alpha=layer_opacity
        )
        ImageOverlay(
            image=aspect_rgba,
            bounds=leaflet_bounds,
            opacity=layer_opacity,
            name="Slope Direction (Where Water Flows)",
            show=is_aspect_active,
            interactive=True,
            zindex=5
        ).add_to(fmap)

    # 9. Layer: Field Catchments / Sub-Basins
    if basins_geojson and len(basins_geojson.get("features", [])) > 0:
        folium.GeoJson(
            basins_geojson,
            name="Field Catchments (Sub-Basins)",
            show=(default_layer in ["Field Catchments (Sub-Basins)", "Field Catchments"]),
            style_function=lambda feature: {
                "color": "#a78bfa",
                "weight": 1.5,
                "fillColor": "#8b5cf6",
                "fillOpacity": 0.12,
                "dashArray": "4, 4"
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["basin_id", "area_sq_m"],
                aliases=["Catchment ID:", "Catchment Area (m²):"],
                style="background-color: #14181F; color: #a78bfa; font-size: 12px; font-weight: bold; border-radius: 4px; padding: 4px;"
            )
        ).add_to(fmap)

    # 10. Layer: Waterlogging Depressions (Sinks / Ponding Depth)
    if depression_depth is not None and np.any(depression_depth > 0.0):
        dep_rgba = np.zeros((depression_depth.shape[0], depression_depth.shape[1], 4), dtype=np.uint8)
        dep_mask = depression_depth > 0.0
        
        max_d = float(np.max(depression_depth)) if np.any(dep_mask) else 0.4
        max_d = max(max_d, 0.05)
        norm_d = np.clip(depression_depth / max_d, 0.0, 1.0)
        
        dep_rgba[dep_mask, 0] = (15 + (1.0 - norm_d[dep_mask]) * 35).astype(np.uint8)
        dep_rgba[dep_mask, 1] = (70 + (1.0 - norm_d[dep_mask]) * 130).astype(np.uint8)
        dep_rgba[dep_mask, 2] = 250
        dep_rgba[dep_mask, 3] = 215

        ImageOverlay(
            image=dep_rgba,
            bounds=leaflet_bounds,
            opacity=0.88,
            name="Waterlogging Depth Zones (Ponding)",
            show=True,
            interactive=True,
            zindex=7
        ).add_to(fmap)

    # 11. Layer: Natural Runoff Paths (Extracted Streams)
    if streams_geojson and len(streams_geojson.get("features", [])) > 0:
        if animate_flow:
            stream_group = folium.FeatureGroup(name="Natural Runoff Paths (Streams)", show=True)
            for feat in streams_geojson["features"]:
                geom = feat.get("geometry", {})
                if geom.get("type") == "LineString":
                    coords = geom.get("coordinates", [])
                    lat_lons = [[c[1], c[0]] for c in coords]
                    if len(lat_lons) >= 2:
                        AntPath(
                            locations=lat_lons,
                            delay=900,
                            weight=2.5,
                            color="#00f5d4",
                            pulse_color="#ffffff",
                            dash_array=[8, 16],
                            opacity=0.9
                        ).add_to(stream_group)
            stream_group.add_to(fmap)
        else:
            folium.GeoJson(
                streams_geojson,
                name="Natural Runoff Paths (Streams)",
                show=True,
                style_function=lambda feature: {
                    "color": "#00f5d4",
                    "weight": 2.5,
                    "opacity": 0.95,
                }
            ).add_to(fmap)

    # 12. Layer: Vector Contours (GeoJSON)
    if contour_geojson and len(contour_geojson.get("features", [])) > 0:
        folium.GeoJson(
            contour_geojson,
            name="Elevation Contours",
            show=True,
            style_function=lambda feature: {
                "color": "#e2e8f0",
                "weight": 1.3,
                "opacity": 0.85,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["elevation_str"],
                aliases=["Elevation:"],
                style="background-color: #14181F; color: #F0F4F8; font-size: 12px; font-weight: bold; border-radius: 4px; padding: 4px;"
            )
        ).add_to(fmap)

    # 13. Layer: Priority Waterlogging Sinks
    if priority_sinks:
        sink_group = folium.FeatureGroup(name="Priority Waterlogging Sinks", show=True)
        for sink in priority_sinks:
            lat, lon = sink.pit_coord_wgs84
            folium.CircleMarker(
                location=[lat, lon],
                radius=8,
                color="#00E5FF",
                weight=2.5,
                fill=True,
                fill_color="#00E5FF",
                fill_opacity=0.85,
                tooltip=(
                    f"<b>Priority Sink {sink.sink_id} ({sink.severity})</b><br>"
                    f"Trapped Volume: {sink.volume_m3:,.1f} m³<br>"
                    f"Max Depth: {sink.max_depth_cm:.1f} cm<br>"
                    f"Area: {sink.area_m2:,.0f} m²"
                )
            ).add_to(sink_group)
        sink_group.add_to(fmap)

    # 14. Layer: Candidate Field Outlets
    if candidate_outlets:
        outlet_group = folium.FeatureGroup(name="Candidate Field Outlets", show=True)
        for out in candidate_outlets:
            lat, lon = out.coord_wgs84
            folium.CircleMarker(
                location=[lat, lon],
                radius=7.5,
                color="#FFFFFF",
                weight=2.5,
                fill=True,
                fill_color="#2ECC71",
                fill_opacity=0.95,
                tooltip=(
                    f"<b>Field Outlet #{out.outlet_id}</b><br>"
                    f"Type: {out.outlet_type.replace('_', ' ').title()}<br>"
                    f"Ground Elevation: {out.elevation_m:.2f} m"
                )
            ).add_to(outlet_group)
        outlet_group.add_to(fmap)

    # 15. Layer: Proposed Surface Drainage Routes
    if drainage_routes:
        routes_group = folium.FeatureGroup(name="Proposed Surface Drainage Routes", show=True)
        for r in drainage_routes:
            if r.is_recommended:
                color = "#2ecc71"  # Emerald Green
                pulse_color = "#00f5d4"  # Cyan Pulse
                weight = 5
                dash = None
                label = f"⭐ {r.route_id} (Recommended / Optimal)"
            elif r.feasibility_rank == "Moderate Excavation":
                color = "#f39c12"  # Amber Orange
                pulse_color = "#fef08a"  # Light Yellow Pulse
                weight = 3.5
                dash = "6, 6"
                label = f"{r.route_id} (Moderate Excavation)"
            else:
                color = "#e74c3c"  # Crimson Red
                pulse_color = "#fca5a5"  # Pinkish Pulse
                weight = 3.5
                dash = "4, 8"
                label = f"{r.route_id} (High Excavation)"

            # Tie animation delay (speed) to bed invert slope (%)
            # Steeper routes flow faster (lower delay in ms)
            slope = max(0.1, abs(getattr(r, "bed_slope_pct", getattr(r, "avg_slope_pct", 0.5))))
            pulse_delay = int(np.clip(1800.0 / (1.0 + slope * 2.5), 350, 2200))

            popup_html = f"""
            <div style="font-family: sans-serif; font-size: 12px; color: #14181f; padding: 4px; min-width: 180px;">
                <b style="color: {color}; font-size: 13px;">{label}</b><br>
                <b>Length:</b> {r.length_m:,.1f} m<br>
                <b>Bed Slope:</b> {r.bed_slope_pct:.2f}%<br>
                <b>Max Trench Depth:</b> {r.max_cut_depth_m:.2f} m<br>
                <b>Avg Trench Depth:</b> {r.avg_cut_depth_m:.2f} m<br>
                <b>Excavation Volume:</b> {r.earthwork_vol_m3:,.1f} m³<br>
                <b>Rating:</b> {r.feasibility_rank}
            </div>
            """

            if animate_flow:
                AntPath(
                    locations=r.path_coords_wgs84,
                    delay=pulse_delay,
                    weight=weight,
                    color=color,
                    pulse_color=pulse_color,
                    dash_array=[10, 20],
                    opacity=0.95,
                    tooltip=f"{label} &bull; {r.length_m:.1f} m, Bed Slope: {r.bed_slope_pct:.2f}% (Flow: {pulse_delay}ms)",
                    popup=folium.Popup(popup_html, max_width=300)
                ).add_to(routes_group)
            else:
                folium.PolyLine(
                    locations=r.path_coords_wgs84,
                    color=color,
                    weight=weight,
                    opacity=0.95,
                    dash_array=dash,
                    tooltip=f"{label} &bull; {r.length_m:.1f} m, Max Cut: {r.max_cut_depth_m:.2f} m",
                    popup=folium.Popup(popup_html, max_width=300)
                ).add_to(routes_group)
        routes_group.add_to(fmap)

    # 16. Layer: Extracted Open Ditches (Roelens et al.)
    if extracted_ditches_geojson and len(extracted_ditches_geojson.get("features", [])) > 0:
        folium.GeoJson(
            extracted_ditches_geojson,
            name="Extracted Ditches (Roelens LiDAR)",
            show=True,
            style_function=lambda feature: {
                "color": "#06b6d4",
                "weight": 3.5,
                "opacity": 0.95,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["segment_id", "length_m", "start_elevation_m", "end_elevation_m"],
                aliases=["Ditch Segment:", "Length (m):", "Start Elev (m):", "End Elev (m):"],
                style="background-color: #14181F; color: #06b6d4; font-size: 12px; font-weight: bold; border-radius: 4px; padding: 4px;"
            )
        ).add_to(fmap)

    # 17. Layer: Reconnected Culverts (Driveways & Embankments)
    if reconnected_culverts_geojson and len(reconnected_culverts_geojson.get("features", [])) > 0:
        folium.GeoJson(
            reconnected_culverts_geojson,
            name="Reconnected Culverts (Driveways/Roads)",
            show=True,
            style_function=lambda feature: {
                "color": "#f59e0b",
                "weight": 4.5,
                "opacity": 0.95,
                "dashArray": "6, 8"
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["culvert_id", "gap_length_m", "confidence_score"],
                aliases=["Culvert ID:", "Gap Length (m):", "Connection Confidence (%):"],
                style="background-color: #14181F; color: #fbbf24; font-size: 12px; font-weight: bold; border-radius: 4px; padding: 4px;"
            )
        ).add_to(fmap)

    # 18. Field Perimeter Boundary
    folium.Rectangle(
        bounds=leaflet_bounds,
        color="#ef4444",
        weight=2.5,
        fill=False,
        name="Field Boundary Extent"
    ).add_to(fmap)

    # 14. Dynamic Elevation Colormap Legend
    try:
        cmap = plt.colormaps[colormap_name]
    except (AttributeError, KeyError):
        cmap = plt.cm.get_cmap(colormap_name)

    sample_points = np.linspace(0.0, 1.0, 6)
    hex_colors = [matplotlib_color_to_hex(cmap(p)) for p in sample_points]
    
    legend = cm.LinearColormap(
        colors=hex_colors,
        vmin=round(elevation_stats.min_elevation, 2),
        vmax=round(elevation_stats.max_elevation, 2),
        caption=f"Elevation (meters above MSL) [{elevation_stats.crs_display}]"
    )
    legend.add_to(fmap)

    # 15. Add Top-Right Layer Control Switcher
    folium.LayerControl(position="topright", collapsed=True).add_to(fmap)

    # 16. Floating Controls: Top-Right Toolbar, Bottom-Left Opacity Pill & Bottom-Right Vertical Scale Bar
    elev_min_val = float(elevation_stats.min_elevation)
    elev_max_val = float(elevation_stats.max_elevation)
    elev_mid_val = (elev_min_val + elev_max_val) / 2.0
    opacity_pct = int(layer_opacity * 100)

    floating_ui_html = f"""
    <!-- Floating Top-Right Map Toolbar -->
    <div style="
        position: absolute;
        top: 12px;
        left: 60px;
        z-index: 1000;
        display: flex;
        align-items: center;
        gap: 6px;
        background: rgba(20, 24, 31, 0.88);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid #232A35;
        border-radius: 30px;
        padding: 4px 10px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
    ">
        <button onclick="parent.postMessage({{type:'MAP_ZOOM_IN'}}, '*')" title="Zoom In (+)" style="
            background: #1C222B; color: #F0F4F8; border: 1px solid #232A35; border-radius: 50%; width: 26px; height: 26px;
            display: flex; align-items: center; justify-content: center; cursor: pointer; font-size: 13px; font-weight: bold;
        ">+</button>
        <button onclick="parent.postMessage({{type:'MAP_ZOOM_OUT'}}, '*')" title="Zoom Out (-)" style="
            background: #1C222B; color: #F0F4F8; border: 1px solid #232A35; border-radius: 50%; width: 26px; height: 26px;
            display: flex; align-items: center; justify-content: center; cursor: pointer; font-size: 13px; font-weight: bold;
        ">&minus;</button>
        <span style="color: #232A35;">|</span>
        <span style="color: #8B949E; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.4px;">
            GIS Tools:
        </span>
        <span style="
            background: rgba(46, 204, 113, 0.15); color: #2ECC71; border: 1px solid rgba(46, 204, 113, 0.35);
            border-radius: 12px; padding: 2px 8px; font-size: 11px; font-weight: 600;
        ">
            🎯 Reset Extent
        </span>
        <span style="
            background: rgba(0, 229, 255, 0.12); color: #00E5FF; border: 1px solid rgba(0, 229, 255, 0.3);
            border-radius: 12px; padding: 2px 8px; font-size: 11px; font-weight: 600;
        ">
            🛰️ Basemap
        </span>
        <span style="
            background: rgba(245, 158, 11, 0.12); color: #F59E0B; border: 1px solid rgba(245, 158, 11, 0.3);
            border-radius: 12px; padding: 2px 8px; font-size: 11px; font-weight: 600;
        ">
            📏 Measure
        </span>
        <span style="
            background: rgba(16, 185, 129, 0.15); color: #10B981; border: 1px solid rgba(16, 185, 129, 0.35);
            border-radius: 12px; padding: 2px 8px; font-size: 11px; font-weight: 600;
        ">
            🌐 2D/3D Mode
        </span>
    </div>

    <!-- Floating Bottom-Left Opacity Pill -->
    <div style="
        position: absolute;
        bottom: 22px;
        left: 14px;
        z-index: 1000;
        background: rgba(20, 24, 31, 0.88);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid #232A35;
        border-radius: 20px;
        padding: 5px 12px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.45);
        color: #F0F4F8;
        font-size: 11px;
        display: flex;
        align-items: center;
        gap: 6px;
    ">
        <span style="color: #8B949E; font-weight: 500;">Layer Opacity:</span>
        <b style="color: #00E5FF; font-family: monospace;">{opacity_pct}%</b>
    </div>

    <!-- Floating Bottom-Right Vertical Elevation Legend -->
    <div style="
        position: absolute;
        bottom: 22px;
        right: 14px;
        z-index: 1000;
        background: rgba(20, 24, 31, 0.90);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid #232A35;
        border-radius: 10px;
        padding: 10px 12px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.55);
        color: #F0F4F8;
        font-family: sans-serif;
        min-width: 135px;
    ">
        <div style="font-size: 10px; font-weight: 700; color: #8B949E; letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 6px;">
            ELEVATION (MSL)
        </div>
        <div style="display: flex; align-items: center; gap: 10px;">
            <div style="
                width: 10px; height: 80px; border-radius: 3px;
                background: linear-gradient(to bottom, #A3E635 0%, #10B981 35%, #0D9488 65%, #1E3A8A 100%);
                border: 1px solid rgba(255,255,255,0.1);
            "></div>
            <div style="display: flex; flex-direction: column; justify-content: space-between; height: 80px; font-family: monospace; font-size: 11px;">
                <span style="color: #A3E635; font-weight: bold;">▲ High ({elev_max_val:.1f}m)</span>
                <span style="color: #8B949E;">• Mid ({elev_mid_val:.1f}m)</span>
                <span style="color: #38BDF8; font-weight: bold;">▼ Low ({elev_min_val:.1f}m)</span>
            </div>
        </div>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(floating_ui_html))

    # 17. Dark Theme Styling for Leaflet & Controls
    dark_css = """
    <style>
        .leaflet-container {
            background-color: #0b0d10 !important;
            font-family: 'Inter', -apple-system, sans-serif !important;
        }
        .leaflet-control-layers {
            background: rgba(20, 24, 31, 0.92) !important;
            backdrop-filter: blur(12px) !important;
            -webkit-backdrop-filter: blur(12px) !important;
            color: #f0f4f8 !important;
            border: 1px solid #232a35 !important;
            border-radius: 8px !important;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.45) !important;
        }
        .leaflet-control-layers-expanded {
            background: rgba(20, 24, 31, 0.95) !important;
            color: #f0f4f8 !important;
        }
        .leaflet-control-layers label {
            color: #f0f4f8 !important;
            font-size: 12px !important;
            font-weight: 500 !important;
        }
        .leaflet-control-zoom a {
            background-color: #14181F !important;
            color: #00E5FF !important;
            border-color: #232A35 !important;
        }
        .leaflet-control-zoom a:hover {
            background-color: #1C222B !important;
            color: #2ECC71 !important;
        }
        .caption text, svg text {
            fill: #f0f4f8 !important;
            font-family: sans-serif !important;
            font-size: 11px !important;
        }
    </style>
    """
    fmap.get_root().html.add_child(folium.Element(dark_css))

    return fmap


def create_focused_map(
    mode: str,
    elevation: np.ndarray,
    metadata: RasterMetadata,
    elevation_stats: ElevationStats,
    terrain_summary: Optional[TerrainSummary] = None,
    hillshade: Optional[np.ndarray] = None,
    depression_depth: Optional[np.ndarray] = None,
    contour_geojson: Optional[Dict[str, Any]] = None,
    drainage_routes: Optional[List[Any]] = None,
    priority_sinks: Optional[List[Any]] = None,
    candidate_outlets: Optional[List[Any]] = None,
    extracted_ditches_geojson: Optional[Dict[str, Any]] = None,
    layer_opacity: float = 0.8
) -> folium.Map:
    """
    Renders a dedicated, focused map for side-by-side comparison.
    mode == 'terrain': Highlights bare-earth elevation relief, hillshade, and contours.
    mode == 'drainage': Highlights waterlogging risk zones, existing ditches, and recommended trench routes.
    """
    min_lon, min_lat, max_lon, max_lat = metadata.bounds_wgs84
    center_lat = (min_lat + max_lat) / 2.0
    center_lon = (min_lon + max_lon) / 2.0
    leaflet_bounds = [[min_lat, min_lon], [max_lat, max_lon]]

    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=18,
        max_zoom=22,
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr="&copy; OpenStreetMap contributors &copy; CARTO"
    )

    if mode == "terrain":
        # 1. Hillshade
        if hillshade is not None:
            hs_rgba = array_to_rgba(hillshade, colormap_name="gray", alpha=0.55)
            ImageOverlay(
                image=hs_rgba,
                bounds=leaflet_bounds,
                opacity=0.55,
                name="3D Land Shape (Hillshade)",
                overlay=True,
                control=False
            ).add_to(fmap)

        # 2. Elevation Surface
        elev_rgba = array_to_rgba(
            elevation,
            colormap_name="terrain",
            vmin=elevation_stats.min_elevation,
            vmax=elevation_stats.max_elevation,
            alpha=layer_opacity
        )
        ImageOverlay(
            image=elev_rgba,
            bounds=leaflet_bounds,
            opacity=layer_opacity,
            name="Land Elevation Map",
            overlay=True,
            control=False
        ).add_to(fmap)

        # 3. Contours
        if contour_geojson and contour_geojson.get("features"):
            folium.GeoJson(
                contour_geojson,
                name="Elevation Contours",
                style_function=lambda feature: {
                    "color": "#f1f5f9" if feature["properties"].get("is_index", False) else "#94a3b8",
                    "weight": 1.8 if feature["properties"].get("is_index", False) else 0.9,
                    "opacity": 0.8
                }
            ).add_to(fmap)

    elif mode == "drainage":
        # Subtle hillshade backdrop
        if hillshade is not None:
            hs_rgba = array_to_rgba(hillshade, colormap_name="gray", alpha=0.35)
            ImageOverlay(
                image=hs_rgba,
                bounds=leaflet_bounds,
                opacity=0.35,
                name="Hillshade Backdrop",
                overlay=True,
                control=False
            ).add_to(fmap)

        # Waterlogging Depression Heatmap
        if depression_depth is not None and np.any(~np.isnan(depression_depth)):
            dep_cm = depression_depth * 100.0
            dep_rgba = array_to_rgba(
                dep_cm,
                colormap_name="Blues",
                vmin=0.0,
                vmax=max(float(np.nanmax(dep_cm)), 10.0),
                alpha=0.85
            )
            ImageOverlay(
                image=dep_rgba,
                bounds=leaflet_bounds,
                opacity=0.85,
                name="Waterlogging Risk Zones",
                overlay=True,
                control=False
            ).add_to(fmap)

        # Extracted Ditches (Roelens et al.)
        if extracted_ditches_geojson and extracted_ditches_geojson.get("features"):
            folium.GeoJson(
                extracted_ditches_geojson,
                name="Existing Ditches (Roelens et al.)",
                style_function=lambda feature: {
                    "color": "#f97316",
                    "weight": 3.0,
                    "opacity": 0.85
                }
            ).add_to(fmap)

        # Candidate Outlets
        if candidate_outlets:
            for out in candidate_outlets:
                folium.CircleMarker(
                    location=[out.coord_wgs84[0], out.coord_wgs84[1]],
                    radius=6,
                    color="#f59e0b",
                    weight=2,
                    fill=True,
                    fill_color="#fbbf24",
                    fill_opacity=0.9,
                    tooltip=f"<b>Field Exit #{out.outlet_id}</b><br>Elevation: {out.elevation_m:.2f} m"
                ).add_to(fmap)

        # Proposed Drainage Routes with Animated AntPath
        if drainage_routes:
            for r in drainage_routes:
                route_color = "#2ecc71" if r.is_recommended else "#38bdf8"
                AntPath(
                    locations=r.path_coords_wgs84,
                    color=route_color,
                    weight=4.5 if r.is_recommended else 2.5,
                    opacity=0.9,
                    dash_array=[10, 18],
                    delay=600,
                    pulse_color="#ffffff",
                    tooltip=f"<b>Route {r.route_id}</b> ({r.length_m:.1f} m, {r.bed_slope_pct:.2f}% grade)"
                ).add_to(fmap)

    # Dark Minimal Theme Styling
    dark_css = """
    <style>
        .leaflet-container { background-color: #0b0d10 !important; }
    </style>
    """
    fmap.get_root().html.add_child(folium.Element(dark_css))
    return fmap


def create_side_by_side_swipe_map(
    elevation: np.ndarray,
    metadata: RasterMetadata,
    elevation_stats: ElevationStats,
    hillshade: Optional[np.ndarray],
    depression_depth: Optional[np.ndarray],
    drainage_routes: Optional[List[Any]] = None,
    layer_opacity: float = 0.85
) -> folium.Map:
    """
    Renders an interactive swipe split-screen map with Folium SideBySideLayers:
    Left: Raw Bare-Earth Elevation Surface & Hillshade.
    Right: Waterlogging Depressions & Proposed Drainage Alignment.
    """
    min_lon, min_lat, max_lon, max_lat = metadata.bounds_wgs84
    center_lat = (min_lat + max_lat) / 2.0
    center_lon = (min_lon + max_lon) / 2.0
    leaflet_bounds = [[min_lat, min_lon], [max_lat, max_lon]]

    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=18,
        max_zoom=22,
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr="&copy; OpenStreetMap contributors &copy; CARTO"
    )

    # Left Layer: Elevation Surface (DTM)
    left_rgba = array_to_rgba(
        elevation,
        colormap_name="terrain",
        vmin=elevation_stats.min_elevation,
        vmax=elevation_stats.max_elevation,
        alpha=layer_opacity
    )
    layer_left = ImageOverlay(
        image=left_rgba,
        bounds=leaflet_bounds,
        opacity=layer_opacity,
        name="Bare-Earth Elevation (Left)"
    ).add_to(fmap)

    # Right Layer: Waterlogging Risk Zones
    if depression_depth is not None and np.any(~np.isnan(depression_depth)):
        dep_cm = depression_depth * 100.0
        right_rgba = array_to_rgba(
            dep_cm,
            colormap_name="Blues",
            vmin=0.0,
            vmax=max(float(np.nanmax(dep_cm)), 10.0),
            alpha=0.9
        )
    else:
        right_rgba = array_to_rgba(elevation, colormap_name="viridis", alpha=0.9)

    layer_right = ImageOverlay(
        image=right_rgba,
        bounds=leaflet_bounds,
        opacity=0.9,
        name="Waterlogging Risk (Right)"
    ).add_to(fmap)

    # Add SideBySideLayers plugin
    folium.plugins.SideBySideLayers(layer_left=layer_left, layer_right=layer_right).add_to(fmap)

    # Overlay proposed routes on map
    if drainage_routes:
        for r in drainage_routes:
            if r.is_recommended:
                folium.PolyLine(
                    locations=r.path_coords_wgs84,
                    color="#2ecc71",
                    weight=4,
                    opacity=0.95,
                    tooltip=f"<b>Primary Trench {r.route_id}</b> ({r.length_m:.1f} m, {r.bed_slope_pct:.2f}% grade)"
                ).add_to(fmap)

    dark_css = """
    <style>
        .leaflet-container { background-color: #0b0d10 !important; }
        .leaflet-sbs { background: #2ecc71 !important; width: 4px !important; }
        .leaflet-sbs-divider { background: #2ecc71 !important; }
    </style>
    """
    fmap.get_root().html.add_child(folium.Element(dark_css))
    return fmap


def create_elevation_map(
    elevation: np.ndarray,
    metadata: RasterMetadata,
    stats: ElevationStats,
    colormap_name: str = "terrain",
    opacity: float = 0.75
) -> folium.Map:
    """Backward compatible wrapper for create_terrain_map."""
    return create_terrain_map(
        elevation=elevation,
        metadata=metadata,
        elevation_stats=stats,
        colormap_name=colormap_name,
        layer_opacity=opacity
    )
