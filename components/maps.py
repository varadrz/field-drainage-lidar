"""
Interactive Web Map Component
Builds responsive Folium / Leaflet maps with styled colored raster overlays,
multi-layer controls (Elevation, Slope, Aspect, Hillshade, Contours, Depressions),
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
        colormap_name: Name of Matplotlib colormap (e.g. 'terrain', 'YlOrRd', 'twilight').
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
    colormap_name: str = "terrain",
    layer_opacity: float = 0.75,
    default_layer: str = "Elevation Surface (DTM)"
) -> folium.Map:
    """
    Renders an interactive multi-layer GIS map with Elevation, Slope, Aspect,
    Hillshade, Vector Contours, and Waterlogging Depressions.
    
    Args:
        elevation: 2D array of elevation values.
        metadata: RasterMetadata object.
        elevation_stats: ElevationStats summary.
        terrain_summary: Optional TerrainSummary with slope and depression stats.
        slope_pct: Optional 2D array of slope in percent grade.
        aspect: Optional 2D array of aspect in compass degrees.
        hillshade: Optional 2D uint8 array (0-255).
        depression_depth: Optional 2D array of sink depths in meters.
        contour_geojson: Optional GeoJSON FeatureCollection of contour lines.
        colormap_name: Colormap for elevation surface.
        layer_opacity: Transparency for raster overlays.
        default_layer: Which primary raster overlay to activate initially.
        
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

    # 1. Base Dark Minimal (CartoDB) - Sleek default for dark mode
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr="&copy; OpenStreetMap contributors &copy; CARTO",
        subdomains="abcd",
        name="Dark Minimal (CartoDB)",
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

    # 3. Base OpenStreetMap
    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr="OpenStreetMap",
        name="OpenStreetMap",
        overlay=False,
        control=True
    ).add_to(fmap)

    # Determine which layer to activate initially
    is_elev_active = default_layer in ["Land Elevation Map", "Elevation Surface (DTM)"]
    is_slope_active = default_layer in ["Land Slope (% Grade)", "Slope Map (% Grade)"]
    is_hs_active = default_layer in ["3D Land Shape (Hillshade)", "Hillshade (Shaded Relief)"]
    is_aspect_active = default_layer in ["Slope Direction (Aspect)", "Aspect Map (Direction)"]

    # 3. Layer: Hillshade (Shaded Relief)
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

    # 4. Layer: Elevation Surface (DTM)
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

    # 5. Layer: Slope Map (% Grade)
    if slope_pct is not None:
        # Agricultural slope scale: 0% to 5% is standard field range; cap at 95th percentile
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

    # 6. Layer: Aspect Map (Compass Direction)
    if aspect is not None:
        aspect_display = np.copy(aspect)
        aspect_display[aspect < 0] = np.nan  # flat areas transparent
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

    # 7. Layer: Waterlogging Depressions (Sinks)
    if depression_depth is not None and np.any(depression_depth > 0.0):
        # Create an eye-catching blue/cyan overlay for depressions
        dep_rgba = np.zeros((depression_depth.shape[0], depression_depth.shape[1], 4), dtype=np.uint8)
        dep_mask = depression_depth > 0.0
        
        # Color mapping: light blue for shallow (5cm) to deep navy for deep (>30cm)
        max_d = float(np.max(depression_depth)) if np.any(dep_mask) else 0.4
        max_d = max(max_d, 0.1)
        norm_d = np.clip(depression_depth / max_d, 0.0, 1.0)
        
        # Blue gradient: R: 20->50, G: 60->190, B: 230, A: 210
        dep_rgba[dep_mask, 0] = (20 + (1.0 - norm_d[dep_mask]) * 30).astype(np.uint8)
        dep_rgba[dep_mask, 1] = (60 + (1.0 - norm_d[dep_mask]) * 130).astype(np.uint8)
        dep_rgba[dep_mask, 2] = 230
        dep_rgba[dep_mask, 3] = 210

        ImageOverlay(
            image=dep_rgba,
            bounds=leaflet_bounds,
            opacity=0.85,
            name="Waterlogging Risk Spots (Ponding)",
            show=True,
            interactive=True,
            zindex=6
        ).add_to(fmap)

    # 8. Layer: Vector Contours (GeoJSON)
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

    # 9. Field Perimeter Boundary
    folium.Rectangle(
        bounds=leaflet_bounds,
        color="#ef4444",
        weight=2.5,
        fill=False,
        name="Field Boundary Extent"
    ).add_to(fmap)

    # 10. Dynamic Elevation Colormap Legend
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

    # 11. Add Top-Right Layer Control Switcher
    folium.LayerControl(position="topright", collapsed=False).add_to(fmap)

    # 12. Dark Theme Styling for Leaflet & Branca Legend
    dark_css = """
    <style>
        .leaflet-container {
            background-color: #0b0d10 !important;
        }
        .leaflet-control-layers {
            background: #14181f !important;
            color: #f0f4f8 !important;
            border: 1px solid #2d3748 !important;
            border-radius: 6px !important;
        }
        .leaflet-control-layers-expanded {
            background: #14181f !important;
            color: #f0f4f8 !important;
        }
        .leaflet-control-layers label {
            color: #f0f4f8 !important;
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
