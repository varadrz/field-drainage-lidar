"""
Interactive Web Map Component
Builds responsive Folium / Leaflet maps with styled colored raster overlays,
WGS84 coordinate handling, and dynamic colorbar legends.
"""

from typing import Tuple, Optional
import numpy as np
import folium
from folium.raster_layers import ImageOverlay
import branca.colormap as cm
import matplotlib.pyplot as plt
from modules.data_loader import RasterMetadata
from modules.elevation_analysis import ElevationStats


def create_elevation_map(
    elevation: np.ndarray,
    metadata: RasterMetadata,
    stats: ElevationStats,
    colormap_name: str = "terrain",
    opacity: float = 0.75
) -> folium.Map:
    """
    Renders an elevation surface as an interactive Folium map overlay.
    
    Args:
        elevation: 2D numpy array with NaNs for NoData
        metadata: RasterMetadata object containing WGS84 bounding box
        stats: ElevationStats containing min/max values for legend scaling
        colormap_name: Matplotlib colormap identifier (e.g. 'terrain', 'viridis', 'plasma')
        opacity: Transparency value between 0.0 and 1.0
        
    Returns:
        folium.Map instance ready for st_folium display
    """
    # 1. Calculate map center and bounds in [lat, lon] format for Leaflet
    min_lon, min_lat, max_lon, max_lat = metadata.bounds_wgs84
    center_lat = (min_lat + max_lat) / 2.0
    center_lon = (min_lon + max_lon) / 2.0
    leaflet_bounds = [[min_lat, min_lon], [max_lat, max_lon]]

    # 2. Normalize elevation array to [0.0, 1.0] for colormap mapping
    valid_mask = ~np.isnan(elevation)
    min_val = stats.min_elevation
    max_val = stats.max_elevation
    val_range = max_val - min_val if (max_val - min_val) > 1e-4 else 1.0

    normalized = np.zeros_like(elevation, dtype=np.float32)
    normalized[valid_mask] = (elevation[valid_mask] - min_val) / val_range
    normalized = np.clip(normalized, 0.0, 1.0)

    # 3. Apply Matplotlib colormap to produce RGBA image array (0-255 uint8)
    try:
        cmap = plt.colormaps[colormap_name]
    except (AttributeError, KeyError):
        cmap = plt.cm.get_cmap(colormap_name)

    rgba_img = (cmap(normalized) * 255).astype(np.uint8)
    # Set NoData pixels to completely transparent
    rgba_img[~valid_mask] = [0, 0, 0, 0]

    # 4. Initialize Folium Map with clean Esri Satellite & OpenStreetMap basemaps
    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=18,
        max_zoom=22,
        tiles=None
    )

    # Add Esri World Imagery (Satellite)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite Imagery (Esri)",
        overlay=False,
        control=True
    ).add_to(fmap)

    # Add OpenStreetMap as alternative basemap
    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr="OpenStreetMap",
        name="OpenStreetMap",
        overlay=False,
        control=True
    ).add_to(fmap)

    # 5. Add Elevation Raster as ImageOverlay
    ImageOverlay(
        image=rgba_img,
        bounds=leaflet_bounds,
        opacity=opacity,
        name="Elevation Surface (DTM)",
        interactive=True,
        cross_origin=False,
        zindex=1
    ).add_to(fmap)

    # 6. Add Bounding Box Outline to clearly demarcate field borders
    folium.Rectangle(
        bounds=leaflet_bounds,
        color="#e74c3c",
        weight=2,
        fill=False,
        name="Field Boundary Extent"
    ).add_to(fmap)

    # 7. Add Interactive Dynamic Colormap Legend
    # Generate sampled hex colors from the matplotlib colormap
    sample_points = np.linspace(0.0, 1.0, 6)
    hex_colors = [matplotlib_color_to_hex(cmap(p)) for p in sample_points]
    
    legend = cm.LinearColormap(
        colors=hex_colors,
        vmin=round(min_val, 2),
        vmax=round(max_val, 2),
        caption=f"Elevation (meters above MSL) [{stats.crs_display}]"
    )
    legend.add_to(fmap)

    # 8. Add Layer Control to easily toggle satellite vs elevation
    folium.LayerControl(position="topright", collapsed=False).add_to(fmap)

    return fmap


def matplotlib_color_to_hex(rgba_tuple: Tuple[float, float, float, float]) -> str:
    """Converts a matplotlib RGBA float tuple (0-1) to an HTML hex color string."""
    r, g, b = int(rgba_tuple[0] * 255), int(rgba_tuple[1] * 255), int(rgba_tuple[2] * 255)
    return f"#{r:02x}{g:02x}{b:02x}"
