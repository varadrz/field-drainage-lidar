"""
Terrain Analysis Engine
Decoupled GIS computational module for agricultural topography.
Calculates slope (degrees & percent grade), aspect, hillshade, vector contours,
and morphological depression (sink) detection.
Zero UI / Streamlit code; strictly preserved metric coordinates.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List
import heapq
import numpy as np
import scipy.ndimage as ndi
import pyproj
import contourpy

from .data_loader import RasterMetadata


@dataclass
class TerrainSummary:
    """Statistical summary of terrain topography, slope classification, and depressions."""
    mean_slope_pct: float
    max_slope_pct: float
    median_slope_pct: float
    mean_slope_deg: float
    max_slope_deg: float
    flat_area_sq_m: float
    flat_area_acres: float
    flat_area_pct: float
    depression_count: int
    depression_sq_m: float
    depression_acres: float
    depression_ha: float
    depression_pct: float
    max_depression_depth_m: float
    max_depression_depth_cm: float
    est_depression_volume_m3: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert terrain summary to serializable dictionary."""
        return {
            "Mean Slope (%)": f"{self.mean_slope_pct:.2f}%",
            "Max Slope (%)": f"{self.max_slope_pct:.2f}%",
            "Median Slope (%)": f"{self.median_slope_pct:.2f}%",
            "Mean Slope (deg)": f"{self.mean_slope_deg:.2f}°",
            "Max Slope (deg)": f"{self.max_slope_deg:.2f}°",
            "Flat / Poorly Drained Area (< 0.5%)": f"{self.flat_area_acres:.2f} ac ({self.flat_area_pct:.1f}%)",
            "Total Depression Area": f"{self.depression_acres:.2f} ac ({self.depression_sq_m:,.0f} m²)",
            "Depression Coverage": f"{self.depression_pct:.1f}%",
            "Identified Sink Clusters": self.depression_count,
            "Max Depression Depth": f"{self.max_depression_depth_cm:.1f} cm",
            "Est. Stored Water Volume": f"{self.est_depression_volume_m3:,.1f} m³",
        }


def compute_derivatives(
    elevation: np.ndarray,
    res_x: float,
    res_y: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes horizontal (dz/dx) and vertical (dz/dy) spatial derivatives using Horn's 3x3 method.
    
    Args:
        elevation: 2D array with NaNs for NoData.
        res_x: Spatial resolution in Easting (meters).
        res_y: Spatial resolution in Northing (meters).
        
    Returns:
        dz_dx: Rate of change along X (East-West)
        dz_dy: Rate of change along Y (North-South, positive = higher towards North)
    """
    valid_mask = ~np.isnan(elevation)
    
    # Fill NaNs temporarily with local mean for smooth border convolution
    working = np.copy(elevation)
    if np.any(~valid_mask):
        mean_val = float(np.nanmean(elevation))
        working[~valid_mask] = mean_val

    # Horn 3x3 Kernels
    # kx: East (+) vs West (-)
    kx = np.array([
        [-1.0, 0.0, 1.0],
        [-2.0, 0.0, 2.0],
        [-1.0, 0.0, 1.0]
    ], dtype=np.float32) / (8.0 * res_x)

    # ky: North (+) vs South (-)
    # Since row 0 is top (North) and higher row is South:
    # ky checks top rows vs bottom rows
    ky = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0]
    ], dtype=np.float32) / (8.0 * res_y)

    dz_dx = ndi.correlate(working, kx, mode="nearest")
    dz_dy = ndi.correlate(working, ky, mode="nearest")

    # Restore NaNs
    dz_dx[~valid_mask] = np.nan
    dz_dy[~valid_mask] = np.nan

    return dz_dx, dz_dy


def compute_slope(
    elevation: np.ndarray,
    res_x: float,
    res_y: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes terrain slope in both degrees and agricultural percent grade (%).
    
    Args:
        elevation: 2D numpy array with NaNs for NoData.
        res_x: Pixel resolution in X (meters).
        res_y: Pixel resolution in Y (meters).
        
    Returns:
        Tuple of (slope_degrees, slope_percent)
    """
    dz_dx, dz_dy = compute_derivatives(elevation, res_x, res_y)
    rise_run = np.sqrt(dz_dx**2 + dz_dy**2)
    
    slope_percent = rise_run * 100.0
    slope_degrees = np.degrees(np.arctan(rise_run))

    return slope_degrees, slope_percent


def compute_aspect(
    elevation: np.ndarray,
    res_x: float,
    res_y: float
) -> np.ndarray:
    """
    Computes terrain aspect (compass face direction in degrees 0°–360°).
    0° = North, 90° = East, 180° = South, 270° = West.
    Flat areas (slope < 0.001%) are assigned -1.0.
    
    Args:
        elevation: 2D numpy array.
        res_x: Pixel resolution in X (meters).
        res_y: Pixel resolution in Y (meters).
        
    Returns:
        2D numpy array of aspect angles.
    """
    dz_dx, dz_dy = compute_derivatives(elevation, res_x, res_y)
    
    # Downhill descent vector
    vx = -dz_dx
    vy = -dz_dy
    
    # Compass angle clockwise from North (+y)
    aspect = (np.degrees(np.arctan2(vx, vy)) + 360.0) % 360.0
    
    # Flag nearly flat areas
    gradient_mag = np.sqrt(dz_dx**2 + dz_dy**2)
    flat_mask = gradient_mag < 1e-5
    aspect[flat_mask] = -1.0
    
    # Restore NaNs
    aspect[np.isnan(elevation)] = np.nan
    
    return aspect


def compute_hillshade(
    elevation: np.ndarray,
    res_x: float,
    res_y: float,
    azimuth_deg: float = 315.0,
    altitude_deg: float = 45.0
) -> np.ndarray:
    """
    Generates a 3D-like shaded relief raster (0–255) using standard cartographic illumination.
    
    Args:
        elevation: 2D numpy array.
        res_x: Resolution in X (meters).
        res_y: Resolution in Y (meters).
        azimuth_deg: Illumination direction (default 315° from NW).
        altitude_deg: Illumination angle above horizon (default 45°).
        
    Returns:
        2D uint8 numpy array with values from 0 to 255.
    """
    dz_dx, dz_dy = compute_derivatives(elevation, res_x, res_y)
    
    # Slope and aspect in radians
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    aspect_rad = np.arctan2(-dz_dx, -dz_dy)
    # Wrap aspect radians to [0, 2*pi]
    aspect_rad = (aspect_rad + 2.0 * np.pi) % (2.0 * np.pi)
    
    # Illumination angles in radians
    zenith_rad = np.radians(90.0 - altitude_deg)
    azimuth_rad = np.radians(azimuth_deg)
    
    # Lambertian reflectance equation
    shaded = (
        np.cos(zenith_rad) * np.cos(slope_rad) +
        np.sin(zenith_rad) * np.sin(slope_rad) * np.cos(azimuth_rad - aspect_rad)
    )
    
    # Clamp to [0, 1] and scale to 0-255 uint8
    shaded = np.clip(shaded, 0.0, 1.0)
    hillshade = (shaded * 255.0).astype(np.uint8)
    
    return hillshade


def detect_depressions(
    elevation: np.ndarray,
    metadata: RasterMetadata,
    min_depth_m: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Identifies localized topographic depressions (sinks/waterlogging hotspots) using
    the Priority-Flood morphological sink filling algorithm (Barnes et al. 2014 / Wang & Liu 2006).
    
    Compares the raw DEM against a depression-free filled surface and isolates genuine
    ponding areas exceeding min_depth_m (default 0.05 m / 5 cm).
    
    Args:
        elevation: 2D numpy array with NaNs for NoData.
        metadata: Associated RasterMetadata object.
        min_depth_m: Minimum depth threshold to classify as genuine sink (filters tillage noise).
        
    Returns:
        Tuple of:
            - depression_mask: 2D boolean array (True where depression exists)
            - depression_depth: 2D float32 array of sink depths in meters (0 outside sinks)
            - stats: Dictionary of depression summary metrics
    """
    valid_mask = ~np.isnan(elevation)
    total_valid_pixels = int(np.sum(valid_mask))
    
    if total_valid_pixels == 0:
        return np.zeros_like(elevation, dtype=bool), np.zeros_like(elevation, dtype=np.float32), {}

    nrows, ncols = elevation.shape
    dem = np.copy(elevation)
    
    # Replace NaNs with a very high elevation so they act as impenetrable barriers
    max_elev = float(np.nanmax(dem))
    dem[~valid_mask] = max_elev + 1000.0

    # 1. Initialize Priority-Flood Sink Filling
    filled = np.copy(dem)
    visited = np.zeros((nrows, ncols), dtype=bool)
    heap: List[Tuple[float, int, int]] = []

    # Push all perimeter boundary pixels into min-heap
    for r in range(nrows):
        for c in (0, ncols - 1):
            if valid_mask[r, c]:
                heapq.heappush(heap, (float(dem[r, c]), r, c))
                visited[r, c] = True
            else:
                visited[r, c] = True

    for c in range(ncols):
        for r in (0, nrows - 1):
            if not visited[r, c]:
                if valid_mask[r, c]:
                    heapq.heappush(heap, (float(dem[r, c]), r, c))
                visited[r, c] = True

    # 8-neighbor connectivity offsets
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    # 2. Spillway propagation
    while heap:
        z, r, c = heapq.heappop(heap)
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if 0 <= nr < nrows and 0 <= nc < ncols and not visited[nr, nc]:
                visited[nr, nc] = True
                if valid_mask[nr, nc]:
                    if dem[nr, nc] < z:
                        filled[nr, nc] = z
                        heapq.heappush(heap, (z, nr, nc))
                    else:
                        heapq.heappush(heap, (float(dem[nr, nc]), nr, nc))

    # 3. Compute Depression Depth Grid
    diff = filled - dem
    diff[~valid_mask] = 0.0
    
    # 4. Filter by Minimum Depth Threshold (filter minor noise / tillage clods)
    depression_mask = (diff >= min_depth_m) & valid_mask
    depression_depth = np.where(depression_mask, diff, 0.0).astype(np.float32)

    # 5. Connected Component Analysis (Cluster Count)
    labeled_sinks, num_features = ndi.label(depression_mask)
    
    # Area and volume calculations
    pixel_area_m2 = metadata.res_x * metadata.res_y
    depression_pixel_count = int(np.sum(depression_mask))
    total_area_sq_m = float(depression_pixel_count * pixel_area_m2)
    total_area_ha = total_area_sq_m / 10000.0
    total_area_acres = total_area_sq_m / 4046.8564224
    
    field_total_area_m2 = total_valid_pixels * pixel_area_m2
    depression_pct = (total_area_sq_m / field_total_area_m2 * 100.0) if field_total_area_m2 > 0 else 0.0

    max_depth_m = float(np.max(depression_depth)) if depression_pixel_count > 0 else 0.0
    max_depth_cm = max_depth_m * 100.0
    
    # Estimated stored water volume = sum(depth * pixel_area)
    est_water_vol_m3 = float(np.sum(depression_depth[depression_mask]) * pixel_area_m2) if depression_pixel_count > 0 else 0.0

    stats = {
        "depression_count": int(num_features),
        "total_sq_m": total_area_sq_m,
        "total_ha": total_area_ha,
        "total_acres": total_area_acres,
        "depression_pct": depression_pct,
        "max_depth_m": max_depth_m,
        "max_depth_cm": max_depth_cm,
        "est_volume_m3": est_water_vol_m3,
        "pixel_count": depression_pixel_count
    }

    return depression_mask, depression_depth, stats


def extract_contours_geojson(
    elevation: np.ndarray,
    metadata: RasterMetadata,
    interval: float = 0.5,
    min_length_pts: int = 4
) -> Dict[str, Any]:
    """
    Extracts vector contour lines as a GeoJSON FeatureCollection with user-defined intervals.
    Coordinates are transformed into WGS84 (EPSG:4326) for interactive Leaflet display.
    
    Args:
        elevation: 2D numpy array.
        metadata: Associated RasterMetadata object.
        interval: Contour interval in meters (e.g. 0.1, 0.25, 0.5, 1.0).
        min_length_pts: Minimum number of vertices to keep a contour line.
        
    Returns:
        GeoJSON FeatureCollection dict.
    """
    valid_mask = ~np.isnan(elevation)
    if not np.any(valid_mask):
        return {"type": "FeatureCollection", "features": []}

    min_elev = float(np.nanmin(elevation))
    max_elev = float(np.nanmax(elevation))

    # Calculate discrete elevation levels
    first_level = np.ceil(min_elev / interval) * interval
    if first_level < min_elev:
        first_level += interval
    levels = np.arange(first_level, max_elev, interval)

    if len(levels) == 0:
        return {"type": "FeatureCollection", "features": []}

    # Prepare coordinate grids in native coordinates
    nrows, ncols = elevation.shape
    cols = np.arange(ncols)
    rows = np.arange(nrows)
    
    # Affine transform: x = c + (col + 0.5)*a, y = f + (row + 0.5)*e
    t = metadata.transform
    x_native = t[2] + (cols + 0.5) * t[0]
    y_native = t[5] + (rows + 0.5) * t[4]

    # Coordinate transformer to WGS84
    if metadata.crs_wkt and metadata.crs_wkt != "Unknown CRS" and not metadata.is_geographic:
        try:
            transformer = pyproj.Transformer.from_crs(metadata.crs_wkt, "EPSG:4326", always_xy=True)
        except Exception:
            transformer = None
    else:
        transformer = None

    # Replace NaNs for contourpy
    working_dem = np.copy(elevation)
    working_dem[~valid_mask] = min_elev - 10.0

    c_gen = contourpy.contour_generator(x_native, y_native, working_dem)
    features = []

    for level in levels:
        lines = c_gen.lines(level)
        for line in lines:
            if len(line) < min_length_pts:
                continue

            if transformer:
                lons, lats = transformer.transform(line[:, 0], line[:, 1])
                coords = np.column_stack([lons, lats]).tolist()
            else:
                coords = line.tolist()

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords
                },
                "properties": {
                    "elevation": round(float(level), 2),
                    "elevation_str": f"{level:.2f} m",
                    "interval": interval
                }
            }
            features.append(feature)

    return {
        "type": "FeatureCollection",
        "features": features
    }


def compute_terrain_summary(
    elevation: np.ndarray,
    metadata: RasterMetadata,
    min_depth_m: float = 0.05
) -> Tuple[TerrainSummary, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Executes full terrain analysis suite and compiles agronomic metrics.
    
    Returns:
        Tuple of:
            - summary: TerrainSummary dataclass
            - slope_deg: 2D array of slope in degrees
            - slope_pct: 2D array of slope in percent grade
            - aspect: 2D array of aspect in compass degrees
            - hillshade: 2D uint8 array (0-255)
            - depression_depth: 2D float32 array of depression depths in meters
    """
    res_x = metadata.res_x
    res_y = metadata.res_y

    slope_deg, slope_pct = compute_slope(elevation, res_x, res_y)
    aspect = compute_aspect(elevation, res_x, res_y)
    hillshade = compute_hillshade(elevation, res_x, res_y)
    
    dep_mask, dep_depth, dep_stats = detect_depressions(elevation, metadata, min_depth_m=min_depth_m)

    valid_mask = ~np.isnan(elevation)
    valid_slope_pct = slope_pct[valid_mask]
    valid_slope_deg = slope_deg[valid_mask]

    pixel_area_m2 = res_x * res_y
    total_valid_pixels = int(np.sum(valid_mask))
    total_area_m2 = total_valid_pixels * pixel_area_m2

    # Poorly drained / flat land classification (< 0.5% slope)
    flat_mask = (valid_slope_pct < 0.5)
    flat_count = int(np.sum(flat_mask))
    flat_area_sq_m = float(flat_count * pixel_area_m2)
    flat_area_acres = flat_area_sq_m / 4046.8564224
    flat_area_pct = (flat_area_sq_m / total_area_m2 * 100.0) if total_area_m2 > 0 else 0.0

    summary = TerrainSummary(
        mean_slope_pct=float(np.mean(valid_slope_pct)),
        max_slope_pct=float(np.max(valid_slope_pct)),
        median_slope_pct=float(np.median(valid_slope_pct)),
        mean_slope_deg=float(np.mean(valid_slope_deg)),
        max_slope_deg=float(np.max(valid_slope_deg)),
        flat_area_sq_m=flat_area_sq_m,
        flat_area_acres=flat_area_acres,
        flat_area_pct=flat_area_pct,
        depression_count=dep_stats.get("depression_count", 0),
        depression_sq_m=dep_stats.get("total_sq_m", 0.0),
        depression_acres=dep_stats.get("total_acres", 0.0),
        depression_ha=dep_stats.get("total_ha", 0.0),
        depression_pct=dep_stats.get("depression_pct", 0.0),
        max_depression_depth_m=dep_stats.get("max_depth_m", 0.0),
        max_depression_depth_cm=dep_stats.get("max_depth_cm", 0.0),
        est_depression_volume_m3=dep_stats.get("est_volume_m3", 0.0),
    )

    return summary, slope_deg, slope_pct, aspect, hillshade, dep_depth
