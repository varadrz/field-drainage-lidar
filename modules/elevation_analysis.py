"""
Elevation Analysis and Statistical Profiling Module
Computes terrain metrics, spatial extents, and elevation statistics.
Completely decoupled from presentation logic.
"""

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
from .data_loader import RasterMetadata


@dataclass
class ElevationStats:
    """Summary statistics for an elevation terrain surface."""
    min_elevation: float
    max_elevation: float
    mean_elevation: float
    median_elevation: float
    std_elevation: float
    elevation_range: float
    valid_pixels: int
    nodata_pixels: int
    total_pixels: int
    valid_fraction: float
    area_sq_m: float
    area_ha: float
    area_acres: float
    spatial_resolution_m: float
    crs_display: str
    is_projected_metric: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to a serializable dictionary."""
        return {
            "Min Elevation (m)": f"{self.min_elevation:.2f}",
            "Max Elevation (m)": f"{self.max_elevation:.2f}",
            "Mean Elevation (m)": f"{self.mean_elevation:.2f}",
            "Median Elevation (m)": f"{self.median_elevation:.2f}",
            "Std Dev (m)": f"{self.std_elevation:.2f}",
            "Elevation Relief Range (m)": f"{self.elevation_range:.2f}",
            "Field Area (Acres)": f"{self.area_acres:.2f}",
            "Field Area (Hectares)": f"{self.area_ha:.2f}",
            "Field Area (m²)": f"{self.area_sq_m:,.0f}",
            "Spatial Resolution": f"{self.spatial_resolution_m:.2f} m/pixel",
            "Valid Coverage": f"{self.valid_fraction * 100:.1f}%",
            "CRS": self.crs_display,
        }


def compute_elevation_stats(
    elevation: np.ndarray,
    metadata: RasterMetadata
) -> ElevationStats:
    """
    Computes elevation metrics and area calculations from a validated elevation array.
    
    Args:
        elevation: 2D NumPy array with NoData represented as np.nan
        metadata: Associated RasterMetadata object
        
    Returns:
        ElevationStats dataclass instance
        
    Raises:
        ValueError: If all pixels are NoData or NaN.
    """
    valid_mask = ~np.isnan(elevation)
    valid_count = int(np.sum(valid_mask))
    total_count = int(elevation.size)
    nodata_count = total_count - valid_count

    if valid_count == 0:
        raise ValueError("Cannot compute elevation statistics: all raster cells are NoData/NaN.")

    valid_vals = elevation[valid_mask]

    min_elev = float(np.min(valid_vals))
    max_elev = float(np.max(valid_vals))
    mean_elev = float(np.mean(valid_vals))
    median_elev = float(np.median(valid_vals))
    std_elev = float(np.std(valid_vals))
    elev_range = max_elev - min_elev

    # Estimate field area based on pixel resolution
    # If projected metric, res_x * res_y is in square meters
    if metadata.is_projected:
        pixel_area = metadata.res_x * metadata.res_y
        area_sq_m = float(valid_count * pixel_area)
        area_ha = area_sq_m / 10000.0
        area_acres = area_sq_m / 4046.8564224
        resolution_m = float((metadata.res_x + metadata.res_y) / 2.0)
    else:
        # Approximate degrees to meters (at equatorial / tropical Indian latitudes ~18°N)
        meters_per_deg = 111320.0
        pixel_area = (metadata.res_x * meters_per_deg) * (metadata.res_y * meters_per_deg * np.cos(np.radians(18.0)))
        area_sq_m = float(valid_count * pixel_area)
        area_ha = area_sq_m / 10000.0
        area_acres = area_sq_m / 4046.8564224
        resolution_m = float(((metadata.res_x + metadata.res_y) / 2.0) * meters_per_deg)

    crs_display = f"EPSG:{metadata.crs_epsg}" if metadata.crs_epsg else metadata.units

    return ElevationStats(
        min_elevation=min_elev,
        max_elevation=max_elev,
        mean_elevation=mean_elev,
        median_elevation=median_elev,
        std_elevation=std_elev,
        elevation_range=elev_range,
        valid_pixels=valid_count,
        nodata_pixels=nodata_count,
        total_pixels=total_count,
        valid_fraction=float(valid_count / total_count),
        area_sq_m=area_sq_m,
        area_ha=area_ha,
        area_acres=area_acres,
        spatial_resolution_m=resolution_m,
        crs_display=crs_display,
        is_projected_metric=metadata.is_projected,
    )
