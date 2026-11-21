"""
Tier 3 Sample Data Generator: Synthetic Indian Agricultural Farm
Simulates a ~5-acre agricultural plot in Indian UTM CRS (EPSG:32644).
Features:
- 1 meter spatial resolution
- Gentle natural planar slope (dipping NW to SE)
- Micro-topographic waterlogging depression (sink)
- Shallow drainage ditch
- Earthen perimeter field bunds (medh/daul)
"""

import os
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin


def generate_synthetic_farm(output_path: str = "data/sample/synthetic_indian_farm.tif") -> str:
    """
    Generates a 1m resolution GeoTIFF simulating an Indian agricultural field.
    Dimensions: 150m (rows) x 140m (cols) = 21,000 m² (~5.18 acres).
    Projected in EPSG:32644 (WGS 84 / UTM Zone 44N).
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Grid dimensions (150 rows, 140 cols at 1.0m resolution)
    nrows, ncols = 150, 140
    res = 1.0  # 1 meter per pixel

    # Coordinate Reference System & Origin (UTM Zone 44N - Central/Eastern India)
    # Approx: Lon 79.5°E, Lat 18.5°N
    origin_x = 340000.0  # Easting
    origin_y = 2050150.0  # Northing
    crs_epsg = 32644

    # 1. Base Planar Topography: 285.0m base with a gentle 0.8% slope dipping NW to SE
    y_indices, x_indices = np.indices((nrows, ncols))
    base_elevation = 285.5
    slope_x = -0.005  # -0.5% slope from West to East
    slope_y = -0.006  # -0.6% slope from North to South
    dem = base_elevation + (x_indices * slope_x) + (y_indices * slope_y)

    # 2. Add Realistic Surface Roughness (tillage & soil clods)
    np.random.seed(42)
    roughness = np.random.normal(loc=0.0, scale=0.02, size=(nrows, ncols))
    dem += roughness

    # 3. Simulate a Waterlogging Depression (Sink)
    # Center around row 95, col 75, radius ~14m, max depth ~0.45m
    sink_row, sink_col = 95, 75
    sink_radius = 14.0
    max_sink_depth = 0.45
    dist_to_sink = np.sqrt((y_indices - sink_row)**2 + (x_indices - sink_col)**2)
    sink_mask = dist_to_sink <= sink_radius
    # Parabolic bowl depression profile
    sink_depth = max_sink_depth * (1.0 - (dist_to_sink[sink_mask] / sink_radius)**2)
    dem[sink_mask] -= sink_depth

    # 4. Simulate a Shallow Field Drainage Ditch
    # Runs from the depression towards the eastern field boundary outlet (row 110, col 139)
    ditch_start_r, ditch_start_c = 95, 75
    ditch_end_r, ditch_end_c = 110, 138
    ditch_width = 3.0  # meters
    ditch_depth = 0.35  # meters

    # Line segment distance calculation
    p1 = np.array([ditch_start_r, ditch_start_c])
    p2 = np.array([ditch_end_r, ditch_end_c])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    line_unit = line_vec / line_len

    coords = np.stack([y_indices, x_indices], axis=-1)
    vec_to_points = coords - p1
    proj_len = np.sum(vec_to_points * line_unit, axis=-1)
    proj_clamped = np.clip(proj_len, 0, line_len)
    closest_pts = p1 + np.outer(proj_clamped, line_unit).reshape((nrows, ncols, 2))
    dist_to_ditch = np.linalg.norm(coords - closest_pts, axis=-1)

    ditch_mask = (dist_to_ditch <= (ditch_width / 2.0)) & (proj_len >= 0) & (proj_len <= line_len)
    # U-shaped ditch incision
    ditch_incision = ditch_depth * (1.0 - (dist_to_ditch[ditch_mask] / (ditch_width / 2.0))**2)
    dem[ditch_mask] -= ditch_incision

    # 5. Simulate Perimeter Field Bunds (Medh / Daul)
    # 2m wide, 0.35m elevated ridge around field perimeter
    bund_width = 2
    bund_height = 0.35
    bund_mask = np.zeros((nrows, ncols), dtype=bool)
    bund_mask[:bund_width, :] = True
    bund_mask[-bund_width:, :] = True
    bund_mask[:, :bund_width] = True
    bund_mask[:, -bund_width:] = True

    # Keep a 4m natural gap in the eastern bund for the drainage outlet
    outlet_mask = (y_indices >= (ditch_end_r - 2)) & (y_indices <= (ditch_end_r + 2)) & (x_indices >= (ncols - bund_width - 1))
    bund_mask[outlet_mask] = False
    dem[bund_mask] += bund_height

    # Convert to float32
    dem = dem.astype(np.float32)

    # Affine GeoTransform: (origin_x, pixel_width, 0, origin_y, 0, -pixel_height)
    transform = from_origin(origin_x, origin_y, res, res)

    # Write out GeoTIFF
    with rasterio.open(
        out_file,
        "w",
        driver="GTiff",
        height=nrows,
        width=ncols,
        count=1,
        dtype=rasterio.float32,
        crs=f"EPSG:{crs_epsg}",
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(dem, 1)

    print(f"[SUCCESS] Synthetic Indian Farm GeoTIFF created at: {out_file.resolve()}")
    print(f"  - Extents: {ncols}x{nrows} pixels ({ncols*res:.1f}m x {nrows*res:.1f}m, ~{(ncols*nrows)/(4046.86):.2f} acres)")
    print(f"  - CRS: EPSG:{crs_epsg} (UTM Zone 44N - Metric)")
    print(f"  - Elevation Range: {np.nanmin(dem):.2f}m to {np.nanmax(dem):.2f}m")
    return str(out_file.resolve())


if __name__ == "__main__":
    generate_synthetic_farm()
