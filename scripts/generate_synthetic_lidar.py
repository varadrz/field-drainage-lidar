"""
Tier 3 Sample LiDAR Generator: Synthetic Indian Agricultural Farm
Generates a realistic airborne LiDAR point cloud (.las) in Indian UTM CRS (EPSG:32644).
Features:
- ~5-acre agricultural plot (150m x 140m)
- ~100,000 points (~4.8 pts/m²)
- ASPRS Standard Classification:
  - Class 2: Bare-Earth Ground returns (~88% of points)
  - Class 5: High Vegetation / Trees / Bund Shrubs (~12% of points)
- Waterlogging depression centered around row 95, col 75
- Shallow agricultural drainage ditch with an intentional 6-meter
  culvert / farm driveway crossing (road plug) to test Roelens et al. connectivity bridging
- Perimeter field bunds (medh / daul) with trees
"""

import os
from pathlib import Path
import numpy as np
import pyproj
import laspy


def generate_synthetic_lidar(output_path: str = "data/sample/synthetic_indian_farm.las") -> str:
    """
    Generates a realistic synthetic LAS point cloud for agricultural drainage testing.
    CRS: EPSG:32644 (WGS 84 / UTM Zone 44N).
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)

    # Field Dimensions
    width_m = 140.0   # X: 0 to 140 m
    height_m = 150.0  # Y: 0 to 150 m
    origin_x = 340000.0  # Easting (UTM 44N)
    origin_y = 2050150.0  # Northing (top-left)

    # Total points (~100,000 points for ~4.8 pts/m²)
    num_points = 100000

    # 1. Generate random 2D positions within the field
    # Notice: x increases Eastward, y in UTM increases Northward
    # Let local_x in [0, 140], local_y in [0, 150] where 150 is North, 0 is South
    # To match synthetic_indian_farm.tif where row 0 is North (2050150) and row 150 is South (2050000):
    # local_y = (150 - row)
    local_x = np.random.uniform(0.0, width_m, num_points)
    local_y = np.random.uniform(0.0, height_m, num_points)

    # 2. Base Planar Topography (same dip NW to SE as GeoTIFF)
    # NW corner is local_x=0, local_y=150; SE corner is local_x=140, local_y=0
    # base elevation = 285.5m at NW corner
    # row = 150 - local_y, col = local_x
    row_approx = height_m - local_y
    col_approx = local_x

    base_elev = 285.5
    slope_x = -0.005  # -0.5% slope from West to East
    slope_y = -0.006  # -0.6% slope from North to South
    z = base_elev + (col_approx * slope_x) + (row_approx * slope_y)

    # Add realistic micro-topography & plowing furrow roughness
    roughness = np.random.normal(loc=0.0, scale=0.018, size=num_points)
    z += roughness

    # 3. Simulate Waterlogging Depression (Sink)
    # Centered around row 95, col 75 -> local_x = 75, local_y = 150 - 95 = 55
    sink_x, sink_y = 75.0, 55.0
    sink_radius = 14.0
    max_sink_depth = 0.45
    dist_to_sink = np.sqrt((local_x - sink_x)**2 + (local_y - sink_y)**2)
    in_sink = dist_to_sink <= sink_radius
    z[in_sink] -= max_sink_depth * (1.0 - (dist_to_sink[in_sink] / sink_radius)**2)

    # 4. Simulate Shallow Agricultural Drainage Ditch
    # Runs from sink (col 75, row 95) -> (col 138, row 110)
    # In local coords: P1 = (75, 55), P2 = (138, 40)
    p1 = np.array([75.0, 55.0])
    p2 = np.array([138.0, 40.0])
    ditch_vec = p2 - p1
    ditch_len = np.linalg.norm(ditch_vec)
    ditch_unit = ditch_vec / ditch_len

    coords_2d = np.stack([local_x, local_y], axis=-1)
    vec_to_pts = coords_2d - p1
    proj_ditch = np.sum(vec_to_pts * ditch_unit, axis=-1)
    proj_clamped = np.clip(proj_ditch, 0, ditch_len)
    closest_ditch_pts = p1 + np.outer(proj_clamped, ditch_unit)
    dist_to_ditch = np.linalg.norm(coords_2d - closest_ditch_pts, axis=-1)

    ditch_width = 3.0
    ditch_depth = 0.35
    in_ditch_zone = (dist_to_ditch <= (ditch_width / 2.0)) & (proj_ditch >= 0) & (proj_ditch <= ditch_len)

    # --- INTENTIONAL CULVERT / DRIVEWAY CROSSING ---
    # At distance 28m to 35m along the ditch, a raised farm road / embankment crosses the ditch
    # In this 7-meter gap, the ditch is NOT excavated (it is covered / bridged by the driveway)
    in_culvert_gap = in_ditch_zone & (proj_ditch >= 28.0) & (proj_ditch <= 35.0)

    # Apply ditch incision only where it is an open trench (NOT in the culvert driveway gap)
    open_ditch = in_ditch_zone & (~in_culvert_gap)
    ditch_incision = ditch_depth * (1.0 - (dist_to_ditch[open_ditch] / (ditch_width / 2.0))**2)
    z[open_ditch] -= ditch_incision

    # Driveway surface is slightly crowned (+0.10m gravel layer)
    z[in_culvert_gap] += 0.10

    # 5. Field Perimeter Bunds (Medh / Daul)
    bund_width = 2.0
    bund_height = 0.35
    in_bund = (
        (local_x <= bund_width) |
        (local_x >= (width_m - bund_width)) |
        (local_y <= bund_width) |
        (local_y >= (height_m - bund_width))
    )

    # Leave 4m opening in the eastern bund near ditch outlet (local_y between 38 and 42)
    outlet_gap = (local_x >= (width_m - bund_width - 1.0)) & (local_y >= 38.0) & (local_y <= 42.0)
    in_bund = in_bund & (~outlet_gap)
    z[in_bund] += bund_height

    # 6. Classification & Simulated Tree / Vegetation Canopy
    # ASPRS Class 2 = Ground, Class 5 = High Vegetation
    classification = np.full(num_points, 2, dtype=np.uint8)

    # Simulate trees and shrubs along perimeter bunds (especially Northern and Western bunds)
    veg_prob = np.zeros(num_points, dtype=float)
    veg_prob[in_bund] = 0.40  # 40% of points on bunds are vegetation returns
    # Cluster trees in NW corner
    nw_cluster = (local_x <= 20.0) & (local_y >= 130.0)
    veg_prob[nw_cluster] = 0.70

    is_veg = np.random.uniform(0.0, 1.0, num_points) < veg_prob
    classification[is_veg] = 5

    # Vegetation height: 1.2m to 5.0m above ground
    veg_height = np.random.uniform(1.2, 5.0, np.sum(is_veg))
    z[is_veg] += veg_height

    # 7. Convert local coordinates to UTM Zone 44N native coordinates
    utm_x = origin_x + local_x
    utm_y = (origin_y - height_m) + local_y

    # 8. Create and Write LAS File using laspy
    header = laspy.LasHeader(point_format=2, version="1.2")
    header.offsets = [origin_x, origin_y - height_m, 280.0]
    header.scales = [0.001, 0.001, 0.001]
    header.add_crs(pyproj.CRS.from_epsg(32644))

    las_data = laspy.LasData(header)
    las_data.x = utm_x
    las_data.y = utm_y
    las_data.z = z
    las_data.classification = classification
    las_data.intensity = np.where(classification == 2, 80, 210).astype(np.uint16)

    las_data.write(str(out_file))
    return str(out_file)


if __name__ == "__main__":
    out = generate_synthetic_lidar()
    print(f"Synthetic LiDAR point cloud generated at: {out}")
