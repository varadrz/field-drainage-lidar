"""
Export README Visual Assets
Renders publication-grade visual figures from the actual app pipelines
and saves them to assets/ for embedding in GitHub README.md.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from modules.data_loader import load_raster
from modules.hydrology import run_hydrological_analysis
from modules.drainage_recommendation import generate_drainage_plan
from modules.lidar_processor import process_lidar_point_cloud
from modules.drainage_extraction import run_ditch_extraction
from components.elevation_profile import create_trench_profile_figure

# Output directory
assets_dir = Path("assets")
assets_dir.mkdir(parents=True, exist_ok=True)

# 1. Load data
dem_path = "data/sample/synthetic_indian_farm.tif"
las_path = "data/sample/synthetic_indian_farm.las"

elevation, metadata = load_raster(dem_path)
hydro_res = run_hydrological_analysis(dem_path)
drainage_plan = generate_drainage_plan(dem_path, hydrology_result=hydro_res)
lidar_res = process_lidar_point_cloud(las_path)
ditch_res = run_ditch_extraction(elevation, metadata, ground_points_xyz=lidar_res.ground_points_xyz)

# --- FIGURE 1: Comprehensive Multi-Panel GIS Dashboard Preview ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#0B0D10", dpi=130)

# Panel A: Elevation DTM & Ditch Network
ax1 = axes[0]
ax1.set_facecolor("#14181F")
im1 = ax1.imshow(elevation, cmap="terrain", origin="upper")
for d in ditch_res.ditches:
    px_coords = [metadata.transform_rc(pt[0], pt[1]) if hasattr(metadata, "transform_rc") else None for pt in d.coords_native]
# Manual pixel conversion for ditch lines
aff = ~rasterio.transform.Affine(*metadata.transform[:6]) if False else None
from rasterio.transform import Affine
inv_aff = ~Affine(*metadata.transform[:6])

for d in ditch_res.ditches:
    rcs = [inv_aff @ pt for pt in d.coords_native]
    cols = [pt[0] for pt in rcs]
    rows = [pt[1] for pt in rcs]
    ax1.plot(cols, rows, color="#00F5D4", linewidth=2.5, label="Extracted Ditch" if d.segment_id == 1 else "")

for c in ditch_res.culverts:
    rcs = [inv_aff @ pt for pt in c.coords_native]
    cols = [pt[0] for pt in rcs]
    rows = [pt[1] for pt in rcs]
    ax1.plot(cols, rows, color="#F59E0B", linewidth=3.0, linestyle="--", label="Bridged Culvert" if c.culvert_id == 1 else "")

ax1.set_title("Bare-Earth DTM & Roelens Ditch Extraction", color="#F0F4F8", fontsize=12, fontweight="bold")
ax1.tick_params(colors="#94A3B8")
for spine in ax1.spines.values():
    spine.set_color("#2D3748")
cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.ax.tick_params(colors="#94A3B8")
cbar1.set_label("Elevation (m MSL)", color="#F0F4F8")
ax1.legend(facecolor="#14181F", edgecolor="#2D3748", labelcolor="#F0F4F8", loc="upper left")

# Panel B: Ponding Waterlogging Depressions & D8 Flow Accumulation
ax2 = axes[1]
ax2.set_facecolor("#14181F")
# Hillshade background
from modules.terrain_analysis import compute_hillshade
hs = compute_hillshade(elevation, metadata.res_x, metadata.res_y)
ax2.imshow(hs, cmap="gray", origin="upper", alpha=0.55)

dep = hydro_res.depression_depth_array
dep_masked = np.ma.masked_where(dep <= 0.02, dep)
im2 = ax2.imshow(dep_masked, cmap="Blues", origin="upper", vmin=0.0, vmax=0.5)

# Plot sink pits
for s in drainage_plan.priority_sinks:
    r, c = s.pit_rc
    ax2.scatter([c], [r], color="#EF4444", s=80, marker="x", linewidths=2.5, label="Deepest Pit Invert" if s.sink_id == 1 else "")

ax2.set_title("Waterlogging Depth & Ponding Sinks", color="#F0F4F8", fontsize=12, fontweight="bold")
ax2.tick_params(colors="#94A3B8")
for spine in ax2.spines.values():
    spine.set_color("#2D3748")
cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.ax.tick_params(colors="#94A3B8")
cbar2.set_label("Ponding Depth (m)", color="#F0F4F8")
ax2.legend(facecolor="#14181F", edgecolor="#2D3748", labelcolor="#F0F4F8", loc="upper left")

# Panel C: Recommended Surface Drainage Route
ax3 = axes[2]
ax3.set_facecolor("#14181F")
ax3.imshow(elevation, cmap="terrain", origin="upper", alpha=0.7)

best_r = drainage_plan.best_route
if best_r:
    rcs = np.array(best_r.path_rc)
    ax3.plot(rcs[:, 1], rcs[:, 0], color="#2ECC71", linewidth=4.0, label=f"Recommended Trench ({best_r.length_m:.1f} m)")
    ax3.scatter([rcs[0, 1]], [rcs[0, 0]], color="#00F5D4", s=90, label="Sink Pit Start", zorder=5)
    ax3.scatter([rcs[-1, 1]], [rcs[-1, 0]], color="#A78BFA", s=90, label="Field Outlet", zorder=5)

for out in drainage_plan.candidate_outlets:
    ax3.scatter([out.rc[1]], [out.rc[0]], color="#38BDF8", s=60, edgecolors="#FFFFFF", linewidths=1.2, label="Boundary Outlet" if out.outlet_id == 1 else "", zorder=4)

ax3.set_title("Engineered Least-Cost Drainage Route", color="#F0F4F8", fontsize=12, fontweight="bold")
ax3.tick_params(colors="#94A3B8")
for spine in ax3.spines.values():
    spine.set_color("#2D3748")
ax3.legend(facecolor="#14181F", edgecolor="#2D3748", labelcolor="#F0F4F8", loc="upper left")

fig.tight_layout()
fig.savefig(assets_dir / "dashboard_multi_view.png", dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.close(fig)
print("Saved dashboard_multi_view.png")

# --- FIGURE 2: Longitudinal Trench Elevation Cross-Section ---
if drainage_plan.best_route:
    fig_profile = create_trench_profile_figure(drainage_plan.best_route)
    fig_profile.savefig(assets_dir / "trench_cross_section_profile.png", dpi=150, facecolor="#14181F", bbox_inches="tight")
    plt.close(fig_profile)
    print("Saved trench_cross_section_profile.png")

# --- FIGURE 3: 3D Topographic Terrain View ---
from mpl_toolkits.mplot3d import Axes3D
fig_3d = plt.figure(figsize=(11, 6), facecolor="#0B0D10", dpi=140)
ax_3d = fig_3d.add_subplot(111, projection="3d")
ax_3d.set_facecolor("#0B0D10")

nrows, ncols = elevation.shape
step = 2
x_grid = np.arange(0, ncols, step) * metadata.res_x
y_grid = np.arange(0, nrows, step) * metadata.res_y
X, Y = np.meshgrid(x_grid, y_grid)
Z = elevation[::step, ::step]

# 10x vertical exaggeration for subtle 0.5% agricultural slope
Z_exagg = (Z - Z.min()) * 10.0 + Z.min()

surf = ax_3d.plot_surface(X, Y, Z_exagg, cmap="terrain", edgecolor="none", alpha=0.92, antialiased=True)

# Overlay 3D trench route
if best_r:
    rcs = np.array(best_r.path_rc)
    route_x = rcs[:, 1] * metadata.res_x
    route_y = rcs[:, 0] * metadata.res_y
    route_z = np.array([elevation[r, c] for r, c in best_r.path_rc])
    route_z_exagg = (route_z - Z.min()) * 10.0 + Z.min() + 0.3
    ax_3d.plot(route_x, route_y, route_z_exagg, color="#00F5D4", linewidth=3.5, label="Proposed Trench Invert")

ax_3d.set_title("3D Bare-Earth Topography (10× Vertical Exaggeration)", color="#F0F4F8", fontsize=13, fontweight="bold", pad=16)
ax_3d.set_xlabel("Easting (meters)", color="#94A3B8", fontsize=9, labelpad=10)
ax_3d.set_ylabel("Northing (meters)", color="#94A3B8", fontsize=9, labelpad=10)
ax_3d.set_zlabel("Elevation (10× MSL)", color="#94A3B8", fontsize=9, labelpad=10)

ax_3d.tick_params(colors="#94A3B8", labelsize=8)
ax_3d.view_init(elev=32, azim=-125)
ax_3d.legend(facecolor="#14181F", edgecolor="#2D3748", labelcolor="#F0F4F8", loc="upper left")

fig_3d.tight_layout()
fig_3d.savefig(assets_dir / "3d_terrain_view.png", dpi=150, facecolor=fig_3d.get_facecolor(), bbox_inches="tight")
plt.close(fig_3d)
print("Saved 3d_terrain_view.png")

# --- FIGURE 4: LiDAR Point Cloud Classification Inspection ---
fig_pc, (ax_pc1, ax_pc2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="#0B0D10", dpi=130)

import laspy
las_data = laspy.read(las_path)
ground_mask = las_data.classification == 2
veg_mask = las_data.classification == 5

# Subsample for plotting
np.random.seed(42)
g_sub = np.random.choice(np.where(ground_mask)[0], size=8000, replace=False)
v_sub = np.random.choice(np.where(veg_mask)[0], size=min(1500, np.sum(veg_mask)), replace=False)

ax_pc1.set_facecolor("#14181F")
ax_pc1.scatter(las_data.x[g_sub] - 340000, las_data.y[g_sub] - 2050000, c="#38BDF8", s=3, alpha=0.5, label=f"Ground Returns (Class 2: {np.sum(ground_mask):,})")
ax_pc1.scatter(las_data.x[v_sub] - 340000, las_data.y[v_sub] - 2050000, c="#2ECC71", s=12, alpha=0.85, label=f"Trees/Bund Vegetation (Class 5: {np.sum(veg_mask):,})")
ax_pc1.set_title("LiDAR Point Cloud ASPRS Classification", color="#F0F4F8", fontsize=11, fontweight="bold")
ax_pc1.set_xlabel("Local Easting (m)", color="#94A3B8", fontsize=9)
ax_pc1.set_ylabel("Local Northing (m)", color="#94A3B8", fontsize=9)
ax_pc1.tick_params(colors="#94A3B8")
for spine in ax_pc1.spines.values():
    spine.set_color("#2D3748")
ax_pc1.legend(facecolor="#14181F", edgecolor="#2D3748", labelcolor="#F0F4F8", loc="upper right")

# Elevation histogram comparison (Ground vs Vegetation)
ax_pc2.set_facecolor("#14181F")
ax_pc2.hist(las_data.z[ground_mask], bins=40, color="#38BDF8", alpha=0.75, label="Ground Points (Class 2)")
ax_pc2.hist(las_data.z[veg_mask], bins=40, color="#2ECC71", alpha=0.75, label="Vegetation Canopy Points (Class 5)")
ax_pc2.set_title("Vertical Point Distribution (Ground vs Tree Canopy)", color="#F0F4F8", fontsize=11, fontweight="bold")
ax_pc2.set_xlabel("Elevation (m above MSL)", color="#94A3B8", fontsize=9)
ax_pc2.set_ylabel("Laser Return Count", color="#94A3B8", fontsize=9)
ax_pc2.tick_params(colors="#94A3B8")
for spine in ax_pc2.spines.values():
    spine.set_color("#2D3748")
ax_pc2.legend(facecolor="#14181F", edgecolor="#2D3748", labelcolor="#F0F4F8", loc="upper right")

fig_pc.tight_layout()
fig_pc.savefig(assets_dir / "lidar_point_cloud_inspection.png", dpi=150, facecolor=fig_pc.get_facecolor(), bbox_inches="tight")
plt.close(fig_pc)
print("Saved lidar_point_cloud_inspection.png")

print("All README visual assets successfully exported to assets/!")
