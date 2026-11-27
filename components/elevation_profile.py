"""
Longitudinal Elevation Profile Component
Renders high-resolution cross-sectional profiles along proposed drainage trenches.
Visualizes natural ground surface vs. engineered ditch bed invert, highlighting
excavation cut depth, bed slope gradient, and key station points.
Styled in a sleek, minimalist dark theme.
"""

import os
from pathlib import Path
from typing import Optional
import numpy as np

# Configure local Matplotlib cache directory to avoid permissions issues
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from modules.drainage_recommendation import DrainageRoute


def create_trench_profile_figure(route: DrainageRoute) -> Figure:
    """
    Builds a styled Matplotlib Figure showing the elevation cross-section of a drainage route.
    
    Args:
        route: DrainageRoute dataclass containing ground profile, bed profile, and metrics.
        
    Returns:
        matplotlib.figure.Figure object ready for st.pyplot().
    """
    dist = np.array(route.distance_profile_m)
    ground = np.array(route.ground_elev_profile_m)
    bed = np.array(route.bed_elev_profile_m)
    cut = np.array(route.cut_depth_profile_m)

    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor="#14181F", dpi=100)
    ax.set_facecolor("#0B0D10")

    # 1. Natural Ground Surface Line & Subtle Gradient Fill
    ax.plot(
        dist,
        ground,
        color="#8B949E",
        linewidth=2.2,
        label="Existing Ground Surface",
        zorder=3
    )
    # Subtle base fill under existing ground
    y_min_base = float(min(np.min(bed), np.min(ground))) - 0.05
    ax.fill_between(
        dist,
        y_min_base,
        ground,
        color="#8B949E",
        alpha=0.07,
        zorder=1
    )

    # 2. Engineered Ditch Invert Bed Line (Electric Cyan Monotonic Downward Slope)
    ax.plot(
        dist,
        bed,
        color="#00E5FF",
        linewidth=2.5,
        linestyle="--",
        label=f"Proposed Ditch Bed Grade ({route.bed_slope_pct:.2f}% Gravity Slope)",
        zorder=4
    )

    # 3. Excavation Cut Volume Fill (Soft Amber Translucent)
    ax.fill_between(
        dist,
        bed,
        ground,
        where=(ground >= bed),
        color="#F59E0B",
        alpha=0.28,
        label="Excavation Cut Zone",
        zorder=2
    )

    # 4. Highlight Maximum Cut Depth Station with Amber Annotation
    max_cut_idx = int(np.argmax(cut))
    max_cut_dist = dist[max_cut_idx]
    max_cut_ground = ground[max_cut_idx]
    max_cut_bed = bed[max_cut_idx]
    
    if route.max_cut_depth_m > 0.05:
        ax.vlines(
            x=max_cut_dist,
            ymin=max_cut_bed,
            ymax=max_cut_ground,
            color="#F59E0B",
            linewidth=2.0,
            linestyle=":",
            label=f"Max Cut: {route.max_cut_depth_m:.2f} m",
            zorder=5
        )
        ax.scatter(
            [max_cut_dist],
            [max_cut_ground],
            color="#F59E0B",
            s=50,
            zorder=6
        )
        # Small #F59E0B annotation badge
        ax.annotate(
            f"Max Cut: {route.max_cut_depth_m:.2f}m",
            xy=(max_cut_dist, (max_cut_ground + max_cut_bed) / 2.0),
            xytext=(max_cut_dist + max(1.0, (dist[-1] - dist[0]) * 0.04), (max_cut_ground + max_cut_bed) / 2.0),
            color="#F59E0B",
            fontsize=8.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#14181F", edgecolor="#F59E0B", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="#F59E0B", lw=1.0),
            zorder=7
        )

    # 5. Station Markers (Start Pit & Field Outlet)
    ax.scatter([dist[0]], [ground[0]], color="#2ECC71", s=70, zorder=6, label=f"Pit Invert ({ground[0]:.2f} m)")
    ax.scatter([dist[-1]], [bed[-1]], color="#00E5FF", s=70, zorder=6, label=f"Field Outlet ({bed[-1]:.2f} m)")

    # 6. Dark Theme & Precise Spines/Grids
    title_text = f"Proposed Channel Elevation Profile &bull; {route.route_id}"
    if route.is_recommended:
        title_text += " [OPTIMAL]"
    ax.set_title(title_text, color="#F0F4F8", fontsize=11.5, fontweight="bold", pad=12)
    ax.set_xlabel("Distance from Waterlogging Pit (m)", color="#8B949E", fontsize=9.5, labelpad=8)
    ax.set_ylabel("Elevation (m MSL)", color="#8B949E", fontsize=9.5, labelpad=8)

    ax.tick_params(colors="#8B949E", labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_color("#232A35")

    ax.grid(True, color="#232A35", linestyle=":", linewidth=0.8)

    # Legend
    ax.legend(
        facecolor="#14181F",
        edgecolor="#232A35",
        labelcolor="#F0F4F8",
        fontsize=8.5,
        loc="upper right"
    )

    fig.tight_layout()
    return fig
