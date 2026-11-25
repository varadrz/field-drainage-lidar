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

    # 1. Natural Ground Surface Line
    ax.plot(
        dist,
        ground,
        color="#38BDF8",
        linewidth=2.2,
        label="Natural Ground Level",
        zorder=3
    )

    # 2. Engineered Ditch Invert Bed Line (Monotonic Gravity Slope)
    ax.plot(
        dist,
        bed,
        color="#2ECC71",
        linewidth=2.5,
        linestyle="--",
        label=f"Engineered Ditch Bed ({route.bed_slope_pct:.2f}% Gravity Grade)",
        zorder=4
    )

    # 3. Excavation Fill Region
    ax.fill_between(
        dist,
        bed,
        ground,
        where=(ground >= bed),
        color="#F59E0B",
        alpha=0.35,
        label="Excavation Cut Zone",
        zorder=2
    )

    # 4. Highlight Maximum Cut Depth Station
    max_cut_idx = int(np.argmax(cut))
    max_cut_dist = dist[max_cut_idx]
    max_cut_ground = ground[max_cut_idx]
    max_cut_bed = bed[max_cut_idx]
    
    if route.max_cut_depth_m > 0.05:
        ax.vlines(
            x=max_cut_dist,
            ymin=max_cut_bed,
            ymax=max_cut_ground,
            color="#EF4444",
            linewidth=2.0,
            linestyle=":",
            label=f"Max Cut: {route.max_cut_depth_m:.2f} m",
            zorder=5
        )
        ax.scatter(
            [max_cut_dist],
            [max_cut_ground],
            color="#EF4444",
            s=45,
            zorder=6
        )

    # 5. Station Markers (Start & Outlet Invert)
    ax.scatter([dist[0]], [ground[0]], color="#00F5D4", s=60, zorder=6, label=f"Pit Invert ({ground[0]:.2f} m)")
    ax.scatter([dist[-1]], [bed[-1]], color="#A78BFA", s=60, zorder=6, label=f"Outlet Bed ({bed[-1]:.2f} m)")

    # 6. Dark Theme Styling
    title_text = f"Longitudinal Elevation Cross-Section &bull; {route.route_id}"
    if route.is_recommended:
        title_text += " [RECOMMENDED / OPTIMAL]"
    ax.set_title(title_text, color="#F0F4F8", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Distance from Waterlogging Pit (meters)", color="#94A3B8", fontsize=10, labelpad=8)
    ax.set_ylabel("Elevation (m above Sea Level)", color="#94A3B8", fontsize=10, labelpad=8)

    ax.tick_params(colors="#94A3B8", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#2D3748")

    ax.grid(True, color="#1F2937", linestyle=":", linewidth=0.8)

    # Legend
    ax.legend(
        facecolor="#14181F",
        edgecolor="#2D3748",
        labelcolor="#F0F4F8",
        fontsize=8.5,
        loc="upper right"
    )

    fig.tight_layout()
    return fig
