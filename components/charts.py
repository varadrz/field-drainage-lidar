"""
Interactive 3D Field Terrain Component
Generates 3D surface visualizations of agricultural topography using Plotly.
Supports automatic raster downsampling for smooth web rendering, user-configurable
vertical exaggeration, 3D drainage route line overlays, and interactive rainfall/flooding simulation.
Styled in a minimalist dark theme.
"""

from typing import Optional, List, Tuple
import numpy as np
import plotly.graph_objects as go

from modules.data_loader import RasterMetadata
from modules.drainage_recommendation import DrainageRoute


def create_3d_terrain_figure(
    elevation: np.ndarray,
    metadata: RasterMetadata,
    selected_route: Optional[DrainageRoute] = None,
    vertical_exaggeration: float = 10.0,
    flood_depth_m: float = 0.0,
    colormap: str = "Viridis",
    max_grid: int = 200
) -> go.Figure:
    """
    Constructs an interactive 3D terrain surface figure.
    
    Args:
        elevation: 2D numpy array of elevations in meters.
        metadata: RasterMetadata holding spatial resolution and dimensions.
        selected_route: Optional DrainageRoute to overlay in 3D.
        vertical_exaggeration: Multiplier for Z-axis relief (default 10x).
        flood_depth_m: Simulated water height in meters above minimum elevation.
        colormap: Plotly colorscale (e.g. 'Viridis', 'Earth', 'Turbo', 'YlGnBu').
        max_grid: Maximum resolution along either dimension to preserve browser FPS.
        
    Returns:
        Plotly go.Figure object ready for st.plotly_chart().
    """
    nrows, ncols = elevation.shape

    # 1. Automatic Downsampling for high-performance browser rendering
    step = max(1, int(np.ceil(max(nrows, ncols) / max_grid)))
    elev_sub = elevation[::step, ::step].copy()
    sub_nrows, sub_ncols = elev_sub.shape

    # Handle NoData / NaNs
    valid_mask = ~np.isnan(elev_sub)
    min_elev = float(np.nanmin(elev_sub[valid_mask])) if np.any(valid_mask) else 0.0
    max_elev = float(np.nanmax(elev_sub[valid_mask])) if np.any(valid_mask) else 1.0
    elev_sub[~valid_mask] = min_elev

    # 2. Metric Grid Coordinates (distance in meters from origin)
    x_coords = np.arange(0, ncols, step) * metadata.res_x
    y_coords = np.arange(0, nrows, step) * metadata.res_y

    # Truncate coordinates to match downsampled slice shape if necessary
    x_coords = x_coords[:sub_ncols]
    y_coords = y_coords[:sub_nrows]

    # 3. Apply Vertical Exaggeration
    # Real subtle farm slopes (e.g. 0.5% grade) are stretched vertically for visual clarity
    z_scaled = min_elev + (elev_sub - min_elev) * vertical_exaggeration

    fig = go.Figure()

    # 4. 3D Terrain Surface Trace with Wireframe Contours
    surface_trace = go.Surface(
        x=x_coords,
        y=y_coords,
        z=z_scaled,
        customdata=elev_sub,
        colorscale=[
            [0.0, "#081018"],
            [0.2, "#0B253A"],
            [0.45, "#0D9488"],
            [0.75, "#10B981"],
            [1.0, "#A3E635"]
        ],
        cmin=float(np.nanmin(z_scaled)),
        cmax=float(np.nanmax(z_scaled)),
        contours=dict(
            z=dict(show=True, usecolormap=False, color="#00E5FF", width=1.5, project_z=False),
            x=dict(show=True, usecolormap=False, color="rgba(46, 204, 113, 0.35)", width=1, project_x=False),
            y=dict(show=True, usecolormap=False, color="rgba(46, 204, 113, 0.35)", width=1, project_y=False)
        ),
        colorbar=dict(
            title=dict(
                text=f"Elevation (m MSL)<br><sub>{vertical_exaggeration:.0f}× Exaggeration</sub>",
                font=dict(color="#F0F4F8", size=10, family="Inter")
            ),
            tickfont=dict(color="#8B949E", size=9, family="JetBrains Mono"),
            len=0.65,
            thickness=12,
            x=1.02
        ),
        hovertemplate=(
            "<b>Field Topography</b><br>"
            "Easting: %{x:.1f} m<br>"
            "Northing: %{y:.1f} m<br>"
            "Real Elevation: %{customdata:.2f} m MSL<br>"
            "<extra></extra>"
        ),
        lighting=dict(
            ambient=0.45,
            diffuse=0.8,
            specular=0.25,
            roughness=0.5,
            fresnel=0.15
        ),
        name="Field Terrain"
    )
    fig.add_trace(surface_trace)

    # 5. Optional Simulated Rainfall / Flooding Plane (Electric Cyan Translucent)
    if flood_depth_m > 0.0:
        real_water_level = min_elev + flood_depth_m
        scaled_water_level = min_elev + flood_depth_m * vertical_exaggeration
        water_z = np.full_like(z_scaled, scaled_water_level)

        water_trace = go.Surface(
            x=x_coords,
            y=y_coords,
            z=water_z,
            colorscale=[[0, "rgba(0, 229, 255, 0.45)"], [1, "rgba(0, 229, 255, 0.45)"]],
            showscale=False,
            opacity=0.65,
            name=f"Water Plane (+{flood_depth_m:.2f} m)",
            hovertemplate=(
                f"<b>Simulated Water Surface</b><br>"
                f"Water Level: {real_water_level:.2f} m MSL (+{flood_depth_m:.2f} m)<br>"
                f"<extra></extra>"
            )
        )
        fig.add_trace(water_trace)

    # 6. Overlay Recommended Drainage Route in 3D
    if selected_route is not None and selected_route.path_rc:
        route_rc = np.array(selected_route.path_rc)
        route_x = route_rc[:, 1] * metadata.res_x
        route_y = route_rc[:, 0] * metadata.res_y

        route_real_z = np.array([float(elevation[r, c]) for r, c in selected_route.path_rc])
        # Lift line slightly above terrain so it does not clip through the surface mesh
        route_scaled_z = min_elev + (route_real_z - min_elev) * vertical_exaggeration + (0.08 * vertical_exaggeration)

        route_trace = go.Scatter3d(
            x=route_x,
            y=route_y,
            z=route_scaled_z,
            customdata=np.stack([route_real_z, np.array(selected_route.cut_depth_profile_m)], axis=-1),
            mode="lines+markers",
            line=dict(
                color="#00E5FF",
                width=6
            ),
            marker=dict(
                size=3.5,
                color="#2ECC71",
                symbol="circle"
            ),
            name=f"Trench: {selected_route.route_id}",
            hovertemplate=(
                f"<b>{selected_route.route_id}</b><br>"
                f"Easting: %{{x:.1f}} m<br>"
                f"Northing: %{{y:.1f}} m<br>"
                f"Ground Elevation: %{{customdata[0]:.2f}} m<br>"
                f"Trench Cut Depth: %{{customdata[1]:.2f}} m<br>"
                f"<extra></extra>"
            )
        )
        fig.add_trace(route_trace)

    # 7. Layout & Minimalist Dark Theme (Isometric View)
    aspect_ratio_y = (nrows / ncols) if ncols > 0 else 1.0
    aspect_ratio_z = max(0.2, min(0.6, 0.35 * (vertical_exaggeration / 10.0)))

    fig.update_layout(
        paper_bgcolor="#14181F",
        plot_bgcolor="#0B0D10",
        title=dict(
            text=f"3D Topographic Mesh &bull; {vertical_exaggeration:.0f}× Relief",
            font=dict(color="#F0F4F8", size=13, family="Inter")
        ),
        margin=dict(l=0, r=0, b=0, t=32),
        scene=dict(
            bgcolor="#0B0D10",
            xaxis=dict(
                title="Easting (m)",
                color="#8B949E",
                gridcolor="#232A35",
                showbackground=True,
                backgroundcolor="#14181F"
            ),
            yaxis=dict(
                title="Northing (m)",
                color="#8B949E",
                gridcolor="#232A35",
                showbackground=True,
                backgroundcolor="#14181F"
            ),
            zaxis=dict(
                title=f"Elevation ({vertical_exaggeration:.0f}×)",
                color="#8B949E",
                gridcolor="#232A35",
                showbackground=True,
                backgroundcolor="#14181F"
            ),
            aspectmode="manual",
            aspectratio=dict(
                x=1.0,
                y=aspect_ratio_y,
                z=aspect_ratio_z
            ),
            camera=dict(
                eye=dict(x=-1.4, y=-1.4, z=1.05)  # Clean Isometric Perspective
            )
        ),
        legend=dict(
            font=dict(color="#F0F4F8", size=10, family="Inter"),
            bgcolor="rgba(20, 24, 31, 0.85)",
            bordercolor="#232A35",
            borderwidth=1,
            x=0.02,
            y=0.98
        )
    )

    return fig
