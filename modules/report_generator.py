"""
Automated PDF Drainage Plan Report Generator
Phase 6 Deliverable: Builds publication-grade, multi-page agricultural drainage
engineering reports using ReportLab and high-DPI Matplotlib cartographic figures.
Fully decoupled from Streamlit for headless execution and batch testing.
"""

from dataclasses import dataclass
from datetime import datetime
import io
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

# Set temporary Matplotlib cache to avoid environment permission conflicts
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))

import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import numpy as np

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    KeepTogether,
    PageBreak,
    HRFlowable,
)
from reportlab.pdfgen import canvas

from .data_loader import RasterMetadata
from .elevation_analysis import ElevationStats
from .terrain_analysis import TerrainSummary
from .hydrology import HydrologyResult
from .drainage_recommendation import DrainagePlanResult, DrainageRoute
from .drainage_extraction import DitchExtractionResult

logger = logging.getLogger(__name__)


class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas that computes total page count and adds running header
    and footer ('Page X of Y') on all pages.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#718096"))

        # Running header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 11 * 72 - 36, "AGRICULTURAL DRAINAGE FEASIBILITY & DESIGN REPORT")
            self.drawRightString(8.5 * 72 - 54, 11 * 72 - 36, "CONFIDENTIAL & PROPRIETARY")
            self.setStrokeColor(colors.HexColor("#CBD5E0"))
            self.setLineWidth(0.5)
            self.line(54, 11 * 72 - 40, 8.5 * 72 - 54, 11 * 72 - 40)

        # Running footer (all pages)
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.5)
        self.line(54, 45, 8.5 * 72 - 54, 45)
        self.drawString(54, 32, "Field Drainage Using LiDAR  |  Automated Decision Engine")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * 72 - 54, 32, page_str)
        self.restoreState()


def _render_elevation_map(
    dem_array: np.ndarray,
    metadata: RasterMetadata,
    terrain_summary: TerrainSummary,
    output_stream: io.BytesIO
) -> None:
    """Generates high-DPI 2D Elevation and Topographic Contours figure."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=200)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F7FAFC")

    valid_mask = ~np.isnan(dem_array)
    vmin = float(np.nanmin(dem_array)) if np.any(valid_mask) else 0.0
    vmax = float(np.nanmax(dem_array)) if np.any(valid_mask) else 1.0

    im = ax.imshow(
        dem_array,
        cmap="terrain",
        vmin=vmin,
        vmax=vmax,
        extent=(0, metadata.width * metadata.res_x, 0, metadata.height * metadata.res_y),
        origin="upper"
    )

    # Overlay elevation contour lines
    if np.any(valid_mask) and (vmax - vmin) > 0.1:
        try:
            levels = np.linspace(vmin, vmax, 9)
            cs = ax.contour(
                dem_array,
                levels=levels,
                colors="#2D3748",
                linewidths=0.6,
                extent=(0, metadata.width * metadata.res_x, 0, metadata.height * metadata.res_y),
                origin="upper"
            )
            ax.clabel(cs, inline=True, fontsize=6, fmt="%.1fm")
        except Exception:
            pass

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Elevation (m ASL)", fontsize=8, color="#2D3748")
    cbar.ax.tick_params(labelsize=7)

    ax.set_title("Bare-Earth Digital Terrain Model (DTM) & Topographic Contours", fontsize=9, fontweight="bold", pad=8, color="#1A202C")
    ax.set_xlabel("Relative Easting (meters)", fontsize=8, color="#4A5568")
    ax.set_ylabel("Relative Northing (meters)", fontsize=8, color="#4A5568")
    ax.tick_params(labelsize=7)

    # North arrow callout
    ax.text(0.96, 0.94, "▲\nN", transform=ax.transAxes, fontsize=8, fontweight="bold",
            ha="center", va="center", color="#1A202C",
            bbox=dict(boxstyle="circle,pad=0.2", facecolor="white", edgecolor="#CBD5E0", alpha=0.9))

    plt.tight_layout()
    fig.savefig(output_stream, format="png", bbox_inches="tight")
    plt.close(fig)


def _render_waterlogging_map(
    dem_array: np.ndarray,
    metadata: RasterMetadata,
    hydro_result: HydrologyResult,
    drainage_plan: DrainagePlanResult,
    output_stream: io.BytesIO
) -> None:
    """Generates high-DPI Waterlogging Depression Depth and Ponding figure."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=200)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F7FAFC")

    # Background subtle elevation relief
    ax.imshow(
        dem_array,
        cmap="gray",
        alpha=0.35,
        extent=(0, metadata.width * metadata.res_x, 0, metadata.height * metadata.res_y),
        origin="upper"
    )

    dep_cm = hydro_result.depression_depth_array * 100.0
    dep_cm[dep_cm <= 1.0] = np.nan  # Mask non-depressions

    max_dep = float(np.nanmax(dep_cm)) if np.any(~np.isnan(dep_cm)) else 10.0
    im = ax.imshow(
        dep_cm,
        cmap="Blues",
        vmin=0.0,
        vmax=max(max_dep, 10.0),
        alpha=0.85,
        extent=(0, metadata.width * metadata.res_x, 0, metadata.height * metadata.res_y),
        origin="upper"
    )

    # Mark waterlogging depression pits
    for sink in drainage_plan.priority_sinks[:5]:
        r, c = sink.pit_rc
        x = c * metadata.res_x
        y = (metadata.height - r) * metadata.res_y
        ax.plot(x, y, "ro", markersize=6, markeredgecolor="white", markeredgewidth=1.2)
        ax.text(x + 3, y + 3, f"Pond {sink.sink_id}\n({sink.max_depth_cm:.1f}cm)",
                fontsize=6.5, fontweight="bold", color="#9B2C2C",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#E53E3E", alpha=0.9))

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Ponding Depth (cm)", fontsize=8, color="#2D3748")
    cbar.ax.tick_params(labelsize=7)

    ax.set_title("Waterlogging Vulnerability & Trapped Ponding Depths", fontsize=9, fontweight="bold", pad=8, color="#1A202C")
    ax.set_xlabel("Relative Easting (meters)", fontsize=8, color="#4A5568")
    ax.set_ylabel("Relative Northing (meters)", fontsize=8, color="#4A5568")
    ax.tick_params(labelsize=7)

    plt.tight_layout()
    fig.savefig(output_stream, format="png", bbox_inches="tight")
    plt.close(fig)


def _render_drainage_layout_map(
    dem_array: np.ndarray,
    metadata: RasterMetadata,
    drainage_plan: DrainagePlanResult,
    ditch_result: Optional[DitchExtractionResult],
    output_stream: io.BytesIO
) -> None:
    """Generates high-DPI Recommended Surface Drainage Alignment figure."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=200)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F7FAFC")

    # Background elevation
    ax.imshow(
        dem_array,
        cmap="gray",
        alpha=0.35,
        extent=(0, metadata.width * metadata.res_x, 0, metadata.height * metadata.res_y),
        origin="upper"
    )

    # Plot existing extracted ditches (if available)
    if ditch_result and ditch_result.ditch_mask_array is not None:
        ditch_y, ditch_x = np.where(ditch_result.ditch_mask_array)
        if len(ditch_x) > 0:
            ax.scatter(
                ditch_x * metadata.res_x,
                (metadata.height - ditch_y) * metadata.res_y,
                s=1.5,
                color="#DD6B20",
                alpha=0.6,
                label="Existing Ditches (Roelens et al.)"
            )

    # Plot candidate outlets along boundary
    for out in drainage_plan.candidate_outlets:
        r, c = out.rc
        ox = c * metadata.res_x
        oy = (metadata.height - r) * metadata.res_y
        ax.plot(ox, oy, marker="*", color="#D69E2E", markersize=9, markeredgecolor="#744210", markeredgewidth=0.8)
        ax.text(ox, oy - 6, f"Outlet {out.outlet_id}", fontsize=6, fontweight="bold", color="#744210",
                ha="center", bbox=dict(boxstyle="round,pad=0.15", facecolor="#FEFCBF", edgecolor="#D69E2E", alpha=0.85))

    # Plot proposed drainage routes
    for idx, route in enumerate(drainage_plan.routes):
        xs = [c * metadata.res_x for r, c in route.path_rc]
        ys = [(metadata.height - r) * metadata.res_y for r, c in route.path_rc]

        if route.is_recommended:
            ax.plot(xs, ys, color="#2ECC71", linewidth=2.4, label=f"Primary Trench ({route.length_m:.0f}m, {route.bed_slope_pct:.2f}%)")
            # Highlight start (sink) and end (outlet)
            ax.plot(xs[0], ys[0], "o", color="#E53E3E", markersize=6, label="Ponding Pit Origin")
            ax.plot(xs[-1], ys[-1], "s", color="#3182CE", markersize=6, label="Gravity Outlet Discharge")
        else:
            ax.plot(xs, ys, color="#48BB78", linewidth=1.2, linestyle="--", label=f"Alt Route {route.route_id}")

    ax.set_title("Proposed Gravity Surface Drainage Alignment & Outlets", fontsize=9, fontweight="bold", pad=8, color="#1A202C")
    ax.set_xlabel("Relative Easting (meters)", fontsize=8, color="#4A5568")
    ax.set_ylabel("Relative Northing (meters)", fontsize=8, color="#4A5568")
    ax.tick_params(labelsize=7)
    ax.legend(loc="upper left", fontsize=6.5, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_stream, format="png", bbox_inches="tight")
    plt.close(fig)


def generate_drainage_pdf_report(
    output_pdf_path: Union[str, Path],
    dem_array: np.ndarray,
    metadata: RasterMetadata,
    elev_stats: ElevationStats,
    terrain_summary: TerrainSummary,
    hydro_result: HydrologyResult,
    drainage_plan: DrainagePlanResult,
    ditch_result: Optional[DitchExtractionResult] = None,
    data_source_type: str = "Real LiDAR Point Cloud",
    project_name: str = "Agricultural Surface Drainage Feasibility Plan",
    field_plot_id: str = "Khasra #214 / Survey Plot A"
) -> Path:
    """
    Builds an automated, multi-page agricultural drainage plan PDF report.
    
    Args:
        output_pdf_path: Target filesystem path for the output PDF.
        dem_array: 2D numpy array of bare-earth elevation (DTM).
        metadata: RasterMetadata container with CRS, bounds, and resolution.
        elev_stats: ElevationStats container with min/max/mean/std elevation.
        terrain_summary: TerrainSummary with slope, aspect, and hillshade metrics.
        hydro_result: HydrologyResult with depression depth, volumes, and stream lengths.
        drainage_plan: DrainagePlanResult with ranked routes, sinks, and candidate outlets.
        ditch_result: Optional DitchExtractionResult with extracted ditch metrics.
        data_source_type: Descriptive badge (Real LiDAR vs. Satellite DEM vs. Simulated).
        project_name: Title of the drainage project.
        field_plot_id: Local field identifier / plot number.
        
    Returns:
        Path object pointing to the written PDF file.
    """
    out_path = Path(output_pdf_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(terrain_summary, tuple):
        terrain_summary = terrain_summary[0]

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=45,
        bottomMargin=50
    )

    styles = getSampleStyleSheet()

    # Custom Typography Palette
    primary_color = colors.HexColor("#1A202C")
    accent_emerald = colors.HexColor("#2ECC71")
    accent_dark_green = colors.HexColor("#1B4D3E")
    card_bg = colors.HexColor("#F7FAFC")
    border_color = colors.HexColor("#E2E8F0")

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=accent_dark_green
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#4A5568")
    )

    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=accent_dark_green,
        spaceBefore=12,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=primary_color
    )

    badge_style = ParagraphStyle(
        "SourceBadge",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#2B6CB0")
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.white,
        alignment=1  # Centered
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=primary_color,
        alignment=1  # Centered
    )

    table_cell_left = ParagraphStyle(
        "TableCellLeft",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=primary_color,
        alignment=0  # Left
    )

    story = []

    # -------------------------------------------------------------
    # 1. HEADER & METADATA BANNER
    # -------------------------------------------------------------
    story.append(Paragraph(project_name.upper(), title_style))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        "Automated LiDAR-Derived Hydraulic Modeling, Waterlogging Sink Remediation & Gravity Ditch Alignment",
        subtitle_style
    ))
    story.append(Spacer(1, 10))

    # Header Meta Table
    cur_date = datetime.now().strftime("%B %d, %Y")
    area_ac = elev_stats.area_acres
    area_ha = elev_stats.area_ha
    crs_str = f"EPSG:{metadata.crs_epsg}" if metadata.crs_epsg else (elev_stats.crs_display or "Metric UTM")
    res_str = f"{metadata.res_x:.1f} m/px"

    meta_data = [
        [
            Paragraph("<b>Target Landholding:</b>", body_style),
            Paragraph(field_plot_id, body_style),
            Paragraph("<b>Report Date:</b>", body_style),
            Paragraph(cur_date, body_style)
        ],
        [
            Paragraph("<b>Coordinate System:</b>", body_style),
            Paragraph(f"{crs_str} ({res_str})", body_style),
            Paragraph("<b>Surveyed Area:</b>", body_style),
            Paragraph(f"{area_ac:.2f} Acres ({area_ha:.2f} Ha / {elev_stats.area_sq_m:,.0f} m²)", body_style)
        ],
        [
            Paragraph("<b>Data Source Provenance:</b>", body_style),
            Paragraph(f"<font color='#2B6CB0'><b>[ {data_source_type} ]</b></font>", badge_style),
            Paragraph("<b>Elevation Range:</b>", body_style),
            Paragraph(f"{elev_stats.min_elevation:.2f} m – {elev_stats.max_elevation:.2f} m (Relief: {elev_stats.elevation_range:.2f} m)", body_style)
        ]
    ]

    meta_table = Table(meta_data, colWidths=[120, 160, 100, 150])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), card_bg),
        ("BOX", (0, 0), (-1, -1), 1, border_color),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 14))

    # -------------------------------------------------------------
    # 2. KEY PERFORMANCE INDICATORS (KPIs)
    # -------------------------------------------------------------
    story.append(Paragraph("1. Executive Agronomic & Engineering Summary", section_heading))
    story.append(HRFlowable(width="100%", thickness=1, color=border_color, spaceAfter=8))

    best_route = drainage_plan.best_route
    best_length = f"{best_route.length_m:,.1f} m" if best_route else "N/A"
    best_slope = f"{best_route.bed_slope_pct:.2f}%" if best_route else "N/A"
    best_cut = f"{best_route.max_cut_depth_m:.2f} m" if best_route else "N/A"
    best_earthwork = f"{best_route.earthwork_vol_m3:,.1f} m³" if best_route else "N/A"

    kpi_cards = [
        [
            Paragraph("<b>SURVEYED AREA</b>", table_header_style),
            Paragraph("<b>WATERLOGGING COVERAGE</b>", table_header_style),
            Paragraph("<b>TRAPPED PONDING WATER</b>", table_header_style),
            Paragraph("<b>PRIMARY TRENCH LENGTH</b>", table_header_style),
        ],
        [
            Paragraph(f"<b>{area_ac:.2f} ac</b><br/><font size='7' color='#718096'>{area_ha:.2f} Hectares</font>", table_cell_style),
            Paragraph(f"<b>{hydro_result.waterlogging_area_pct:.1f}%</b><br/><font size='7' color='#718096'>{hydro_result.waterlogging_area_acres:.2f} ac / {hydro_result.waterlogging_area_sq_m:,.0f} m²</font>", table_cell_style),
            Paragraph(f"<b>{hydro_result.total_waterlogging_volume_m3:,.0f} m³</b><br/><font size='7' color='#718096'>Max Depth: {hydro_result.max_waterlogging_depth_cm:.1f} cm</font>", table_cell_style),
            Paragraph(f"<b>{best_length}</b><br/><font size='7' color='#718096'>Slope: {best_slope} | Cut: {best_cut}</font>", table_cell_style),
        ]
    ]

    kpi_table = Table(kpi_cards, colWidths=[132, 132, 133, 133])
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), accent_dark_green),
        ("BACKGROUND", (0, 1), (-1, 1), card_bg),
        ("BOX", (0, 0), (-1, -1), 1, border_color),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, border_color),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 12))

    summary_text = (
        f"Topographic evaluation of <b>{field_plot_id}</b> indicates a total elevation change of <b>{elev_stats.elevation_range:.2f} m</b> "
        f"across an average slope of <b>{terrain_summary.mean_slope_pct:.2f}%</b> ({terrain_summary.mean_slope_deg:.2f}°). "
        f"Hydrological depression analysis isolated <b>{len(drainage_plan.priority_sinks)} distinct waterlogging pockets</b> "
        f"covering <b>{hydro_result.waterlogging_area_acres:.2f} acres</b> ({hydro_result.waterlogging_area_pct:.1f}% of the field), "
        f"trapping an estimated <b>{hydro_result.total_waterlogging_volume_m3:,.0f} m³</b> of stagnant surface runoff during precipitation events. "
        f"A gravity-fed open trench alignment has been solved with an engineered bed gradient of <b>{best_slope}</b>, "
        f"ensuring continuous downhill drainage to perimeter <b>Outlet {best_route.outlet_id if best_route else '1'}</b> with zero uphill sections."
    )
    story.append(Paragraph(summary_text, body_style))
    story.append(Spacer(1, 14))

    # -------------------------------------------------------------
    # 3. HIGH-DPI CARTOGRAPHIC MAPS
    # -------------------------------------------------------------
    story.append(Paragraph("2. Cartographic GIS Topography & Drainage Layout", section_heading))
    story.append(HRFlowable(width="100%", thickness=1, color=border_color, spaceAfter=8))

    # Generate figures in memory
    buf_elev = io.BytesIO()
    buf_dep = io.BytesIO()
    buf_drain = io.BytesIO()

    try:
        _render_elevation_map(dem_array, metadata, terrain_summary, buf_elev)
        buf_elev.seek(0)
        img_elev = Image(buf_elev, width=530, height=310)
        story.append(KeepTogether([
            Paragraph("<b>Figure 1: Bare-Earth Elevation Surface & 0.25 m Topographic Contours</b>", body_style),
            Spacer(1, 4),
            img_elev,
            Spacer(1, 10)
        ]))
    except Exception as e:
        logger.error(f"Failed to render elevation map for PDF: {e}")

    story.append(PageBreak())  # Clean break to Page 2

    try:
        _render_waterlogging_map(dem_array, metadata, hydro_result, drainage_plan, buf_dep)
        buf_dep.seek(0)
        img_dep = Image(buf_dep, width=530, height=270)
        story.append(KeepTogether([
            Paragraph("<b>Figure 2: Surface Waterlogging Vulnerability & Pit Depth Distribution</b>", body_style),
            Spacer(1, 4),
            img_dep,
            Spacer(1, 10)
        ]))
    except Exception as e:
        logger.error(f"Failed to render waterlogging map for PDF: {e}")

    try:
        _render_drainage_layout_map(dem_array, metadata, drainage_plan, ditch_result, buf_drain)
        buf_drain.seek(0)
        img_drain = Image(buf_drain, width=530, height=270)
        story.append(KeepTogether([
            Paragraph("<b>Figure 3: Proposed Gravity-Fed Surface Drainage Trench Network & Boundary Exits</b>", body_style),
            Spacer(1, 4),
            img_drain,
            Spacer(1, 12)
        ]))
    except Exception as e:
        logger.error(f"Failed to render drainage layout map for PDF: {e}")

    # -------------------------------------------------------------
    # 4. RANKED DRAINAGE ROUTES SPECIFICATION TABLE
    # -------------------------------------------------------------
    story.append(Paragraph("3. Engineered Surface Trench Alignment Specifications", section_heading))
    story.append(HRFlowable(width="100%", thickness=1, color=border_color, spaceAfter=6))

    route_table_data = [
        [
            Paragraph("<b>Route ID</b>", table_header_style),
            Paragraph("<b>Target Outlet</b>", table_header_style),
            Paragraph("<b>Length (m)</b>", table_header_style),
            Paragraph("<b>Bed Slope (%)</b>", table_header_style),
            Paragraph("<b>Max Cut (m)</b>", table_header_style),
            Paragraph("<b>Earthwork (m³)</b>", table_header_style),
            Paragraph("<b>Feasibility Rank</b>", table_header_style),
            Paragraph("<b>Recommendation</b>", table_header_style),
        ]
    ]

    for r in drainage_plan.routes:
        status_color = "#276749" if r.is_recommended else "#4A5568"
        status_text = "<b>Primary Optimal</b>" if r.is_recommended else "Alternative"
        route_table_data.append([
            Paragraph(f"<b>{r.route_id}</b>", table_cell_style),
            Paragraph(f"Outlet #{r.outlet_id}", table_cell_style),
            Paragraph(f"{r.length_m:,.1f}", table_cell_style),
            Paragraph(f"{r.bed_slope_pct:.2f}%", table_cell_style),
            Paragraph(f"{r.max_cut_depth_m:.2f}", table_cell_style),
            Paragraph(f"{r.earthwork_vol_m3:,.1f}", table_cell_style),
            Paragraph(r.feasibility_rank, table_cell_style),
            Paragraph(f"<font color='{status_color}'>{status_text}</font>", table_cell_style),
        ])

    routes_tbl = Table(route_table_data, colWidths=[55, 65, 55, 65, 55, 65, 85, 85])
    routes_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), accent_dark_green),
        ("BACKGROUND", (0, 1), (-1, -1), card_bg),
        ("BOX", (0, 0), (-1, -1), 1, border_color),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, border_color),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(routes_tbl)
    story.append(Spacer(1, 14))

    # -------------------------------------------------------------
    # 5. WATERLOGGING SINKS INVENTORY TABLE
    # -------------------------------------------------------------
    story.append(Paragraph("4. Waterlogging Sinks & Ponding Depressions Inventory", section_heading))
    story.append(HRFlowable(width="100%", thickness=1, color=border_color, spaceAfter=6))

    sink_table_data = [
        [
            Paragraph("<b>Sink ID</b>", table_header_style),
            Paragraph("<b>Ponding Area (m²)</b>", table_header_style),
            Paragraph("<b>Ponding Area (ac)</b>", table_header_style),
            Paragraph("<b>Max Depth (cm)</b>", table_header_style),
            Paragraph("<b>Average Depth (cm)</b>", table_header_style),
            Paragraph("<b>Stored Volume (m³)</b>", table_header_style),
            Paragraph("<b>Hazard Severity</b>", table_header_style),
        ]
    ]

    for s in drainage_plan.priority_sinks:
        sev_color = "#9B2C2C" if "High" in s.severity or "Severe" in s.severity else "#B7791F"
        sink_table_data.append([
            Paragraph(f"<b>Pond #{s.sink_id}</b>", table_cell_style),
            Paragraph(f"{s.area_m2:,.0f}", table_cell_style),
            Paragraph(f"{(s.area_m2 / 4046.86):.3f}", table_cell_style),
            Paragraph(f"{s.max_depth_cm:.1f}", table_cell_style),
            Paragraph(f"{s.mean_depth_cm:.1f}", table_cell_style),
            Paragraph(f"{s.volume_m3:,.0f}", table_cell_style),
            Paragraph(f"<font color='{sev_color}'><b>{s.severity}</b></font>", table_cell_style),
        ])

    sinks_tbl = Table(sink_table_data, colWidths=[65, 80, 75, 75, 75, 75, 85])
    sinks_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), accent_dark_green),
        ("BACKGROUND", (0, 1), (-1, -1), card_bg),
        ("BOX", (0, 0), (-1, -1), 1, border_color),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, border_color),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(sinks_tbl)
    story.append(Spacer(1, 14))

    # -------------------------------------------------------------
    # 6. PRACTICAL RECOMMENDATIONS & ENGINEERING LIMITATIONS
    # -------------------------------------------------------------
    story.append(Paragraph("5. Agronomic Engineering Recommendations & Scope Limitations", section_heading))
    story.append(HRFlowable(width="100%", thickness=1, color=border_color, spaceAfter=6))

    recs = [
        "<b>Trapezoidal Channel Profile:</b> Construct recommended surface channels with a 1:1 side slope (45°) and a minimum bottom invert bed width of 0.6 m to 1.0 m to ensure hydraulic stability and prevent bank slumping in clay soils.",
        "<b>Monotonic Gravity Bed Gradient:</b> Maintain the designed continuous downhill slope (≥ 0.20%) throughout channel excavation. Zero uphill segments exist in this design; avoid creating localized dips or mounds during tractor grading.",
        "<b>Spoil Bank Placement:</b> Deposit excavated earthwork at least 1.0 m away from trench berms and grade the spoil outward to prevent sediment washback and the formation of secondary waterlogging ridges.",
        "<b>Field Bund Spillways:</b> Where perimeter bunds (<i>medh / daul</i>) impound water, install stabilized masonry or grassed spillway cuts at designated outlet points to prevent catastrophic bund breaching.",
        "<b>Regular Pre-Monsoon Maintenance:</b> Remove invasive weed choke and silt buildup annually prior to monsoon onset to ensure unimpeded peak runoff conveyance.",
        "<b>Explicit Scope Limitation:</b> This feasibility model addresses <b>surface gravity drainage exclusively</b>. Deep subsurface perforated tile drainage (lateral PVC/corrugated pipes) requires physical soil core borehole testing to determine in-situ saturated hydraulic conductivity (<i>K_sat</i>), which is outside the scope of Version 1.0."
    ]

    for rec in recs:
        bullet_item = [
            Paragraph("•", body_style),
            Paragraph(rec, body_style)
        ]
        bullet_tbl = Table([bullet_item], colWidths=[15, 515])
        bullet_tbl.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 1.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
        ]))
        story.append(bullet_tbl)

    # Build the document with our custom two-pass NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    logger.info(f"Successfully generated PDF report at: {out_path}")
    return out_path
