"""
Drainage Recommendation Engine
Decoupled GIS and hydraulic optimization module.
Identifies priority waterlogging sinks, detects candidate boundary outlets,
computes least-cost gravity-fed surface trench routes, evaluates longitudinal
elevation and bed invert profiles, and computes earthwork excavation volumes.
Strictly preserved metric CRS and zero UI dependencies.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import json
import numpy as np
import scipy.ndimage
import skimage.graph
import shapely.geometry
import pyproj
import rasterio
from rasterio.transform import Affine

from .data_loader import RasterMetadata, load_raster
from .hydrology import HydrologyResult, run_hydrological_analysis


@dataclass
class WaterloggingSink:
    """Represents an identified physical waterlogging depression."""
    sink_id: int
    pit_rc: Tuple[int, int]
    pit_coord_native: Tuple[float, float]
    pit_coord_wgs84: Tuple[float, float]  # (lat, lon)
    centroid_native: Tuple[float, float]
    centroid_wgs84: Tuple[float, float]  # (lat, lon)
    area_m2: float
    volume_m3: float
    max_depth_m: float
    max_depth_cm: float
    mean_depth_cm: float
    severity: str
    geojson_feature: Dict[str, Any]


@dataclass
class CandidateOutlet:
    """Represents a potential gravity drainage exit point on the field boundary."""
    outlet_id: int
    rc: Tuple[int, int]
    coord_native: Tuple[float, float]
    coord_wgs84: Tuple[float, float]  # (lat, lon)
    elevation_m: float
    outlet_type: str  # 'stream_exit' or 'perimeter_low_point'
    geojson_feature: Dict[str, Any]


@dataclass
class DrainageRoute:
    """Engineered surface drainage trench route and cross-section metrics."""
    route_id: str
    sink_id: int
    outlet_id: int
    path_rc: List[Tuple[int, int]]
    path_coords_native: List[Tuple[float, float]]
    path_coords_wgs84: List[Tuple[float, float]]  # [[lat, lon], ...]
    length_m: float
    start_elev_m: float
    end_elev_m: float
    outlet_bed_elev_m: float
    avg_slope_pct: float
    bed_slope_pct: float
    max_cut_depth_m: float
    avg_cut_depth_m: float
    earthwork_vol_m3: float
    feasibility_rank: str
    is_recommended: bool
    distance_profile_m: List[float]
    ground_elev_profile_m: List[float]
    bed_elev_profile_m: List[float]
    cut_depth_profile_m: List[float]
    geojson_feature: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a farmer-friendly dictionary."""
        return {
            "Route ID": self.route_id,
            "Target Outlet ID": self.outlet_id,
            "Channel Length": f"{self.length_m:,.1f} m",
            "Natural Ground Slope": f"{self.avg_slope_pct:.2f}%",
            "Ditch Bed Slope": f"{self.bed_slope_pct:.2f}%",
            "Start Elevation": f"{self.start_elev_m:.2f} m",
            "Outlet Ground Elevation": f"{self.end_elev_m:.2f} m",
            "Max Trench Depth": f"{self.max_cut_depth_m:.2f} m",
            "Average Trench Depth": f"{self.avg_cut_depth_m:.2f} m",
            "Excavation Earthwork": f"{self.earthwork_vol_m3:,.1f} m³",
            "Feasibility Rating": self.feasibility_rank,
            "Status": "Recommended / Optimal" if self.is_recommended else "Alternative Candidate"
        }


@dataclass
class DrainagePlanResult:
    """Container for the comprehensive drainage recommendation plan."""
    routes: List[DrainageRoute]
    priority_sinks: List[WaterloggingSink]
    candidate_outlets: List[CandidateOutlet]
    best_route: Optional[DrainageRoute]
    total_ditch_length_m: float
    total_excavation_vol_m3: float
    routes_geojson: Dict[str, Any]
    sinks_geojson: Dict[str, Any]
    outlets_geojson: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Summarize plan metrics."""
        return {
            "Identified Priority Sinks": len(self.priority_sinks),
            "Candidate Field Outlets": len(self.candidate_outlets),
            "Evaluated Routes": len(self.routes),
            "Best Route Length": f"{self.best_route.length_m:,.1f} m" if self.best_route else "N/A",
            "Best Route Max Cut": f"{self.best_route.max_cut_depth_m:.2f} m" if self.best_route else "N/A",
            "Total Excavation Volume": f"{self.total_excavation_vol_m3:,.1f} m³",
            "Overall Feasibility": self.best_route.feasibility_rank if self.best_route else "No Route Found"
        }


def _rc_to_native(transform_tuple: Tuple[float, ...], r: float, c: float) -> Tuple[float, float]:
    """Converts pixel row and column to projected native coordinates (x, y)."""
    aff = Affine(*transform_tuple[:6])
    x, y = aff @ (c + 0.5, r + 0.5)
    return float(x), float(y)


def _get_wgs84_transformer(crs_epsg: Optional[int]) -> Optional[pyproj.Transformer]:
    """Creates a pyproj Transformer from native CRS to EPSG:4326."""
    if crs_epsg is None:
        return None
    try:
        return pyproj.Transformer.from_crs(crs_epsg, 4326, always_xy=True)
    except Exception:
        return None


def identify_priority_sinks(
    depression_depth: np.ndarray,
    metadata: RasterMetadata,
    min_depth_m: float = 0.05,
    min_area_m2: float = 10.0,
    min_volume_m3: float = 0.5
) -> List[WaterloggingSink]:
    """
    Identifies and ranks true topographic depressions from depression depth raster.
    Filters micro-tillage noise by requiring minimum area and trapped water volume.
    """
    res_x = metadata.res_x
    res_y = metadata.res_y
    pixel_area_m2 = res_x * res_y

    valid_depth = np.nan_to_num(depression_depth, nan=0.0)
    sink_binary = (valid_depth >= min_depth_m)

    labeled, num_features = scipy.ndimage.label(sink_binary, structure=np.ones((3, 3)))
    if num_features == 0:
        return []

    transformer = _get_wgs84_transformer(metadata.crs_epsg)
    sinks: List[WaterloggingSink] = []

    for feat_id in range(1, num_features + 1):
        idx = np.where(labeled == feat_id)
        pixel_count = len(idx[0])
        area_m2 = float(pixel_count * pixel_area_m2)

        if area_m2 < min_area_m2:
            continue

        depths = valid_depth[idx]
        volume_m3 = float(np.sum(depths) * pixel_area_m2)
        if volume_m3 < min_volume_m3:
            continue

        max_depth_m = float(np.max(depths))
        mean_depth_m = float(np.mean(depths))

        # Deepest point (pit)
        pit_local_idx = np.argmax(depths)
        pit_r = int(idx[0][pit_local_idx])
        pit_c = int(idx[1][pit_local_idx])
        pit_x, pit_y = _rc_to_native(metadata.transform, pit_r, pit_c)

        # Centroid
        cent_r = float(np.mean(idx[0]))
        cent_c = float(np.mean(idx[1]))
        cent_x, cent_y = _rc_to_native(metadata.transform, cent_r, cent_c)

        # WGS84 coordinates (lat, lon)
        if transformer:
            pit_lon, pit_lat = transformer.transform(pit_x, pit_y)
            cent_lon, cent_lat = transformer.transform(cent_x, cent_y)
        else:
            pit_lon, pit_lat = pit_x, pit_y
            cent_lon, cent_lat = cent_x, cent_y

        # Severity
        if max_depth_m >= 0.25:
            severity = "Severe Ponding"
        elif max_depth_m >= 0.10:
            severity = "Moderate Ponding"
        else:
            severity = "Minor Ponding"

        # GeoJSON Feature for sink centroid/pit
        geojson_feat = {
            "type": "Feature",
            "properties": {
                "sink_id": feat_id,
                "area_m2": round(area_m2, 1),
                "volume_m3": round(volume_m3, 1),
                "max_depth_cm": round(max_depth_m * 100.0, 1),
                "severity": severity,
                "pit_elevation_m": None  # Populated downstream if needed
            },
            "geometry": {
                "type": "Point",
                "coordinates": [round(pit_lon, 6), round(pit_lat, 6)]
            }
        }

        sinks.append(WaterloggingSink(
            sink_id=feat_id,
            pit_rc=(pit_r, pit_c),
            pit_coord_native=(pit_x, pit_y),
            pit_coord_wgs84=(pit_lat, pit_lon),
            centroid_native=(cent_x, cent_y),
            centroid_wgs84=(cent_lat, cent_lon),
            area_m2=area_m2,
            volume_m3=volume_m3,
            max_depth_m=max_depth_m,
            max_depth_cm=max_depth_m * 100.0,
            mean_depth_cm=mean_depth_m * 100.0,
            severity=severity,
            geojson_feature=geojson_feat
        ))

    # Rank by trapped water volume descending
    sinks.sort(key=lambda s: s.volume_m3, reverse=True)
    # Re-assign sequential rank IDs (1 = primary critical sink)
    for rank, sink in enumerate(sinks, start=1):
        sink.sink_id = rank
        sink.geojson_feature["properties"]["sink_id"] = rank

    return sinks


def identify_candidate_outlets(
    elevation: np.ndarray,
    metadata: RasterMetadata,
    streams_geojson: Optional[Dict[str, Any]] = None,
    max_outlets: int = 4,
    min_outlet_spacing_m: float = 15.0
) -> List[CandidateOutlet]:
    """
    Identifies lowest boundary perimeter cells and stream exit locations
    that can serve as gravity-fed field outlets.
    """
    nrows, ncols = elevation.shape
    valid_mask = ~np.isnan(elevation) & (elevation > -1000.0)
    transformer = _get_wgs84_transformer(metadata.crs_epsg)

    # 1. Collect perimeter cells
    perimeter_cells: List[Tuple[int, int, float]] = []
    # Row 0 and Row nrows - 1
    for c in range(ncols):
        if valid_mask[0, c]:
            perimeter_cells.append((0, c, float(elevation[0, c])))
        if valid_mask[nrows - 1, c]:
            perimeter_cells.append((nrows - 1, c, float(elevation[nrows - 1, c])))
    # Col 0 and Col ncols - 1
    for r in range(1, nrows - 1):
        if valid_mask[r, 0]:
            perimeter_cells.append((r, 0, float(elevation[r, 0])))
        if valid_mask[r, ncols - 1]:
            perimeter_cells.append((r, ncols - 1, float(elevation[r, ncols - 1])))

    if not perimeter_cells:
        return []

    # Sort perimeter cells by elevation ascending
    perimeter_cells.sort(key=lambda item: item[2])

    selected_outlets: List[CandidateOutlet] = []
    outlet_id = 1

    # Check natural stream outlets first if available
    if streams_geojson and "features" in streams_geojson:
        for feat in streams_geojson["features"]:
            geom = feat.get("geometry", {})
            if geom.get("type") == "LineString":
                coords = geom.get("coordinates", [])
                if coords:
                    # Last coordinate of stream
                    end_lon, end_lat = coords[-1]
                    # Convert to native
                    if transformer:
                        try:
                            rev_trans = pyproj.Transformer.from_crs(4326, metadata.crs_epsg, always_xy=True)
                            end_x, end_y = rev_trans.transform(end_lon, end_lat)
                            aff = Affine(*metadata.transform[:6])
                            inv_aff = ~aff
                            col, row = inv_aff @ (end_x, end_y)
                            r, c = int(round(row)), int(round(col))
                            if 0 <= r < nrows and 0 <= c < ncols and valid_mask[r, c]:
                                elev = float(elevation[r, c])
                                feat_geojson = {
                                    "type": "Feature",
                                    "properties": {
                                        "outlet_id": outlet_id,
                                        "elevation_m": round(elev, 2),
                                        "type": "Natural Stream Exit"
                                    },
                                    "geometry": {
                                        "type": "Point",
                                        "coordinates": [round(end_lon, 6), round(end_lat, 6)]
                                    }
                                }
                                selected_outlets.append(CandidateOutlet(
                                    outlet_id=outlet_id,
                                    rc=(r, c),
                                    coord_native=(end_x, end_y),
                                    coord_wgs84=(end_lat, end_lon),
                                    elevation_m=elev,
                                    outlet_type="stream_exit",
                                    geojson_feature=feat_geojson
                                ))
                                outlet_id += 1
                                if len(selected_outlets) >= 2:
                                    break
                        except Exception:
                            pass

    # Non-maximum suppression over lowest perimeter cells
    res_avg = (metadata.res_x + metadata.res_y) / 2.0
    min_pixel_dist = max(5.0, min_outlet_spacing_m / res_avg)

    for r, c, elev in perimeter_cells:
        if len(selected_outlets) >= max_outlets:
            break

        # Check distance to already selected outlets
        too_close = False
        for out in selected_outlets:
            dr = r - out.rc[0]
            dc = c - out.rc[1]
            dist_px = np.sqrt(dr**2 + dc**2)
            if dist_px < min_pixel_dist:
                too_close = True
                break

        if not too_close:
            x, y = _rc_to_native(metadata.transform, r, c)
            if transformer:
                lon, lat = transformer.transform(x, y)
            else:
                lon, lat = x, y

            feat_geojson = {
                "type": "Feature",
                "properties": {
                    "outlet_id": outlet_id,
                    "elevation_m": round(elev, 2),
                    "type": "Field Boundary Low Point"
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [round(lon, 6), round(lat, 6)]
                }
            }
            selected_outlets.append(CandidateOutlet(
                outlet_id=outlet_id,
                rc=(r, c),
                coord_native=(x, y),
                coord_wgs84=(lat, lon),
                elevation_m=elev,
                outlet_type="perimeter_low_point",
                geojson_feature=feat_geojson
            ))
            outlet_id += 1

    return selected_outlets


def build_cost_surface(
    elevation: np.ndarray,
    streams_mask: Optional[np.ndarray] = None,
    flow_accum: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Constructs an anisotropic terrain friction/cost surface for least-cost path routing.
    - Lowers cost where natural runoff paths or high flow accumulation exist.
    - Penalizes high ground and bund ridges.
    - Sets NoData or boundary void pixels to infinity.
    """
    valid_mask = ~np.isnan(elevation) & (elevation > -1000.0)
    if not np.any(valid_mask):
        return np.ones_like(elevation, dtype=np.float32)

    min_z = float(np.nanmin(elevation[valid_mask]))
    max_z = float(np.nanmax(elevation[valid_mask]))
    range_z = max_z - min_z if (max_z - min_z) > 1e-4 else 1.0

    # Normalized relative elevation (0.0 to 1.0)
    norm_z = np.zeros_like(elevation, dtype=np.float32)
    norm_z[valid_mask] = (elevation[valid_mask] - min_z) / range_z

    # Base cost: quadratic penalty on elevation to favor lower paths
    cost = 1.0 + 8.0 * (norm_z ** 2)

    # Stream alignment bonus: reduce cost by up to 70% along natural channels
    if streams_mask is not None and np.any(streams_mask):
        cost[streams_mask & valid_mask] *= 0.3
    elif flow_accum is not None:
        valid_fa = np.nan_to_num(flow_accum, nan=0.0)
        max_fa = float(np.nanmax(valid_fa))
        if max_fa > 1.0:
            fa_norm = np.clip(np.log1p(valid_fa) / np.log1p(max_fa), 0.0, 1.0)
            cost[valid_mask] *= (1.0 - 0.6 * fa_norm[valid_mask])

    # NoData / Invalid terrain is impassable
    cost[~valid_mask] = np.inf
    cost = np.maximum(cost, 0.05)
    return cost


def analyze_route_engineering(
    path_rc: List[Tuple[int, int]],
    elevation: np.ndarray,
    metadata: RasterMetadata,
    route_id: str,
    sink_id: int,
    outlet_id: int,
    min_bed_slope_pct: float = 0.2,
    ditch_bottom_width_m: float = 0.5,
    ditch_side_slope: float = 1.0
) -> DrainageRoute:
    """
    Analyzes longitudinal cross-section, enforces monotonic gravity-fed bed grade,
    computes excavation cut depth profile, and calculates earthwork volume.
    """
    transformer = _get_wgs84_transformer(metadata.crs_epsg)
    aff = Affine(*metadata.transform[:6])

    # Convert pixel path to native metric coordinates (x, y)
    native_coords = []
    for r, c in path_rc:
        x, y = aff @ (c + 0.5, r + 0.5)
        native_coords.append((float(x), float(y)))

    # Simplify line with tolerance = 0.5 * resolution to smooth staircase artifacts
    tolerance = 0.5 * ((metadata.res_x + metadata.res_y) / 2.0)
    line_geom = shapely.geometry.LineString(native_coords)
    simp_geom = line_geom.simplify(tolerance=tolerance, preserve_topology=True)
    simp_coords = list(simp_geom.coords)

    # Compute cumulative distance along the path
    path_arr = np.array(path_rc)
    diffs = np.diff(path_arr, axis=0)
    # Metric distance per step considering pixel resolution
    step_dx = diffs[:, 1] * metadata.res_x
    step_dy = diffs[:, 0] * metadata.res_y
    step_dists = np.sqrt(step_dx**2 + step_dy**2)
    cum_dist = np.insert(np.cumsum(step_dists), 0, 0.0)
    total_length_m = float(cum_dist[-1])

    # Sample ground elevation along the path
    ground_profile = [float(elevation[r, c]) for r, c in path_rc]
    start_elev = float(ground_profile[0])
    end_elev = float(ground_profile[-1])

    # Overall natural ground slope
    avg_slope_pct = ((start_elev - end_elev) / total_length_m * 100.0) if total_length_m > 0 else 0.0

    # Minimum hydraulic gravity bed grade (e.g. 0.2% = 0.002 m/m)
    s_min = max(0.001, min_bed_slope_pct / 100.0)

    # Ideal uniform bed slope to reach outlet elevation
    natural_bed_slope = (start_elev - end_elev) / total_length_m if total_length_m > 0 else 0.0
    bed_slope = max(s_min, natural_bed_slope)
    bed_slope_pct = float(bed_slope * 100.0)

    # Bed invert profile: starts at pit invert and strictly decreases monotonically
    bed_profile = [float(start_elev - bed_slope * d) for d in cum_dist]
    outlet_bed_elev = float(bed_profile[-1])

    # Cut depth at each station along the route
    ground_arr = np.array(ground_profile, dtype=np.float64)
    bed_arr = np.array(bed_profile, dtype=np.float64)
    cut_arr = np.maximum(0.0, ground_arr - bed_arr)
    max_cut_depth_m = float(np.max(cut_arr))
    avg_cut_depth_m = float(np.mean(cut_arr))

    # Earthwork volume calculation for trapezoidal channel:
    # A(s) = (b + z * cut) * cut
    b = ditch_bottom_width_m
    z = ditch_side_slope
    area_arr = (b + z * cut_arr) * cut_arr
    # Trapezoidal numerical integration over step distances
    vol_m3 = float(np.sum(0.5 * (area_arr[:-1] + area_arr[1:]) * step_dists))

    # Feasibility Rating
    if max_cut_depth_m <= 0.8:
        feasibility_rank = "Recommended (Optimal)"
    elif max_cut_depth_m <= 1.5:
        feasibility_rank = "Moderate Excavation"
    else:
        feasibility_rank = "High Excavation"

    # WGS84 coordinates for Leaflet display
    wgs84_coords = []
    geojson_coords = []
    for x, y in simp_coords:
        if transformer:
            lon, lat = transformer.transform(x, y)
        else:
            lon, lat = x, y
        wgs84_coords.append((float(lat), float(lon)))
        geojson_coords.append([round(lon, 6), round(lat, 6)])

    # GeoJSON Feature
    geojson_feat = {
        "type": "Feature",
        "properties": {
            "route_id": route_id,
            "sink_id": sink_id,
            "outlet_id": outlet_id,
            "length_m": round(total_length_m, 1),
            "start_elev_m": round(start_elev, 2),
            "end_elev_m": round(end_elev, 2),
            "outlet_bed_elev_m": round(outlet_bed_elev, 2),
            "avg_slope_pct": round(avg_slope_pct, 2),
            "bed_slope_pct": round(bed_slope_pct, 2),
            "max_cut_depth_m": round(max_cut_depth_m, 2),
            "avg_cut_depth_m": round(avg_cut_depth_m, 2),
            "earthwork_vol_m3": round(vol_m3, 1),
            "feasibility_rank": feasibility_rank
        },
        "geometry": {
            "type": "LineString",
            "coordinates": geojson_coords
        }
    }

    return DrainageRoute(
        route_id=route_id,
        sink_id=sink_id,
        outlet_id=outlet_id,
        path_rc=path_rc,
        path_coords_native=simp_coords,
        path_coords_wgs84=wgs84_coords,
        length_m=total_length_m,
        start_elev_m=start_elev,
        end_elev_m=end_elev,
        outlet_bed_elev_m=outlet_bed_elev,
        avg_slope_pct=avg_slope_pct,
        bed_slope_pct=bed_slope_pct,
        max_cut_depth_m=max_cut_depth_m,
        avg_cut_depth_m=avg_cut_depth_m,
        earthwork_vol_m3=vol_m3,
        feasibility_rank=feasibility_rank,
        is_recommended=False,  # Updated during multi-route ranking
        distance_profile_m=[float(d) for d in cum_dist],
        ground_elev_profile_m=[float(z) for z in ground_profile],
        bed_elev_profile_m=[float(z) for z in bed_profile],
        cut_depth_profile_m=[float(c) for c in cut_arr],
        geojson_feature=geojson_feat
    )


def generate_drainage_plan(
    dem_source: Union[str, Path],
    hydrology_result: Optional[HydrologyResult] = None,
    output_dir: Union[str, Path] = "data/processed",
    min_waterlogging_depth_m: float = 0.05,
    min_waterlogging_area_m2: float = 10.0,
    min_bed_slope_pct: float = 0.2,
    ditch_bottom_width_m: float = 0.5,
    ditch_side_slope: float = 1.0,
    max_candidate_outlets: int = 4
) -> DrainagePlanResult:
    """
    End-to-end Drainage Recommendation Pipeline.
    1. Loads elevation model & runs hydrology if not supplied.
    2. Identifies and ranks priority waterlogging pockets.
    3. Detects candidate boundary outlets.
    4. Computes least-cost gravity-fed surface trench routes.
    5. Evaluates excavation metrics and designates the Recommended (Optimal) Route.
    """
    elevation, metadata = load_raster(dem_source)

    if hydrology_result is None:
        hydrology_result = run_hydrological_analysis(
            dem_source=dem_source,
            output_dir=output_dir,
            min_waterlogging_depth_m=min_waterlogging_depth_m
        )

    # 1. Identify Priority Sinks
    priority_sinks = identify_priority_sinks(
        depression_depth=hydrology_result.depression_depth_array,
        metadata=metadata,
        min_depth_m=min_waterlogging_depth_m,
        min_area_m2=min_waterlogging_area_m2
    )

    # 2. Identify Candidate Outlets
    candidate_outlets = identify_candidate_outlets(
        elevation=elevation,
        metadata=metadata,
        streams_geojson=hydrology_result.streams_geojson,
        max_outlets=max_candidate_outlets
    )

    if not priority_sinks or not candidate_outlets:
        return DrainagePlanResult(
            routes=[],
            priority_sinks=priority_sinks,
            candidate_outlets=candidate_outlets,
            best_route=None,
            total_ditch_length_m=0.0,
            total_excavation_vol_m3=0.0,
            routes_geojson={"type": "FeatureCollection", "features": []},
            sinks_geojson={"type": "FeatureCollection", "features": [s.geojson_feature for s in priority_sinks]},
            outlets_geojson={"type": "FeatureCollection", "features": [o.geojson_feature for o in candidate_outlets]}
        )

    # 3. Build Anisotropic Cost Surface
    cost_surface = build_cost_surface(
        elevation=elevation,
        flow_accum=hydrology_result.flow_accum_array
    )

    # 4. Generate candidate routes from the primary sink to each candidate outlet
    primary_sink = priority_sinks[0]
    candidate_routes: List[DrainageRoute] = []

    for route_idx, outlet in enumerate(candidate_outlets, start=1):
        try:
            indices, _ = skimage.graph.route_through_array(
                cost_surface,
                primary_sink.pit_rc,
                outlet.rc,
                fully_connected=True,
                geometric=True
            )
            route = analyze_route_engineering(
                path_rc=indices,
                elevation=elevation,
                metadata=metadata,
                route_id=f"Route-{route_idx}",
                sink_id=primary_sink.sink_id,
                outlet_id=outlet.outlet_id,
                min_bed_slope_pct=min_bed_slope_pct,
                ditch_bottom_width_m=ditch_bottom_width_m,
                ditch_side_slope=ditch_side_slope
            )
            candidate_routes.append(route)
        except Exception:
            continue

    if not candidate_routes:
        return DrainagePlanResult(
            routes=[],
            priority_sinks=priority_sinks,
            candidate_outlets=candidate_outlets,
            best_route=None,
            total_ditch_length_m=0.0,
            total_excavation_vol_m3=0.0,
            routes_geojson={"type": "FeatureCollection", "features": []},
            sinks_geojson={"type": "FeatureCollection", "features": [s.geojson_feature for s in priority_sinks]},
            outlets_geojson={"type": "FeatureCollection", "features": [o.geojson_feature for o in candidate_outlets]}
        )

    # 5. Rank routes: prefer minimal excavation volume, then minimal max cut depth
    candidate_routes.sort(key=lambda r: (r.earthwork_vol_m3, r.max_cut_depth_m, r.length_m))
    candidate_routes[0].is_recommended = True
    candidate_routes[0].geojson_feature["properties"]["is_recommended"] = True

    for r in candidate_routes[1:]:
        r.is_recommended = False
        r.geojson_feature["properties"]["is_recommended"] = False

    best_route = candidate_routes[0]
    total_ditch_length_m = sum(r.length_m for r in candidate_routes)
    total_excavation_vol_m3 = sum(r.earthwork_vol_m3 for r in candidate_routes)

    routes_geojson = {
        "type": "FeatureCollection",
        "features": [r.geojson_feature for r in candidate_routes]
    }
    sinks_geojson = {
        "type": "FeatureCollection",
        "features": [s.geojson_feature for s in priority_sinks]
    }
    outlets_geojson = {
        "type": "FeatureCollection",
        "features": [o.geojson_feature for o in candidate_outlets]
    }

    return DrainagePlanResult(
        routes=candidate_routes,
        priority_sinks=priority_sinks,
        candidate_outlets=candidate_outlets,
        best_route=best_route,
        total_ditch_length_m=total_ditch_length_m,
        total_excavation_vol_m3=total_excavation_vol_m3,
        routes_geojson=routes_geojson,
        sinks_geojson=sinks_geojson,
        outlets_geojson=outlets_geojson
    )
