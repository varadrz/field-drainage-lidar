"""
Hydrological Processing Engine
Decoupled GIS hydrological computational module powered by WhiteboxTools Open Core.
Implements depression filling, depression depth calculation, D8 flow direction,
D8 flow accumulation, natural stream extraction & vectorization, and sub-basin delineation.
Strictly preserved metric CRS and zero UI dependencies.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import json
import os
import numpy as np
import rasterio
import rasterio.features
import shapely.geometry
import geopandas as gpd
import whitebox

from .data_loader import RasterMetadata, load_raster


@dataclass
class HydrologyResult:
    """Standardized output container for WhiteboxTools hydrological analysis."""
    filled_dem_path: str
    d8_pointer_path: str
    flow_accum_path: str
    streams_path: str
    streams_shp_path: str
    basins_path: str
    depression_depth_path: str
    total_stream_length_m: float
    max_waterlogging_depth_m: float
    max_waterlogging_depth_cm: float
    total_waterlogging_volume_m3: float
    waterlogging_area_sq_m: float
    waterlogging_area_acres: float
    waterlogging_area_pct: float
    sub_basin_count: int
    stream_segments_count: int
    streams_geojson: Dict[str, Any]
    basins_geojson: Dict[str, Any]
    flow_accum_array: np.ndarray
    depression_depth_array: np.ndarray
    basins_array: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a farmer-friendly serializable dictionary."""
        return {
            "Total Natural Runoff Channel Length": f"{self.total_stream_length_m:,.1f} m",
            "Identified Stream Segments": self.stream_segments_count,
            "Deepest Waterlogging Point": f"{self.max_waterlogging_depth_cm:.1f} cm ({self.max_waterlogging_depth_m:.2f} m)",
            "Total Waterlogging Area": f"{self.waterlogging_area_acres:.2f} ac ({self.waterlogging_area_sq_m:,.0f} m²)",
            "Waterlogging Field Coverage": f"{self.waterlogging_area_pct:.1f}%",
            "Estimated Trapped Ponding Volume": f"{self.total_waterlogging_volume_m3:,.0f} m³",
            "Field Drainage Sub-Catchments": self.sub_basin_count,
        }


def run_hydrological_analysis(
    dem_source: Union[str, Path],
    output_dir: Union[str, Path] = "data/processed",
    flow_accum_threshold: float = 80.0,
    min_waterlogging_depth_m: float = 0.05,
    metadata: Optional[RasterMetadata] = None
) -> HydrologyResult:
    """
    Executes end-to-end WhiteboxTools hydrological modeling on a metric DEM raster.
    
    Args:
        dem_source: Path to bare-earth elevation GeoTIFF (.tif).
        output_dir: Directory where processed raster & vector layers are written.
        flow_accum_threshold: Cell count threshold for stream extraction (lower = finer trickles).
        min_waterlogging_depth_m: Minimum sink depth to filter sensor noise from real puddles.
        metadata: Optional RasterMetadata object (loaded automatically if omitted).
        
    Returns:
        HydrologyResult dataclass holding output paths, metrics, arrays, and GeoJSON vectors.
        
    Raises:
        FileNotFoundError: If dem_source does not exist.
        RuntimeError: If WhiteboxTools execution fails.
    """
    dem_path = Path(dem_source).resolve()
    if not dem_path.exists():
        raise FileNotFoundError(f"Input DEM raster not found at: {dem_path}")

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize WhiteboxTools
    wbt = whitebox.WhiteboxTools()
    wbt.set_verbose_mode(False)

    # File paths for intermediate & deliverable layers (absolute paths for Windows)
    filled_dem_path = str(out_dir / "dem_filled.tif")
    pointer_path = str(out_dir / "d8_pointer.tif")
    flow_accum_path = str(out_dir / "flow_accum.tif")
    streams_path = str(out_dir / "streams.tif")
    stream_shp_path = str(out_dir / "streams.shp")
    basins_path = str(out_dir / "basins.tif")
    dep_depth_path = str(out_dir / "depression_depth.tif")

    # Step 1: Depression Filling (Hydrologic Conditioning)
    ret = wbt.fill_depressions(str(dem_path), filled_dem_path, fix_flats=True)
    if ret != 0 or not Path(filled_dem_path).exists():
        raise RuntimeError("WhiteboxTools FillDepressions execution failed.")

    # Step 2: Calculate True Waterlogging Depth (Filled DEM - Raw DEM)
    with rasterio.open(dem_path) as raw_ds, rasterio.open(filled_dem_path) as filled_ds:
        raw_dem = raw_ds.read(1).astype(np.float32)
        filled_dem = filled_ds.read(1).astype(np.float32)
        raw_meta = raw_ds.meta.copy()
        crs = raw_ds.crs
        transform = raw_ds.transform
        res_x = abs(transform.a)
        res_y = abs(transform.e)

    # Handle NoData in raw DEM
    valid_mask = ~np.isnan(raw_dem) & (raw_dem > -1000.0)
    dep_depth = np.zeros_like(raw_dem, dtype=np.float32)
    dep_depth[valid_mask] = filled_dem[valid_mask] - raw_dem[valid_mask]
    dep_depth[dep_depth < min_waterlogging_depth_m] = 0.0
    dep_depth[~valid_mask] = np.nan

    # Write Depression Depth GeoTIFF
    raw_meta.update(dtype=rasterio.float32, nodata=-9999.0)
    with rasterio.open(dep_depth_path, "w", **raw_meta) as dst:
        out_write = np.copy(dep_depth)
        out_write[np.isnan(out_write)] = -9999.0
        dst.write(out_write, 1)

    # Depression volume and area statistics
    pixel_area_m2 = res_x * res_y
    sink_mask = dep_depth >= min_waterlogging_depth_m
    sink_pixels = int(np.sum(sink_mask))
    total_valid_pixels = int(np.sum(valid_mask))

    waterlogging_area_sq_m = float(sink_pixels * pixel_area_m2)
    waterlogging_area_acres = waterlogging_area_sq_m / 4046.8564224
    total_field_area_m2 = total_valid_pixels * pixel_area_m2
    waterlogging_area_pct = (waterlogging_area_sq_m / total_field_area_m2 * 100.0) if total_field_area_m2 > 0 else 0.0

    max_waterlogging_depth_m = float(np.nanmax(dep_depth)) if sink_pixels > 0 else 0.0
    max_waterlogging_depth_cm = max_waterlogging_depth_m * 100.0
    total_waterlogging_volume_m3 = float(np.sum(dep_depth[sink_mask]) * pixel_area_m2) if sink_pixels > 0 else 0.0

    # Step 3: D8 Flow Direction
    wbt.d8_pointer(filled_dem_path, pointer_path, esri_pntr=False)

    # Step 4: D8 Flow Accumulation (Upslope Contributing Cells)
    wbt.d8_flow_accumulation(filled_dem_path, flow_accum_path, out_type="cells")

    # Step 5: Natural Drainage / Stream Extraction
    wbt.extract_streams(flow_accum_path, streams_path, threshold=float(flow_accum_threshold), zero_background=True)

    # Step 6: Vectorize Streams to Shapefile & GeoJSON
    wbt.raster_streams_to_vector(streams_path, pointer_path, stream_shp_path, esri_pntr=False)

    streams_geojson = {"type": "FeatureCollection", "features": []}
    total_stream_length_m = 0.0
    stream_segments_count = 0

    if Path(stream_shp_path).exists():
        try:
            stream_gdf = gpd.read_file(stream_shp_path)
            if not stream_gdf.empty:
                stream_gdf = stream_gdf[stream_gdf.geometry.notnull() & ~stream_gdf.geometry.is_empty]
                if not stream_gdf.empty:
                    # WhiteboxTools does not write a .prj file for streams.shp; assign native raster CRS
                    if stream_gdf.crs is None and crs is not None:
                        stream_gdf = stream_gdf.set_crs(crs)
                    
                    # Metric length in native coordinates
                    total_stream_length_m = float(stream_gdf.geometry.length.sum())
                    stream_segments_count = len(stream_gdf)
                    
                    # Convert to WGS84 for Leaflet map display
                    if stream_gdf.crs is not None:
                        stream_gdf_wgs84 = stream_gdf.to_crs("EPSG:4326")
                        streams_geojson = json.loads(stream_gdf_wgs84.to_json())
                    else:
                        streams_geojson = json.loads(stream_gdf.to_json())
        except Exception:
            pass

    # Step 7: Sub-Basin / Catchment Delineation
    wbt.basins(pointer_path, basins_path, esri_pntr=False)

    # Polygonize major sub-basins into GeoJSON
    basins_geojson = {"type": "FeatureCollection", "features": []}
    sub_basin_count = 0
    basins_array = np.zeros_like(raw_dem, dtype=np.int32)

    if Path(basins_path).exists():
        with rasterio.open(basins_path) as b_ds:
            basins_array = b_ds.read(1).astype(np.int32)
            valid_basin_mask = (basins_array > 0) & valid_mask
            sub_basin_count = len(np.unique(basins_array[valid_basin_mask]))

            try:
                shapes_gen = rasterio.features.shapes(
                    basins_array,
                    mask=valid_basin_mask,
                    transform=b_ds.transform
                )
                features_list = []
                for geom, val in shapes_gen:
                    poly = shapely.geometry.shape(geom)
                    # Keep basins with non-trivial area (> 50 m²)
                    if poly.area > 50.0:
                        features_list.append({"basin_id": int(val), "area_sq_m": round(poly.area, 1), "geometry": poly})

                if features_list:
                    basin_gdf = gpd.GeoDataFrame(features_list, crs=b_ds.crs)
                    basin_gdf_wgs84 = basin_gdf.to_crs("EPSG:4326")
                    basins_geojson = json.loads(basin_gdf_wgs84.to_json())
            except Exception:
                pass

    # Read flow accumulation array for raster overlay visualization
    with rasterio.open(flow_accum_path) as f_ds:
        flow_accum_array = f_ds.read(1).astype(np.float32)
        flow_accum_array[~valid_mask] = np.nan

    return HydrologyResult(
        filled_dem_path=filled_dem_path,
        d8_pointer_path=pointer_path,
        flow_accum_path=flow_accum_path,
        streams_path=streams_path,
        streams_shp_path=stream_shp_path,
        basins_path=basins_path,
        depression_depth_path=dep_depth_path,
        total_stream_length_m=total_stream_length_m,
        max_waterlogging_depth_m=max_waterlogging_depth_m,
        max_waterlogging_depth_cm=max_waterlogging_depth_cm,
        total_waterlogging_volume_m3=total_waterlogging_volume_m3,
        waterlogging_area_sq_m=waterlogging_area_sq_m,
        waterlogging_area_acres=waterlogging_area_acres,
        waterlogging_area_pct=waterlogging_area_pct,
        sub_basin_count=sub_basin_count,
        stream_segments_count=stream_segments_count,
        streams_geojson=streams_geojson,
        basins_geojson=basins_geojson,
        flow_accum_array=flow_accum_array,
        depression_depth_array=dep_depth,
        basins_array=basins_array
    )
