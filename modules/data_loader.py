"""
Data Ingestion and Raster Validation Module
Pure Python / Rasterio logic completely decoupled from UI.
Handles GeoTIFF loading, CRS validation, NoData replacement, and coordinate transformations.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union, BinaryIO
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.warp import transform_bounds
import pyproj


@dataclass
class RasterMetadata:
    """Standardized metadata container for loaded raster elevation models."""
    width: int
    height: int
    count: int
    crs_wkt: str
    crs_epsg: Optional[int]
    is_projected: bool
    is_geographic: bool
    units: str
    res_x: float
    res_y: float
    bounds_native: Tuple[float, float, float, float]  # (minx, miny, maxx, maxy)
    bounds_wgs84: Tuple[float, float, float, float]   # (min_lon, min_lat, max_lon, max_lat)
    nodata_value: Optional[float]
    transform: Tuple[float, ...]


def load_raster(
    source: Union[str, Path, BinaryIO, bytes]
) -> Tuple[np.ndarray, RasterMetadata]:
    """
    Loads and validates a GeoTIFF elevation raster from a file path or in-memory byte buffer.
    
    Args:
        source: File path (str/Path) or file-like object / raw bytes (e.g. Streamlit UploadedFile)
        
    Returns:
        Tuple containing:
            - elevation_array (np.ndarray): 2D float32 array with NoData values converted to np.nan
            - metadata (RasterMetadata): Detailed coordinate and dimension metadata
            
    Raises:
        ValueError: If file is not a valid single/multi-band elevation raster.
    """
    if isinstance(source, bytes):
        memfile = MemoryFile(source)
        dataset = memfile.open()
    elif hasattr(source, "read"):
        content = source.read()
        memfile = MemoryFile(content)
        dataset = memfile.open()
    else:
        dataset = rasterio.open(source)

    try:
        if dataset.count < 1:
            raise ValueError("Input raster contains no data bands.")

        # Read first band as float32
        band_raw = dataset.read(1).astype(np.float32)
        nodata = dataset.nodata

        # Mask NoData values with np.nan
        elevation = np.copy(band_raw)
        if nodata is not None:
            # Handle float comparison tolerance
            if np.isnan(nodata):
                elevation[np.isnan(elevation)] = np.nan
            else:
                elevation[np.isclose(elevation, nodata, atol=1e-3)] = np.nan

        # Also mask extreme negative outliers (e.g., -32768, -999999)
        elevation[elevation < -1000.0] = np.nan

        # Inspect Coordinate Reference System (CRS)
        crs = dataset.crs
        crs_epsg = crs.to_epsg() if crs else None
        crs_wkt = crs.to_wkt() if crs else "Unknown CRS"
        
        is_projected = False
        is_geographic = False
        units = "unknown"

        if crs:
            try:
                pyproj_crs = pyproj.CRS.from_user_input(crs)
                is_projected = pyproj_crs.is_projected
                is_geographic = pyproj_crs.is_geographic
                units = pyproj_crs.axis_info[0].unit_name if pyproj_crs.axis_info else "unknown"
            except Exception:
                is_projected = crs.is_projected
                is_geographic = crs.is_geographic
                units = "meters" if is_projected else "degrees"

        bounds = dataset.bounds
        native_bounds = (bounds.left, bounds.bottom, bounds.right, bounds.top)

        # Compute WGS84 (EPSG:4326) bounding box for Leaflet mapping
        if crs and not is_geographic:
            try:
                wgs84_bounds = transform_bounds(crs, "EPSG:4326", *native_bounds)
            except Exception:
                wgs84_bounds = native_bounds
        else:
            wgs84_bounds = native_bounds

        res_x = abs(dataset.transform.a)
        res_y = abs(dataset.transform.e)

        meta = RasterMetadata(
            width=dataset.width,
            height=dataset.height,
            count=dataset.count,
            crs_wkt=crs_wkt,
            crs_epsg=crs_epsg,
            is_projected=is_projected,
            is_geographic=is_geographic,
            units=units,
            res_x=res_x,
            res_y=res_y,
            bounds_native=native_bounds,
            bounds_wgs84=wgs84_bounds,
            nodata_value=nodata,
            transform=tuple(dataset.transform)
        )

        return elevation, meta

    finally:
        dataset.close()
