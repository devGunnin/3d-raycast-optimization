"""
DEM loading from various file formats and sources.

This module provides tools for loading real elevation data from files
and converting it to the format expected by the visibility engine.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np


@dataclass
class DEMMetadata:
    """
    Metadata about a loaded DEM.

    Contains information about the coordinate reference system, bounds,
    resolution, and units needed to convert between grid and world coordinates.

    Attributes:
        crs: Coordinate Reference System (e.g., "EPSG:4326", "EPSG:32610")
        bounds: (min_x, min_y, max_x, max_y) in CRS units
        resolution: (x_res, y_res) cell size in CRS units (typically meters)
        nodata_value: Value used for missing data cells
        units: Height units (e.g., "meters", "feet")
        source_file: Original file path if loaded from file
    """
    crs: Optional[str] = None
    bounds: Optional[Tuple[float, float, float, float]] = None  # (min_x, min_y, max_x, max_y)
    resolution: Tuple[float, float] = (1.0, 1.0)  # (x_res, y_res)
    nodata_value: Optional[float] = None
    units: str = "meters"
    source_file: Optional[str] = None

    @property
    def cell_size(self) -> float:
        """Average cell size (assumes roughly square cells)."""
        return (abs(self.resolution[0]) + abs(self.resolution[1])) / 2

    @property
    def width_meters(self) -> Optional[float]:
        """Width of DEM in meters (if bounds available)."""
        if self.bounds is None:
            return None
        return self.bounds[2] - self.bounds[0]

    @property
    def height_meters(self) -> Optional[float]:
        """Height of DEM in meters (if bounds available)."""
        if self.bounds is None:
            return None
        return self.bounds[3] - self.bounds[1]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "crs": self.crs,
            "bounds": list(self.bounds) if self.bounds else None,
            "resolution": list(self.resolution),
            "nodata_value": self.nodata_value,
            "units": self.units,
            "source_file": self.source_file,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DEMMetadata":
        """Create from dictionary."""
        return cls(
            crs=data.get("crs"),
            bounds=tuple(data["bounds"]) if data.get("bounds") else None,
            resolution=tuple(data.get("resolution", [1.0, 1.0])),
            nodata_value=data.get("nodata_value"),
            units=data.get("units", "meters"),
            source_file=data.get("source_file"),
        )


def load_dem(
    source: Union[str, Path],
    target_crs: Optional[str] = None,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    resolution: Optional[float] = None,
    fill_nodata: bool = True,
    fill_value: float = 0.0,
) -> Tuple[np.ndarray, DEMMetadata]:
    """
    Load a DEM from a file.

    Supports GeoTIFF (.tif, .tiff) and other raster formats supported by rasterio.
    Can optionally reproject to a different CRS and resample to a different resolution.

    Args:
        source: Path to DEM file (GeoTIFF or other rasterio-supported format)
        target_crs: Optional target CRS to reproject to (e.g., "EPSG:32610" for UTM)
        bounds: Optional (min_x, min_y, max_x, max_y) to crop to
        resolution: Optional target resolution in CRS units (meters)
        fill_nodata: Whether to fill nodata values
        fill_value: Value to use for nodata cells if fill_nodata=True

    Returns:
        dem: float32 array of elevation values
        metadata: DEMMetadata with CRS, bounds, and resolution info

    Raises:
        ImportError: If rasterio is not installed
        FileNotFoundError: If source file doesn't exist
        ValueError: If file cannot be read as a DEM

    Example:
        >>> dem, meta = load_dem("terrain.tif")
        >>> print(f"Shape: {dem.shape}, CRS: {meta.crs}")

    Note:
        Requires rasterio package. Install with: pip install rasterio
    """
    try:
        import rasterio
        from rasterio.warp import calculate_default_transform, reproject, Resampling
    except ImportError:
        raise ImportError(
            "rasterio is required for loading GeoTIFF files. "
            "Install with: pip install rasterio"
        )

    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"DEM file not found: {source}")

    with rasterio.open(source) as src:
        # Read the first band
        dem = src.read(1).astype(np.float32)
        src_crs = str(src.crs) if src.crs else None
        src_bounds = src.bounds
        src_transform = src.transform
        src_nodata = src.nodata

        # Get resolution from transform
        x_res = src_transform.a
        y_res = -src_transform.e  # Typically negative in GeoTIFFs

        metadata = DEMMetadata(
            crs=src_crs,
            bounds=(src_bounds.left, src_bounds.bottom, src_bounds.right, src_bounds.top),
            resolution=(x_res, y_res),
            nodata_value=src_nodata,
            units="meters",
            source_file=str(source),
        )

        # Handle reprojection if requested
        if target_crs and src_crs and target_crs != src_crs:
            # Calculate transform for new CRS
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds,
                resolution=resolution
            )

            # Create destination array
            dst_dem = np.empty((dst_height, dst_width), dtype=np.float32)

            # Reproject
            reproject(
                source=dem,
                destination=dst_dem,
                src_transform=src_transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear,
            )

            dem = dst_dem
            metadata.crs = target_crs
            metadata.resolution = (dst_transform.a, -dst_transform.e)

            # Update bounds
            from rasterio.transform import array_bounds
            metadata.bounds = array_bounds(dst_height, dst_width, dst_transform)

    # Fill nodata values
    if fill_nodata and metadata.nodata_value is not None:
        nodata_mask = dem == metadata.nodata_value
        dem[nodata_mask] = fill_value

    # Ensure no NaN or Inf values
    dem = np.nan_to_num(dem, nan=fill_value, posinf=fill_value, neginf=fill_value)

    return dem, metadata


def load_dem_from_array(
    array: np.ndarray,
    resolution: float = 1.0,
    origin: Tuple[float, float] = (0.0, 0.0),
    crs: Optional[str] = None,
    units: str = "meters",
) -> Tuple[np.ndarray, DEMMetadata]:
    """
    Create a DEM from an existing numpy array with metadata.

    Useful for wrapping synthetic DEMs or arrays from other sources
    with proper metadata for coordinate transformations.

    Args:
        array: 2D array of elevation values
        resolution: Cell size in units
        origin: (x, y) coordinate of top-left corner
        crs: Optional coordinate reference system
        units: Height units (e.g., "meters")

    Returns:
        dem: float32 array (same as input, converted to float32)
        metadata: DEMMetadata with specified parameters
    """
    dem = array.astype(np.float32)
    height, width = dem.shape

    metadata = DEMMetadata(
        crs=crs,
        bounds=(
            origin[0],
            origin[1],
            origin[0] + width * resolution,
            origin[1] + height * resolution,
        ),
        resolution=(resolution, resolution),
        nodata_value=None,
        units=units,
        source_file=None,
    )

    return dem, metadata


def save_dem(
    dem: np.ndarray,
    path: Union[str, Path],
    metadata: Optional[DEMMetadata] = None,
    compress: bool = True,
) -> None:
    """
    Save a DEM to a GeoTIFF file.

    Args:
        dem: 2D array of elevation values
        path: Output file path
        metadata: Optional metadata (CRS, bounds, resolution)
        compress: Whether to compress the output file

    Note:
        Requires rasterio package.
    """
    try:
        import rasterio
        from rasterio.transform import from_bounds
    except ImportError:
        raise ImportError(
            "rasterio is required for saving GeoTIFF files. "
            "Install with: pip install rasterio"
        )

    path = Path(path)
    height, width = dem.shape

    # Build transform from metadata
    if metadata and metadata.bounds:
        transform = from_bounds(
            metadata.bounds[0], metadata.bounds[1],
            metadata.bounds[2], metadata.bounds[3],
            width, height
        )
        crs = metadata.crs
    else:
        transform = rasterio.transform.from_origin(0, height, 1, 1)
        crs = None

    # Write file
    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'width': width,
        'height': height,
        'count': 1,
        'crs': crs,
        'transform': transform,
    }

    if compress:
        profile['compress'] = 'lzw'

    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(dem.astype(np.float32), 1)
