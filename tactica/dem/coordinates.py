"""
Coordinate system transformations for DEMs.

This module provides tools for converting between grid coordinates
(row, column) and world coordinates (x, y in CRS units).
"""

from typing import Tuple, Optional, Union, List
import numpy as np

from tactica.dem.loader import DEMMetadata


def grid_to_world(
    row: Union[float, np.ndarray],
    col: Union[float, np.ndarray],
    metadata: DEMMetadata,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Convert grid coordinates (row, col) to world coordinates (x, y).

    Args:
        row: Row index (or array of row indices)
        col: Column index (or array of column indices)
        metadata: DEM metadata with bounds and resolution

    Returns:
        x: X coordinate(s) in CRS units
        y: Y coordinate(s) in CRS units

    Example:
        >>> meta = DEMMetadata(bounds=(0, 0, 100, 100), resolution=(1.0, 1.0))
        >>> x, y = grid_to_world(50, 25, meta)
        >>> print(f"World: ({x}, {y})")
    """
    if metadata.bounds is None:
        raise ValueError("DEMMetadata.bounds must be set for coordinate transforms")

    min_x, min_y, max_x, max_y = metadata.bounds
    x_res, y_res = metadata.resolution

    # Convert to world coordinates
    # Note: row 0 is at max_y (top of image)
    x = min_x + (col + 0.5) * x_res
    y = max_y - (row + 0.5) * y_res

    return x, y


def world_to_grid(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    metadata: DEMMetadata,
    clamp: bool = False,
    dem_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Convert world coordinates (x, y) to grid coordinates (row, col).

    Args:
        x: X coordinate(s) in CRS units
        y: Y coordinate(s) in CRS units
        metadata: DEM metadata with bounds and resolution
        clamp: If True, clamp coordinates to valid grid range
        dem_shape: Required if clamp=True, (height, width) of DEM

    Returns:
        row: Row index (or array of row indices)
        col: Column index (or array of column indices)

    Example:
        >>> meta = DEMMetadata(bounds=(0, 0, 100, 100), resolution=(1.0, 1.0))
        >>> row, col = world_to_grid(25.5, 75.5, meta)
        >>> print(f"Grid: ({row}, {col})")
    """
    if metadata.bounds is None:
        raise ValueError("DEMMetadata.bounds must be set for coordinate transforms")

    min_x, min_y, max_x, max_y = metadata.bounds
    x_res, y_res = metadata.resolution

    # Convert to grid coordinates
    col = (x - min_x) / x_res - 0.5
    row = (max_y - y) / y_res - 0.5

    if clamp:
        if dem_shape is None:
            raise ValueError("dem_shape required when clamp=True")
        height, width = dem_shape
        row = np.clip(row, 0, height - 1)
        col = np.clip(col, 0, width - 1)

    return row, col


def latlon_to_utm(
    lat: Union[float, np.ndarray],
    lon: Union[float, np.ndarray],
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], str]:
    """
    Convert latitude/longitude to UTM coordinates.

    Automatically determines the appropriate UTM zone.

    Args:
        lat: Latitude(s) in degrees
        lon: Longitude(s) in degrees

    Returns:
        easting: UTM easting(s)
        northing: UTM northing(s)
        utm_crs: UTM CRS string (e.g., "EPSG:32610")

    Note:
        Requires pyproj package. Install with: pip install pyproj
    """
    try:
        from pyproj import CRS, Transformer
    except ImportError:
        raise ImportError(
            "pyproj is required for coordinate transformations. "
            "Install with: pip install pyproj"
        )

    # Determine UTM zone
    if isinstance(lon, np.ndarray):
        center_lon = np.mean(lon)
        center_lat = np.mean(lat)
    else:
        center_lon = lon
        center_lat = lat

    zone = int((center_lon + 180) / 6) + 1
    hemisphere = 'N' if center_lat >= 0 else 'S'

    # Construct UTM CRS
    epsg_code = 32600 + zone if hemisphere == 'N' else 32700 + zone
    utm_crs = f"EPSG:{epsg_code}"

    # Transform
    wgs84 = CRS.from_epsg(4326)
    utm = CRS.from_epsg(epsg_code)
    transformer = Transformer.from_crs(wgs84, utm, always_xy=True)

    easting, northing = transformer.transform(lon, lat)

    return easting, northing, utm_crs


def utm_to_latlon(
    easting: Union[float, np.ndarray],
    northing: Union[float, np.ndarray],
    utm_crs: str,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Convert UTM coordinates to latitude/longitude.

    Args:
        easting: UTM easting(s)
        northing: UTM northing(s)
        utm_crs: UTM CRS string (e.g., "EPSG:32610")

    Returns:
        lat: Latitude(s) in degrees
        lon: Longitude(s) in degrees

    Note:
        Requires pyproj package.
    """
    try:
        from pyproj import CRS, Transformer
    except ImportError:
        raise ImportError(
            "pyproj is required for coordinate transformations. "
            "Install with: pip install pyproj"
        )

    wgs84 = CRS.from_epsg(4326)
    utm = CRS.from_string(utm_crs)
    transformer = Transformer.from_crs(utm, wgs84, always_xy=True)

    lon, lat = transformer.transform(easting, northing)

    return lat, lon


def compute_grid_bounds(
    dem_shape: Tuple[int, int],
    metadata: DEMMetadata,
) -> Tuple[float, float, float, float]:
    """
    Compute world bounds for a DEM given its shape and metadata.

    Args:
        dem_shape: (height, width) of DEM
        metadata: DEM metadata

    Returns:
        (min_x, min_y, max_x, max_y) in CRS units
    """
    if metadata.bounds is not None:
        return metadata.bounds

    height, width = dem_shape
    x_res, y_res = metadata.resolution

    return (0, 0, width * x_res, height * y_res)


def cameras_to_geojson(
    cameras: List,
    metadata: DEMMetadata,
    include_fov: bool = True,
) -> dict:
    """
    Convert camera list to GeoJSON for visualization.

    Args:
        cameras: List of Camera objects
        metadata: DEM metadata for coordinate transforms
        include_fov: Whether to include FOV polygons

    Returns:
        GeoJSON FeatureCollection dict
    """
    features = []

    for i, cam in enumerate(cameras):
        # Convert camera position to world coordinates
        x, y = grid_to_world(cam.y, cam.x, metadata)

        # Point feature for camera location
        point_feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(x), float(y)],
            },
            "properties": {
                "camera_id": i,
                "z": float(cam.z),
                "yaw_degrees": float(np.rad2deg(cam.yaw)),
                "pitch_degrees": float(np.rad2deg(cam.pitch)),
                "hfov_degrees": float(np.rad2deg(cam.hfov)),
                "vfov_degrees": float(np.rad2deg(cam.vfov)),
                "max_range": float(cam.max_range),
            },
        }
        features.append(point_feature)

        # FOV polygon (simplified as a triangle in 2D)
        if include_fov:
            # Calculate FOV triangle vertices
            half_hfov = cam.hfov / 2
            fov_range = cam.max_range * metadata.resolution[0]  # Convert to world units

            # Direction vectors for left and right edges of FOV
            left_angle = cam.yaw + half_hfov
            right_angle = cam.yaw - half_hfov

            left_x = x + fov_range * np.cos(left_angle)
            left_y = y + fov_range * np.sin(left_angle)
            right_x = x + fov_range * np.cos(right_angle)
            right_y = y + fov_range * np.sin(right_angle)

            fov_feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [float(x), float(y)],
                        [float(left_x), float(left_y)],
                        [float(right_x), float(right_y)],
                        [float(x), float(y)],
                    ]],
                },
                "properties": {
                    "camera_id": i,
                    "type": "fov",
                },
            }
            features.append(fov_feature)

    return {
        "type": "FeatureCollection",
        "features": features,
    }
