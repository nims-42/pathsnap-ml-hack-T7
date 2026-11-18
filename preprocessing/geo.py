"""Geometry helpers. Everything is done in metres (UTM 40N), never in raw degrees."""
import numpy as np
from pyproj import Transformer

from config import CRS_UTM

_to_utm = Transformer.from_crs("EPSG:4326", CRS_UTM, always_xy=True)


def project(lon, lat):
    """lon/lat (degrees, longitude FIRST) -> x/y in metres."""
    x, y = _to_utm.transform(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
    return np.asarray(x), np.asarray(y)


def point_to_segment(px, py, ax, ay, bx, by):
    """Distance from points to segments A->B, plus the projection fraction in [0, 1].
    All arguments broadcast, so this works for one point vs many arcs or pair-by-pair."""
    dx, dy = bx - ax, by - ay
    len2 = dx * dx + dy * dy
    safe = np.where(len2 > 0, len2, 1.0)
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / safe, 0.0, 1.0)
    t = np.where(len2 > 0, t, 0.0)
    cx, cy = ax + t * dx, ay + t * dy
    return np.hypot(px - cx, py - cy), t


def bearing(dx, dy):
    """Compass-style bearing in degrees from an x/y displacement (0 = north)."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def angle_diff(a, b):
    """Smallest absolute difference between two bearings, 0..180."""
    return np.abs((a - b + 180.0) % 360.0 - 180.0)
