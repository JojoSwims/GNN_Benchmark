"""Geographic utility functions."""

from math import asin, cos, radians, sin, sqrt


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great-circle distance between two coordinates.

    Uses the haversine formula with the mean Earth radius.

    Args:
        lat1: Latitude of first point (degrees).
        lon1: Longitude of first point (degrees).
        lat2: Latitude of second point (degrees).
        lon2: Longitude of second point (degrees).

    Returns:
        Distance in meters.
    """
    R = 6371008.8  # Mean Earth radius in meters

    phi1, lambda1, phi2, lambda2 = map(radians, (lat1, lon1, lat2, lon2))

    dphi = phi2 - phi1
    dlambda = lambda2 - lambda1
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    c = 2 * asin(sqrt(a))

    return R * c
