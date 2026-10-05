"""Tile-polygon geometry helpers.

Polygon tests run on the gnomonic (tangent-plane) projection about the tile
centre, where the tile edges are straight lines. Working in raw RA/Dec
instead breaks for tiles straddling RA = 0/360 and distorts tiles near the
poles (including the polar-cap tile, whose corners circle the pole).
"""
from __future__ import annotations

from typing import Any

import numpy as np
from astropy.table import Table
from matplotlib.path import Path


def tile_corners(tile_info: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return the tile's ``(ra, dec)`` corners [deg] as stored.

    ``tile_info`` is a single-row :class:`~astropy.table.Table` or a
    dict-like with keys ``ra1/dec1 ... ra4/dec4``.
    """
    def _get(key: str) -> float:
        if isinstance(tile_info, Table):
            val = tile_info[key]
            return float(val[0] if hasattr(val, "__len__") and len(val) else val)
        return float(tile_info[key])

    ra = np.array([_get(f"ra{i}") for i in (1, 2, 3, 4)], dtype=float)
    dec = np.array([_get(f"dec{i}") for i in (1, 2, 3, 4)], dtype=float)
    return ra, dec


def tile_center(ra_corners, dec_corners) -> tuple[float, float]:
    """Centre ``(ra, dec)`` [deg] of the corners: their mean on the sphere."""
    ra, dec = np.radians(ra_corners), np.radians(dec_corners)
    x = np.sum(np.cos(dec) * np.cos(ra))
    y = np.sum(np.cos(dec) * np.sin(ra))
    z = np.sum(np.sin(dec))
    ra_c = np.degrees(np.arctan2(y, x)) % 360.0
    dec_c = np.degrees(np.arctan2(z, np.hypot(x, y)))
    return float(ra_c), float(dec_c)


def gnomonic(ra, dec, ra0: float, dec0: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tangent-plane coordinates ``(xi, eta)`` [deg] about ``(ra0, dec0)``.

    Also returns ``front``: ``True`` for points on the hemisphere facing the
    tangent point (the projection is meaningless behind it).
    """
    ra, dec = np.radians(np.asarray(ra, dtype=float)), np.radians(np.asarray(dec, dtype=float))
    a0, d0 = np.radians(ra0), np.radians(dec0)
    dra = ra - a0
    cos_c = np.sin(d0) * np.sin(dec) + np.cos(d0) * np.cos(dec) * np.cos(dra)
    front = cos_c > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = np.cos(dec) * np.sin(dra) / cos_c
        eta = (np.cos(d0) * np.sin(dec) - np.sin(d0) * np.cos(dec) * np.cos(dra)) / cos_c
    return np.degrees(xi), np.degrees(eta), front


def tile_plane(tile_info: Any, margin: float = 0.0):
    """Tile centre and its corners on the tangent plane, shrunk by ``margin``.

    Returns ``(ra_c, dec_c, xi, eta)``; ``xi``/``eta`` [deg] are the four
    corners, moved ``margin`` of the way toward the centre.
    """
    ra, dec = tile_corners(tile_info)
    ra_c, dec_c = tile_center(ra, dec)
    xi, eta, _ = gnomonic(ra, dec, ra_c, dec_c)
    return ra_c, dec_c, xi * (1.0 - margin), eta * (1.0 - margin)


def trim_to_tile_polygon(
    tile_info: Any,
    catalog: Table,
    *,
    margin: float = 0.05,
    rakey: str = "ra",
    deckey: str = "dec",
) -> Table:
    """Trim ``catalog`` to sources inside the tile polygon.

    The polygon is defined by four corners ``(ra1, dec1) ... (ra4, dec4)``
    found in ``tile_info``. The polygon is shrunk toward its centre by a
    fraction ``margin`` to discard sources right on the edge. The test is
    done on the tangent plane about the tile centre, so tiles straddling
    RA = 0/360 and polar tiles are handled.

    Parameters
    ----------
    tile_info
        Either a single-row :class:`~astropy.table.Table` or a dict-like with
        keys ``ra1/dec1/ra2/dec2/ra3/dec3/ra4/dec4``.
    catalog
        Reference catalog table with RA/Dec columns.
    margin
        Polygon shrink factor; corners move ``margin`` of the way toward the
        polygon centroid. ``margin=0`` keeps the original corners.
    rakey, deckey
        Column names for RA / Dec in ``catalog``.

    Returns
    -------
    Table
        Trimmed catalog (rows unchanged).
    """
    ra_c, dec_c, xi_c, eta_c = tile_plane(tile_info, margin)
    polygon = Path(np.column_stack((np.append(xi_c, xi_c[0]), np.append(eta_c, eta_c[0]))))

    ra = np.ma.filled(np.ma.asarray(catalog[rakey], dtype=float), np.nan)
    dec = np.ma.filled(np.ma.asarray(catalog[deckey], dtype=float), np.nan)
    xi, eta, front = gnomonic(ra, dec, ra_c, dec_c)

    bbox = (
        front
        & (xi > xi_c.min()) & (xi < xi_c.max())
        & (eta > eta_c.min()) & (eta < eta_c.max())
    )
    inside = np.zeros(len(catalog), dtype=bool)
    if np.any(bbox):
        inside[bbox] = polygon.contains_points(np.column_stack((xi[bbox], eta[bbox])))
    return catalog[inside]


__all__ = [
    "trim_to_tile_polygon",
    "tile_corners",
    "tile_center",
    "tile_plane",
    "gnomonic",
]
