"""Tile-polygon geometry helpers."""
from __future__ import annotations

from typing import Any

import numpy as np
from astropy.table import Table
from matplotlib.path import Path


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
    fraction ``margin`` to discard sources right on the edge.

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
        Trimmed catalog.
    """
    def _get(key: str) -> float:
        if isinstance(tile_info, Table):
            return float(tile_info[key][0])
        return float(tile_info[key])

    ra_corners = np.array([_get(f"ra{i}") for i in (1, 2, 3, 4)])
    dec_corners = np.array([_get(f"dec{i}") for i in (1, 2, 3, 4)])

    ra_center = float(np.mean(ra_corners))
    dec_center = float(np.mean(dec_corners))
    ra_corners = ra_corners - (ra_corners - ra_center) * margin
    dec_corners = dec_corners - (dec_corners - dec_center) * margin

    polygon = Path(
        np.column_stack(
            (np.append(ra_corners, ra_corners[0]), np.append(dec_corners, dec_corners[0]))
        )
    )

    ra_min, ra_max = float(np.min(ra_corners)), float(np.max(ra_corners))
    dec_min, dec_max = float(np.min(dec_corners)), float(np.max(dec_corners))

    # If the polygon crosses RA=0/360, RA wraps; use an "outside-the-middle" test.
    if ra_max - ra_min > 180:
        bbox = (
            (catalog[rakey] > ra_min) | (catalog[rakey] < ra_max)
        ) & (catalog[deckey] > dec_min) & (catalog[deckey] < dec_max)
    else:
        bbox = (
            (catalog[rakey] > ra_min)
            & (catalog[rakey] < ra_max)
            & (catalog[deckey] > dec_min)
            & (catalog[deckey] < dec_max)
        )

    bbox_cat = catalog[bbox]
    if len(bbox_cat) == 0:
        return bbox_cat
    pts = np.column_stack((bbox_cat[rakey], bbox_cat[deckey]))
    return bbox_cat[polygon.contains_points(pts)]


__all__ = ["trim_to_tile_polygon"]
