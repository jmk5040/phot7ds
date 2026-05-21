"""
Canonical schema for the unified photometry catalog.

The pipeline emits one FITS catalog per ``(image set, detection image)`` pair
with a stable column order so downstream consumers can stack runs without
worrying about missing or renamed columns.

Three behaviours:

1. SE++ ``-N`` duplicate columns (created when the same image is registered
   more than once in a single SE++ run) are dropped.
2. Missing canonical columns are inserted as plain :class:`~astropy.table.Column`
   filled with a finite sentinel (``PLACEHOLDER_FILL = -99.0``) and tagged
   ``description="PLACEHOLDER"``. A finite sentinel is used (rather than NaN)
   so the column reads back as a plain Column instead of a ``MaskedColumn``,
   keeping it visually distinct from SE++'s genuine low-SNR masked cells.
3. Any extra columns not in the schema are kept at the end of the table and
   counted in ``meta["NEXTRA"]``.

Read catalogs back with :func:`load_unified_catalog` to optionally convert the
on-disk sentinel into ``np.nan``.
"""
from __future__ import annotations

import re
from typing import Iterable, Sequence

import astropy.units as u
import numpy as np
from astropy.table import Column, MaskedColumn, Table

PLACEHOLDER_TAG = "PLACEHOLDER"
PLACEHOLDER_FILL = -99.0

_DUP_RE = re.compile(r"-\d+$")

CANONICAL_BASIC_COLS: list[str] = """
source_id
detection_id
group_id
world_centroid_alpha
world_centroid_delta
pixel_centroid_x
pixel_centroid_y
error_ellipse_a
error_ellipse_b
error_ellipse_theta
error_centroid_x2
error_centroid_y2
error_centroid_xy
peak_value
peak_value_x
peak_value_y
snrratio
kron_radius
kron_flag
source_flags
isophotal_image_flags_badpix
isophotal_image_flags_cover
isophotal_image_flags_pixel_count_badpix
isophotal_image_flags_pixel_count_cover
ellipse_a
ellipse_b
ellipse_theta
ellipse_cxx
ellipse_cyy
ellipse_cxy
area
elongation
ellipticity
""".split()


def _flux_fraction_suffix(f) -> str:
    """Normalise a flux fraction to the SE++ column suffix.

    ``0.5 -> '50'``, ``0.9 -> '90'``, ``'50' -> '50'``.
    """
    if isinstance(f, str):
        return f
    return str(int(round(float(f) * 100)))


def build_canonical_schema(
    bands: Sequence[str],
    apertures: Sequence[str],
    flux_fractions: Sequence = (0.5, 0.9),
) -> list[str]:
    """Return the canonical ordered list of column names.

    Parameters
    ----------
    bands
        Filter names, e.g. ``['g', 'r', 'i', 'm400', ..., 'm875']``.
    apertures
        Aperture names, e.g. ``['aper5', 'aper10', 'auto']``.
    flux_fractions
        Either fractions like 0.5/0.9 or SE++ suffixes ``'50'``/``'90'``.
    """
    bands = list(bands)
    apertures = list(apertures)
    suffixes = [_flux_fraction_suffix(f) for f in flux_fractions]

    cols = list(CANONICAL_BASIC_COLS)
    for aperture in apertures:
        for col_fmt in ("flux", "flux_err", "mag", "mag_err", "flags"):
            for band in bands:
                cols.append(f"{aperture}_{col_fmt}_{band}")
            if "mag" in col_fmt:
                # Spatially-corrected (2D zero-point) magnitudes.
                for band in bands:
                    cols.append(f"{aperture}c_{col_fmt}_{band}")
    for suffix in suffixes:
        for band in bands:
            cols.append(f"flux_rad_{suffix}_{band}")
    return cols


def drop_seplusplus_duplicates(cat: Table) -> list[str]:
    """Drop columns whose name ends in ``-N`` (N is an integer)."""
    dups = [c for c in cat.colnames if _DUP_RE.search(c)]
    if dups:
        cat.remove_columns(dups)
    return dups


def _infer_canonical_dtype(col_name: str) -> np.dtype:
    """Fallback dtype for columns absent from every reference catalog."""
    if col_name.endswith("_id") or col_name.endswith("_flags") or "_flags_" in col_name:
        return np.dtype(">i8")
    if col_name == "area":
        return np.dtype(">i8")
    if col_name == "kron_flag":
        return np.dtype(">i4")
    return np.dtype(">f4")


def standardize_catalog(
    cat: Table,
    schema: Sequence[str],
    dtype_map: dict[str, np.dtype] | None = None,
) -> Table:
    """Return a new :class:`~astropy.table.Table` conforming to ``schema``.

    - SE++ ``-N`` duplicate columns are dropped.
    - Missing canonical columns are inserted as a plain Column filled with
      ``PLACEHOLDER_FILL`` (integer dtypes promoted to float64); these are
      tagged ``description="PLACEHOLDER"``.
    - Columns not in ``schema`` are kept at the end and counted in ``meta``.

    Counts populated in ``cat.meta``: ``NPLACE``, ``NEXTRA``, ``NDUPS``.
    """
    dtype_map = dtype_map or {}
    n = len(cat)

    dropped_dups = drop_seplusplus_duplicates(cat)

    placeholders: list[str] = []
    for col in schema:
        if col in cat.colnames:
            continue
        dtype = dtype_map.get(col) or _infer_canonical_dtype(col)
        if np.issubdtype(dtype, np.floating):
            data = np.full(n, PLACEHOLDER_FILL, dtype=dtype)
        else:
            data = np.full(n, PLACEHOLDER_FILL, dtype=np.float64)
        cat[col] = Column(data=data, name=col)
        cat[col].description = PLACEHOLDER_TAG
        placeholders.append(col)

    extras = [c for c in cat.colnames if c not in schema]
    out = cat[list(schema) + extras]

    out.meta["NPLACE"] = len(placeholders)
    out.meta["NEXTRA"] = len(extras)
    out.meta["NDUPS"] = len(dropped_dups)
    return out


def placeholder_columns(cat: Table) -> list[str]:
    """Return column names added by the unifier (placeholder fills)."""
    return [
        c for c in cat.colnames
        if getattr(cat[c], "description", None) == PLACEHOLDER_TAG
    ]


def is_placeholder_column(col) -> bool:
    """Return ``True`` if ``col`` was added by the unifier."""
    return getattr(col, "description", None) == PLACEHOLDER_TAG


def strip_nonfits_units(table: Table) -> None:
    """Replace astropy units that don't round-trip through the FITS unit
    parser with a FITS-safe equivalent (or drop the unit entirely).

    Modifies ``table`` in place.
    """
    for col in table.itercols():
        unit = getattr(col, "unit", None)
        if unit is None:
            continue
        unit_str = str(unit).replace("pixel", "pix").replace("^{", "**").replace("}", "")
        try:
            col.unit = u.Unit(unit_str)
            col.unit.to_string(format="fits")
        except Exception:
            col.unit = None


def load_unified_catalog(path: str, fill_nan: bool = True) -> Table:
    """Read a unified FITS catalog written by this pipeline.

    Parameters
    ----------
    path
        FITS file path.
    fill_nan
        If True (default), replace the on-disk sentinel ``PLACEHOLDER_FILL``
        with ``np.nan`` in memory. If False, keep the sentinel visible.

    Notes
    -----
    Placeholder columns are written with the finite sentinel
    ``PLACEHOLDER_FILL`` so the default :meth:`Table.read` returns a plain
    (unmasked) Column for them. SE++'s low-SNR cells stay as
    :class:`~astropy.table.MaskedColumn` (``'--'``) in both modes.
    """
    cat = Table.read(path)
    for cname in placeholder_columns(cat):
        col = cat[cname]
        data = np.asarray(col.data, dtype=col.dtype).copy()
        if fill_nan:
            if not np.issubdtype(data.dtype, np.floating):
                data = data.astype(np.float64, copy=False)
            data[data == PLACEHOLDER_FILL] = np.nan
        cat.replace_column(cname, Column(data=data, name=cname, description=PLACEHOLDER_TAG))
    return cat


def convert_nan_placeholders_to_sentinel(path: str, out_path: str | None = None) -> int:
    """Convert an existing unified catalog whose placeholder cells are
    stored as NaN (and therefore masked on read) into one that uses the
    finite ``PLACEHOLDER_FILL`` sentinel.

    Returns the number of columns fixed.
    """
    cat = Table.read(path)
    n_fixed = 0
    for cname in placeholder_columns(cat):
        col = cat[cname]
        if isinstance(col, MaskedColumn):
            data = np.asarray(col.data, dtype=col.dtype).copy()
            if np.issubdtype(data.dtype, np.floating):
                data[np.isnan(data) | np.asarray(col.mask)] = PLACEHOLDER_FILL
            else:
                data = np.full(len(col), PLACEHOLDER_FILL, dtype=np.float64)
            cat.replace_column(
                cname, Column(data=data, name=cname, description=PLACEHOLDER_TAG)
            )
            n_fixed += 1
    cat.write(out_path or path, overwrite=True)
    return n_fixed


__all__ = [
    "PLACEHOLDER_TAG",
    "PLACEHOLDER_FILL",
    "CANONICAL_BASIC_COLS",
    "build_canonical_schema",
    "drop_seplusplus_duplicates",
    "standardize_catalog",
    "placeholder_columns",
    "is_placeholder_column",
    "strip_nonfits_units",
    "load_unified_catalog",
    "convert_nan_placeholders_to_sentinel",
]
