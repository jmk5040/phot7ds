"""
Sky cross-matching between catalogs.

:func:`matching` joins two catalogs by angular separation, supporting the
usual SQL-style join semantics (``left``/``right``/``inner``/``outer``) and
either nearest-neighbour (``closest``) or all-pairs (``all``) matching.

Ported from the 7DT ``Utils_7DT`` helper so phot7ds is standalone; it only
depends on numpy/astropy.
"""
from __future__ import annotations

from typing import Sequence

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import MaskedColumn, Table, vstack


def _matching_buffer(n_rows: int, column) -> np.ndarray:
    """Allocate placeholder data for a MaskedColumn (ignored where masked)."""
    dt = column.dtype
    kind = np.dtype(dt).kind
    if kind == "f":
        return np.full(n_rows, np.nan, dtype=dt)
    if kind in "iu":
        return np.zeros(n_rows, dtype=dt)
    if kind == "b":
        return np.zeros(n_rows, dtype=bool)
    if kind == "c":
        return np.full(n_rows, np.nan + 0j, dtype=dt)
    if kind in "SU":
        out = np.empty(n_rows, dtype=dt)
        out[:] = ""
        return out
    return np.array([None] * n_rows, dtype=object)


def _matching_ref_name(col: str, effective_prefix: str) -> str:
    return f"{effective_prefix}{col}" if effective_prefix else col


def _matching_empty_merged(intbl, reftbl, effective_prefix, sep_col, n_rows):
    cols = []
    for col in intbl.colnames:
        cols.append(
            MaskedColumn(
                data=_matching_buffer(n_rows, intbl[col]),
                mask=np.ones(n_rows, dtype=bool),
                name=col,
            )
        )
    for col in reftbl.colnames:
        name = _matching_ref_name(col, effective_prefix)
        cols.append(
            MaskedColumn(
                data=_matching_buffer(n_rows, reftbl[col]),
                mask=np.ones(n_rows, dtype=bool),
                name=name,
            )
        )
    cols.append(
        MaskedColumn(
            data=np.full(n_rows, np.nan, dtype=float),
            mask=np.ones(n_rows, dtype=bool),
            name=sep_col,
        )
    )
    return Table(cols)


def _matching_fill_int_rows(table, intbl, rows, src_rows):
    for col in intbl.colnames:
        src = intbl[col][src_rows]
        table[col][rows] = src
        if hasattr(intbl[col], "mask"):
            table[col].mask[rows] = np.ma.getmaskarray(src)
        else:
            table[col].mask[rows] = False


def _matching_fill_ref_rows(table, reftbl, effective_prefix, rows, src_rows):
    for col in reftbl.colnames:
        name = _matching_ref_name(col, effective_prefix)
        src = reftbl[col][src_rows]
        table[name][rows] = src
        if hasattr(reftbl[col], "mask"):
            table[name].mask[rows] = np.ma.getmaskarray(src)
        else:
            table[name].mask[rows] = False


def _matching_resolve_prefix(intbl, reftbl, ref_prefix):
    overlap = set(intbl.colnames) & set(reftbl.colnames)
    effective_prefix = ref_prefix
    if overlap and ref_prefix == "":
        effective_prefix = "ref_"
        print(f"[matching] Column name overlap detected: {sorted(overlap)}")
        print("[matching] Using ref_prefix='ref_' to avoid collisions.")
    sep_col = f"{effective_prefix}sep" if effective_prefix else "sep"
    return effective_prefix, sep_col


def _matching_closest(
    intbl, reftbl, incoord, refcoord, sep_angle, join_type, effective_prefix, sep_col
):
    """Nearest-neighbour match: each input row gets at most one ref within sep."""
    indx, d2d, _ = incoord.match_to_catalog_sky(refcoord)
    match_mask = d2d < sep_angle
    matched_rows = np.where(match_mask)[0]
    ref_rows = indx[match_mask]

    left_tbl = _matching_empty_merged(intbl, reftbl, effective_prefix, sep_col, len(intbl))
    _matching_fill_int_rows(left_tbl, intbl, np.arange(len(intbl)), np.arange(len(intbl)))
    if np.any(match_mask):
        _matching_fill_ref_rows(left_tbl, reftbl, effective_prefix, matched_rows, ref_rows)
        left_tbl[sep_col][matched_rows] = d2d.arcsec[match_mask]
        left_tbl[sep_col].mask[matched_rows] = False

    if join_type == "left":
        return left_tbl
    if join_type == "inner":
        return left_tbl[match_mask]

    rindx, rd2d, _ = refcoord.match_to_catalog_sky(incoord)
    rmask = rd2d < sep_angle
    right_tbl = _matching_empty_merged(intbl, reftbl, effective_prefix, sep_col, len(reftbl))
    _matching_fill_ref_rows(right_tbl, reftbl, effective_prefix, np.arange(len(reftbl)), np.arange(len(reftbl)))
    if np.any(rmask):
        rrows = np.where(rmask)[0]
        _matching_fill_int_rows(right_tbl, intbl, rrows, rindx[rmask])
        right_tbl[sep_col][rrows] = rd2d.arcsec[rmask]
        right_tbl[sep_col].mask[rrows] = False

    if join_type == "right":
        return right_tbl

    unmatched_ref = ~rmask
    if not np.any(unmatched_ref):
        return left_tbl
    n_unm = np.count_nonzero(unmatched_ref)
    extra = _matching_empty_merged(intbl, reftbl, effective_prefix, sep_col, n_unm)
    ref_unm = np.where(unmatched_ref)[0]
    _matching_fill_ref_rows(extra, reftbl, effective_prefix, np.arange(n_unm), ref_unm)
    return vstack([left_tbl, extra], join_type="exact")


def _matching_all_pairs(
    intbl, reftbl, incoord, refcoord, sep_angle, join_type, effective_prefix, sep_col
):
    """All pairs within sep (may duplicate rows on either side)."""
    idx_ref, idx_in, d2d, _ = incoord.search_around_sky(refcoord, sep_angle)
    matched_tbl = _matching_empty_merged(intbl, reftbl, effective_prefix, sep_col, len(idx_in))
    if len(idx_in) > 0:
        _matching_fill_int_rows(matched_tbl, intbl, np.arange(len(idx_in)), idx_in)
        _matching_fill_ref_rows(matched_tbl, reftbl, effective_prefix, np.arange(len(idx_in)), idx_ref)
        matched_tbl[sep_col][:] = d2d.arcsec
        matched_tbl[sep_col].mask[:] = False

    if join_type == "inner":
        return matched_tbl

    if join_type in {"left", "outer"}:
        unmatched_int = np.setdiff1d(np.arange(len(intbl)), np.unique(idx_in))
        if len(unmatched_int) > 0:
            extra = _matching_empty_merged(intbl, reftbl, effective_prefix, sep_col, len(unmatched_int))
            _matching_fill_int_rows(extra, intbl, np.arange(len(unmatched_int)), unmatched_int)
            matched_tbl = vstack([matched_tbl, extra], join_type="exact")

    if join_type in {"right", "outer"}:
        unmatched_ref = np.setdiff1d(np.arange(len(reftbl)), np.unique(idx_ref))
        if len(unmatched_ref) > 0:
            extra = _matching_empty_merged(intbl, reftbl, effective_prefix, sep_col, len(unmatched_ref))
            _matching_fill_ref_rows(extra, reftbl, effective_prefix, np.arange(len(unmatched_ref)), unmatched_ref)
            matched_tbl = vstack([matched_tbl, extra], join_type="exact")

    return matched_tbl


def matching(
    intbl: Table,
    reftbl: Table,
    inra: Sequence[float],
    indec: Sequence[float],
    refra: Sequence[float],
    refdec: Sequence[float],
    sep: float = 2.0,
    join_type: str = "inner",
    duplicate: str = "closest",
    ref_prefix: str = "",
) -> Table:
    """Match two catalogs by sky position (RA/Dec).

    Parameters
    ----------
    intbl, reftbl
        Input and reference catalogs.
    inra, indec, refra, refdec
        RA/Dec (degrees) for the input and reference rows.
    sep
        Match radius in arcseconds.
    join_type : {'left', 'right', 'inner', 'outer'}
        Join strategy for unmatched rows.
    duplicate : {'closest', 'all'}
        ``closest``: each input row gets the nearest reference object within
        ``sep`` (at most one match). ``all``: every pair within ``sep`` is
        returned (multiple rows per object possible).
    ref_prefix
        Prefix for reference columns. If there are collisions with ``intbl``
        and ``ref_prefix`` is empty, it is overridden to ``"ref_"``.

    Returns
    -------
    Table
        Merged table with a ``{prefix}sep`` column (arcsec) for matched rows.
        Unmatched cells are masked.
    """
    join_type = join_type.lower()
    duplicate = duplicate.lower()
    if join_type not in {"left", "right", "inner", "outer"}:
        raise ValueError("join_type must be one of 'left', 'right', 'inner', 'outer'")
    if duplicate not in {"closest", "all"}:
        raise ValueError("duplicate must be one of 'closest' or 'all'")

    incoord = SkyCoord(inra, indec, unit=(u.deg, u.deg))
    refcoord = SkyCoord(refra, refdec, unit=(u.deg, u.deg))
    sep_angle = sep * u.arcsec
    effective_prefix, sep_col = _matching_resolve_prefix(intbl, reftbl, ref_prefix)

    if duplicate == "closest":
        return _matching_closest(
            intbl, reftbl, incoord, refcoord, sep_angle, join_type, effective_prefix, sep_col
        )
    return _matching_all_pairs(
        intbl, reftbl, incoord, refcoord, sep_angle, join_type, effective_prefix, sep_col
    )


__all__ = ["matching"]
