"""
Catalog file I/O: FITS or Parquet, chosen by column count.

A FITS binary table holds at most 999 columns (``TFIELDS``). With the full
filter registry the calibrated catalog can exceed that (a default run on
39+ bands), so :func:`write_catalog` resolves ``output_format``:

``"auto"``
    FITS when the table has <= 999 columns, otherwise Parquet (warning).
``"fits"``
    FITS; more than 999 columns raises :class:`ValueError`.
``"parquet"``
    Parquet.

The file suffix follows the format (``.fits`` / ``.parquet``). Parquet keeps
everything the FITS table carries: column units, descriptions, masks and the
table ``meta`` (header cards, ``(value, comment)`` tuples included).
:func:`read_catalog` reads either format and, by default, reduces
``(value, comment)`` meta entries to plain values, as a FITS read gives.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from astropy.table import Table

log = logging.getLogger(__name__)

FITS_MAX_COLUMNS = 999
OUTPUT_FORMATS = ("auto", "fits", "parquet")
CATALOG_SUFFIXES = {"fits": ".fits", "parquet": ".parquet"}
_KNOWN_SUFFIXES = (".fits", ".fit", ".fits.gz", ".parquet", ".pq")


def catalog_format_of(path: str | Path) -> str:
    """``'parquet'`` for ``.parquet``/``.pq`` paths, else ``'fits'``."""
    return "parquet" if str(path).lower().endswith((".parquet", ".pq")) else "fits"


def strip_catalog_suffix(name: str) -> str:
    """``name`` without a known catalog suffix (``.fits``, ``.parquet``, ...)."""
    lowered = name.lower()
    for suffix in sorted(_KNOWN_SUFFIXES, key=len, reverse=True):
        if lowered.endswith(suffix):
            return name[: -len(suffix)]
    return name


def with_catalog_suffix(path: str | Path, fmt: str) -> Path:
    """``path`` with its catalog suffix replaced by the one for ``fmt``."""
    p = Path(path)
    return p.with_name(strip_catalog_suffix(p.name) + CATALOG_SUFFIXES[fmt])


def catalog_siblings(path: str | Path) -> list[Path]:
    """The FITS and Parquet variants of ``path`` (same stem)."""
    return [with_catalog_suffix(path, fmt) for fmt in CATALOG_SUFFIXES]


def resolve_output_format(n_columns: int, output_format: str = "auto") -> str:
    """Concrete format (``'fits'``/``'parquet'``) for a table of ``n_columns``."""
    fmt = (output_format or "auto").lower()
    if fmt not in OUTPUT_FORMATS:
        raise ValueError(f"output_format must be one of {OUTPUT_FORMATS}, got {output_format!r}")
    if fmt == "auto":
        return "fits" if n_columns <= FITS_MAX_COLUMNS else "parquet"
    if fmt == "fits" and n_columns > FITS_MAX_COLUMNS:
        raise ValueError(
            f"Catalog has {n_columns} columns; FITS tables hold at most "
            f"{FITS_MAX_COLUMNS}. Use output_format='auto' or 'parquet', or "
            "trim bands/apertures/per-band masks."
        )
    return fmt


def write_catalog(
    table: Table,
    path: str | Path,
    output_format: str = "auto",
    *,
    overwrite: bool = True,
    remove_stale: bool = False,
) -> Path:
    """Write ``table`` as FITS or Parquet (see module docstring).

    Returns the path actually written: ``path`` with the suffix of the
    resolved format. With ``remove_stale`` the other-format sibling of that
    path (an earlier run's output) is deleted so only one catalog remains.
    """
    fmt = resolve_output_format(len(table.colnames), output_format)
    out = with_catalog_suffix(path, fmt)
    if fmt == "parquet" and (output_format or "auto").lower() == "auto":
        log.warning(
            "Catalog has %d columns (> %d FITS limit); writing Parquet: %s",
            len(table.colnames), FITS_MAX_COLUMNS, out,
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        table.write(str(out), format="parquet", overwrite=overwrite)
    else:
        table.write(str(out), format="fits", overwrite=overwrite)
    if remove_stale:
        for sib in catalog_siblings(out):
            if sib != out and sib.exists():
                log.info("Removing stale %s catalog: %s", catalog_format_of(sib), sib)
                sib.unlink()
    return out


def _plain_meta_value(value: Any) -> Any:
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], str):
        return value[0]
    return value


def read_catalog(path: str | Path, *, plain_meta: bool = True, **kwargs: Any) -> Table:
    """Read a phot7ds catalog written as FITS or Parquet.

    ``plain_meta`` reduces ``(value, comment)`` meta entries (kept by
    Parquet) to the value, so both formats give the same ``meta``. Extra
    ``kwargs`` go to :meth:`astropy.table.Table.read`.
    """
    fmt = catalog_format_of(path)
    tbl = Table.read(str(path), format=fmt, **kwargs)
    if plain_meta and fmt == "parquet":
        for key in list(tbl.meta):
            tbl.meta[key] = _plain_meta_value(tbl.meta[key])
    return tbl


def catalog_nrows(path: str | Path) -> int:
    """Row count of a FITS or Parquet catalog without reading the data."""
    if catalog_format_of(path) == "parquet":
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(str(path)).metadata.num_rows)
    from astropy.io import fits

    with fits.open(str(path)) as hdul:
        return int(hdul[1].header.get("NAXIS2", 0))


def find_existing_catalog(path: str | Path) -> Path | None:
    """The existing FITS or Parquet variant of ``path`` (exact path first)."""
    p = Path(path)
    if p.exists():
        return p
    for sib in catalog_siblings(p):
        if sib.exists():
            return sib
    return None


__all__ = [
    "CATALOG_SUFFIXES",
    "FITS_MAX_COLUMNS",
    "OUTPUT_FORMATS",
    "catalog_format_of",
    "catalog_nrows",
    "catalog_siblings",
    "find_existing_catalog",
    "read_catalog",
    "resolve_output_format",
    "strip_catalog_suffix",
    "with_catalog_suffix",
    "write_catalog",
]
