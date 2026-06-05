"""
Value-added catalog assembly.

Merges the photo-z (eazy-py) and SED-fitting (FAST++) outputs back onto the
matched galaxy catalog, aligned by the ``target_ids`` row order, and writes
``{tile}_{ref}_value_added.fits``.
"""
from __future__ import annotations

import logging
import os

import numpy as np
from astropy.table import Table, hstack

from .config import VACConfig

log = logging.getLogger(__name__)


def _prefix_columns(tbl: Table, prefix: str, skip=("id",)) -> Table:
    out = Table()
    for col in tbl.colnames:
        name = col if col in skip else f"{prefix}{col}"
        out[name] = tbl[col]
    return out


def assemble_value_added(
    id_table: Table,
    cfg: VACConfig,
    tile: str,
    photoz_tbl: Table | None = None,
    fastpp_tbl: Table | None = None,
) -> str:
    """Merge photo-z + SED outputs onto the id table and write the VAC.

    All inputs share the row order of ``id_table`` (which was produced by
    the validity cut), so the merge is a positional ``hstack``. Missing
    stages are simply skipped.

    Returns
    -------
    str
        Path to the written value-added catalog.
    """
    n = len(id_table)
    pieces = [id_table]

    if photoz_tbl is not None:
        if len(photoz_tbl) != n:
            log.warning("photo-z rows (%d) != id rows (%d); aligning by index.",
                        len(photoz_tbl), n)
            photoz_tbl = _align(photoz_tbl, n)
        pieces.append(_prefix_columns(photoz_tbl, "eazy_"))

    if fastpp_tbl is not None:
        if len(fastpp_tbl) != n:
            log.warning("FAST++ rows (%d) != id rows (%d); aligning by index.",
                        len(fastpp_tbl), n)
            fastpp_tbl = _align(fastpp_tbl, n)
        pieces.append(_prefix_columns(fastpp_tbl, "fastpp_"))

    vac = hstack(pieces, join_type="exact", metadata_conflicts="silent")

    out_dir = cfg.vac_dir()
    os.makedirs(out_dir, exist_ok=True)
    out_path = str(out_dir / f"{tile}_{cfg.detection_ref}_value_added.fits")
    vac.write(out_path, format="fits", overwrite=True)
    log.info("Wrote value-added catalog: %s (%d rows, %d columns)",
             out_path, len(vac), len(vac.colnames))
    return out_path


def _align(tbl: Table, n: int) -> Table:
    """Pad/trim a table to ``n`` rows (defensive; should not normally fire)."""
    if len(tbl) >= n:
        return tbl[:n]
    pad = Table(tbl[:1])
    for col in pad.colnames:
        pad[col] = [np.nan if np.issubdtype(np.asarray(tbl[col]).dtype, np.floating) else tbl[col][0]]
    from astropy.table import vstack
    while len(tbl) < n:
        tbl = vstack([tbl, pad])
    return tbl


__all__ = ["assemble_value_added"]
