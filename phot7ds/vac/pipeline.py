"""
Value-added catalog orchestrator.

:func:`run_value_added` ties together cross-matching, flux assembly,
eazy-py photo-z, FAST++ SED fitting, and final catalog assembly, mirroring
the design of :func:`phot7ds.run_photometry`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.table import Table

from .catalog import assemble_value_added
from .config import VACConfig
from .crossmatch import build_galaxy_catalog
from .fluxes import build_flux_catalog
from .report import write_run_log

log = logging.getLogger(__name__)


@dataclass
class VACResult:
    """Outcome of a value-added catalog run."""

    tile: str
    n_matched: int
    n_flux: int
    value_added_path: str | None
    photoz_done: bool
    sedfit_done: bool
    log_path: str | None = None


def _select_tile_row(tile_table: Table, tile: str) -> Table:
    """Find the row for ``tile`` in the tile table (id/tile/name column)."""
    for key in ("id", "tile", "name", "TILE", "ID"):
        if key in tile_table.colnames:
            mask = np.asarray(tile_table[key]).astype(str) == str(tile)
            if mask.any():
                return tile_table[mask][:1]
    raise KeyError(f"Tile {tile!r} not found in tile table (columns {tile_table.colnames}).")


def run_value_added(
    catalog_path: str | Path,
    tile: str,
    tile_table: Table | str | Path,
    config: VACConfig,
    *,
    do_photoz: bool = True,
    do_sedfit: bool = True,
) -> VACResult:
    """Build a value-added catalog for one tile.

    Parameters
    ----------
    catalog_path
        phot7ds photometric catalog (FITS) for the tile.
    tile
        Tile identifier.
    tile_table
        Tile-definition table (or path to one) with the polygon corners.
    config
        :class:`VACConfig` (validated up front).
    do_photoz, do_sedfit
        Toggle the eazy-py and FAST++ stages.

    Returns
    -------
    VACResult
    """
    config.validate(
        require_fastpp=do_sedfit,
        require_eazy_bin=do_photoz and config.photoz_engine == "binary",
    )

    if not isinstance(tile_table, Table):
        tile_table = Table.read(tile_table)
    tile_info = _select_tile_row(tile_table, tile)

    catalog = Table.read(catalog_path, format="fits")
    log.info("Loaded catalog %s (%d rows).", catalog_path, len(catalog))

    # Coverage filter (drop flagged rows when the column is present).
    flag_col = config.cover_flag_column
    if flag_col in catalog.colnames:
        good = np.asarray(catalog[flag_col]) == 0
        log.info("Coverage cut: %d/%d rows kept.", int(good.sum()), len(catalog))
        catalog = catalog[good]

    galaxy_tbl = build_galaxy_catalog(catalog, tile_info, config, tile)
    flux_tbl, id_table, flux_info = build_flux_catalog(galaxy_tbl, config, tile, tile_info)

    photoz_tbl = None
    sedfit_tbl = None
    eazy_info: dict | None = None
    fastpp_info: dict | None = None
    photoz_done = False
    sedfit_done = False
    name_zphot = "z_a"  # default when no prior / no photo-z

    if do_photoz:
        if config.photoz_engine == "binary":
            from .photoz_binary import run_eazy_binary

            photoz_tbl, eazy_info = run_eazy_binary(config, tile)
        else:
            from .photoz import run_eazy

            photoz_tbl, eazy_info = run_eazy(config, tile)
        name_zphot = eazy_info["zphot_column"]
        photoz_done = True

    if do_sedfit:
        if not do_photoz:
            log.warning("FAST++ needs photo-z input; enabling photo-z output usage.")
        from .sedfit import run_fastpp

        sedfit_tbl, fastpp_info = run_fastpp(config, tile, name_zphot=name_zphot)
        sedfit_done = True

    vac_path = assemble_value_added(
        id_table, config, tile, photoz_tbl=photoz_tbl, fastpp_tbl=sedfit_tbl
    )

    log_path = write_run_log(
        config,
        tile,
        config.vac_dir() / f"{tile}_{config.detection_ref}_vac.log",
        match_info={
            "n_galaxies_matched": len(galaxy_tbl),
            "input_rows": len(catalog),
        },
        flux_info=flux_info,
        eazy_info=eazy_info,
        fastpp_info=fastpp_info,
        extra={"value_added_catalog": vac_path},
    )

    return VACResult(
        tile=tile,
        n_matched=len(galaxy_tbl),
        n_flux=len(flux_tbl),
        value_added_path=vac_path,
        photoz_done=photoz_done,
        sedfit_done=sedfit_done,
        log_path=log_path,
    )


__all__ = ["run_value_added", "VACResult"]
