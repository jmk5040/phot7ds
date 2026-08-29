"""
Value-added catalog orchestrator.

:func:`run_value_added` ties together cross-matching, flux assembly,
eazy-py photo-z, FAST++ SED fitting, and final catalog assembly, mirroring
the design of :func:`phot7ds.run_photometry`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
from astropy.table import Table

from .catalog import assemble_value_added
from .config import VACConfig
from .crossmatch import build_galaxy_catalog
from .fluxes import build_flux_catalog, drop_dead_bands, write_flux_inputs
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


def select_tile_row(tile_table: Table, tile: str) -> Table:
    """Find the row for ``tile`` in the tile table (id/tile/name column)."""
    for key in ("id", "tile", "name", "TILE", "ID"):
        if key in tile_table.colnames:
            mask = np.asarray(tile_table[key]).astype(str) == str(tile)
            if mask.any():
                return tile_table[mask][:1]
    raise KeyError(f"Tile {tile!r} not found in tile table (columns {tile_table.colnames}).")


# Retained: external scripts imported the private name before it was public.
_select_tile_row = select_tile_row


def _load_catalog(
    catalog_path: str | Path, cfg: VACConfig, *, drop_empty_bands: bool,
    apertures: Sequence[str],
) -> Table:
    """Read a photometric catalog and apply the coverage / dead-band cuts."""
    catalog = Table.read(catalog_path, format="fits")
    log.info("Loaded catalog %s (%d rows).", catalog_path, len(catalog))

    flag_col = cfg.cover_flag_column
    if flag_col in catalog.colnames:
        good = np.asarray(catalog[flag_col]) == 0
        log.info("Coverage cut: %d/%d rows kept.", int(good.sum()), len(catalog))
        catalog = catalog[good]

    if drop_empty_bands:
        catalog, _live = drop_dead_bands(catalog, apertures)
    return catalog


def run_value_added(
    catalog_path: str | Path,
    tile: str,
    tile_table: Table | str | Path,
    config: VACConfig,
    *,
    do_photoz: bool = True,
    do_sedfit: bool = True,
    drop_empty_bands: bool = False,
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
    drop_empty_bands
        Remove magnitude columns that are entirely empty before detecting
        filters (see :func:`phot7ds.vac.fluxes.drop_dead_bands`). Worth
        enabling when tiles differ in band coverage.

    Returns
    -------
    VACResult
    """
    # Up-front sanity check: verify every required config file / template /
    # binary for the requested stages, reporting all problems at once before
    # any long-running work begins.
    config.preflight(do_photoz=do_photoz, do_sedfit=do_sedfit)

    if not isinstance(tile_table, Table):
        tile_table = Table.read(tile_table)
    tile_info = select_tile_row(tile_table, tile)

    catalog = _load_catalog(catalog_path, config,
                            drop_empty_bands=drop_empty_bands,
                            apertures=(config.aperture,))

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


def _match_vhs_mag_set(
    config_photoz: VACConfig, config_sedfit: VACConfig, tile: str
) -> str:
    """Which VHS staging to cross-match against for a split run.

    The ap6 files are a superset of the plain VHS ones, so when either stage
    asks for ap6 that staging can serve both and is preferred. Preference
    yields to what is actually on disk, though: matching against an absent
    ap6 file would drop the NIR bands from *both* stages, where matching the
    staged petro file at least keeps them for the stage that wants Petrosian
    magnitudes. With nothing staged the preferred set is kept so that
    ``auto_download`` still has something to fetch.
    """
    wanted = (config_photoz.vhs_mag_set, config_sedfit.vhs_mag_set)
    preferred = ["ap6"] if "ap6" in wanted else []
    preferred += [s for s in wanted if s not in preferred]
    for mag_set in preferred:
        if replace(config_photoz, vhs_mag_set=mag_set).vhs_path(tile).exists():
            return mag_set
    return preferred[0]


def run_value_added_split(
    catalog_path: str | Path,
    tile: str,
    tile_table: Table | str | Path,
    config_photoz: VACConfig,
    config_sedfit: VACConfig,
    *,
    do_photoz: bool = True,
    do_sedfit: bool = True,
    drop_empty_bands: bool = False,
) -> VACResult:
    """Value-added catalog with independent photo-z and SED-fit inputs.

    :func:`run_value_added` builds a single flux catalog and hands it to both
    EAzY and FAST++. This variant builds one per stage, so the photo-z can run
    on (say) 7DS fixed apertures alone while the SED fit uses total magnitudes
    plus VHS/GALEX/WISE. The two configs may differ in ``aperture``,
    ``vhs_mag_set`` and the ``use_*`` external toggles; everything else is
    taken from ``config_sedfit`` for the merged output.

    Everything downstream is aligned by row order — FAST++ reads the ``.zout``
    that EAzY wrote, and :func:`assemble_value_added` hstacks positionally — so
    both catalogs are cut to the galaxies that survive **both** coverage cuts
    before either binary runs.
    """
    config_photoz.preflight(do_photoz=do_photoz, do_sedfit=False)
    config_sedfit.preflight(do_photoz=False, do_sedfit=do_sedfit)

    if not isinstance(tile_table, Table):
        tile_table = Table.read(str(tile_table))
    tile_info = select_tile_row(tile_table, tile)

    catalog = _load_catalog(
        catalog_path, config_photoz, drop_empty_bands=drop_empty_bands,
        apertures=(config_photoz.aperture, config_sedfit.aperture),
    )

    # Match once, against the union of what either stage needs.
    match_cfg = replace(
        config_photoz,
        vhs_mag_set=_match_vhs_mag_set(config_photoz, config_sedfit, tile),
        use_vhs=config_photoz.use_vhs or config_sedfit.use_vhs,
        use_galex=config_photoz.use_galex or config_sedfit.use_galex,
        use_wise=config_photoz.use_wise or config_sedfit.use_wise,
    )
    galaxy_tbl = build_galaxy_catalog(catalog, tile_info, match_cfg, tile)

    flux_pz, ids_pz, info_pz = build_flux_catalog(
        galaxy_tbl, config_photoz, tile, tile_info, write=False)
    flux_sed, ids_sed, info_sed = build_flux_catalog(
        galaxy_tbl, config_sedfit, tile, tile_info, write=False)

    # '#id' is assigned before either coverage cut, so it identifies the same
    # galaxy in both tables and can be intersected directly.
    common = np.intersect1d(np.asarray(flux_pz["#id"]), np.asarray(flux_sed["#id"]))
    keep_pz = np.isin(np.asarray(flux_pz["#id"]), common)
    keep_sed = np.isin(np.asarray(flux_sed["#id"]), common)
    flux_pz, ids_pz = flux_pz[keep_pz], ids_pz[keep_pz]
    flux_sed, ids_sed = flux_sed[keep_sed], ids_sed[keep_sed]
    log.info("Split coverage cuts: photo-z %d / SED %d -> %d galaxies in both",
             int(keep_pz.sum()), int(keep_sed.sum()), len(common))

    write_flux_inputs(flux_pz, ids_pz, config_photoz.photoz_dir(tile), tile,
                      config_photoz)
    write_flux_inputs(flux_sed, ids_sed, config_sedfit.sedfit_dir(tile), tile,
                      config_sedfit)

    photoz_tbl = sedfit_tbl = None
    eazy_info: dict | None = None
    fastpp_info: dict | None = None
    name_zphot = "z_a"
    if do_photoz:
        if config_photoz.photoz_engine == "binary":
            from .photoz_binary import run_eazy_binary

            photoz_tbl, eazy_info = run_eazy_binary(config_photoz, tile)
        else:
            from .photoz import run_eazy

            photoz_tbl, eazy_info = run_eazy(config_photoz, tile)
        name_zphot = eazy_info["zphot_column"]
    if do_sedfit:
        from .sedfit import run_fastpp

        sedfit_tbl, fastpp_info = run_fastpp(config_sedfit, tile,
                                             name_zphot=name_zphot)

    vac_path = assemble_value_added(ids_sed, config_sedfit, tile,
                                    photoz_tbl=photoz_tbl, fastpp_tbl=sedfit_tbl)
    log_path = write_run_log(
        config_sedfit, tile,
        config_sedfit.vac_dir() / f"{tile}_{config_sedfit.detection_ref}_vac.log",
        match_info={"n_galaxies_matched": len(galaxy_tbl),
                    "input_rows": len(catalog)},
        flux_info={**info_sed, "n_pass": len(flux_sed),
                   "photoz_stage": {"aperture": config_photoz.aperture,
                                    "vhs_mag_set": config_photoz.vhs_mag_set,
                                    "filters": info_pz["filters"]},
                   "sedfit_stage": {"aperture": config_sedfit.aperture,
                                    "vhs_mag_set": config_sedfit.vhs_mag_set,
                                    "filters": info_sed["filters"]}},
        eazy_info=eazy_info, fastpp_info=fastpp_info,
        extra={"value_added_catalog": vac_path, "split_stages": True},
    )
    return VACResult(
        tile=tile,
        n_matched=len(galaxy_tbl),
        n_flux=len(flux_sed),
        value_added_path=vac_path,
        photoz_done=do_photoz,
        sedfit_done=do_sedfit,
        log_path=log_path,
    )


__all__ = [
    "run_value_added",
    "run_value_added_split",
    "VACResult",
    "select_tile_row",
]
