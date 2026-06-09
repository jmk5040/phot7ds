"""
Galaxy cross-matching for the value-added catalog.

The phot7ds catalog is matched against the REGALADE galaxy reference
(required) and, when available, VHS (NIR) and GALEX (UV). VHS Vega
magnitudes are converted to AB. Duplicate REGALADE associations are
collapsed to the brightest 7DS match.
"""
from __future__ import annotations

import logging
import os

import numpy as np
from astropy.table import Table

from ..crossmatch import matching
from ..tile_geometry import trim_to_tile_polygon
from .config import VACConfig
from .vizier import ensure_external_catalog

log = logging.getLogger(__name__)

# VHS Vega -> AB offsets (mag).
_VHS_VEGA_TO_AB = {"Ypmag": 0.63, "Jpmag": 0.94, "Hpmag": 1.38, "Kspmag": 1.90}


def build_galaxy_catalog(
    catalog: Table,
    tile_info,
    cfg: VACConfig,
    tile: str,
) -> Table:
    """Match the photometric catalog to galaxy references and dedup.

    Parameters
    ----------
    catalog
        The phot7ds photometric catalog (already coverage-filtered).
    tile_info
        Single-row tile table (for the polygon trim / center).
    cfg
        :class:`VACConfig`.
    tile
        Tile identifier (selects the per-tile reference files).

    Returns
    -------
    Table
        Galaxy catalog: REGALADE-matched rows (prefixed ``regalade_``,
        optional ``vhs_``/``galex_``), deduped to one row per REGALADE
        source, with an added ``nmatch`` column.
    """
    regalade_path = cfg.regalade_path(tile)
    if not os.path.exists(regalade_path):
        ensure_external_catalog("regalade", tile, tile_info, regalade_path, cfg)
    if not os.path.exists(regalade_path):
        raise FileNotFoundError(
            f"REGALADE reference catalog not found for tile {tile}: {regalade_path}. "
            "Provide it or enable VACConfig.auto_download."
        )
    regalcat = trim_to_tile_polygon(
        tile_info,
        Table.read(regalade_path, format="fits"),
        rakey=cfg.regalade_ra_key,
        deckey=cfg.regalade_dec_key,
    )
    log.info("REGALADE galaxies in tile %s: %d", tile, len(regalcat))

    mtbl = matching(
        catalog,
        regalcat,
        inra=np.array(catalog[cfg.ra_column]),
        indec=np.array(catalog[cfg.dec_column]),
        refra=np.array(regalcat[cfg.regalade_ra_key]),
        refdec=np.array(regalcat[cfg.regalade_dec_key]),
        sep=cfg.match_radius_arcsec,
        join_type="inner",
        duplicate="closest",
        ref_prefix="regalade_",
    )
    log.info("REGALADE matches: %d (%.1f%% of references)",
             len(mtbl), 100.0 * len(mtbl) / max(len(regalcat), 1))

    if cfg.use_vhs:
        mtbl = _match_vhs(mtbl, cfg, tile, tile_info)
    if cfg.use_galex:
        mtbl = _match_galex(mtbl, cfg, tile, tile_info)

    return _dedup_brightest(mtbl, cfg)


def _match_vhs(mtbl: Table, cfg: VACConfig, tile: str, tile_info) -> Table:
    path = cfg.vhs_path(tile)
    if not os.path.exists(path):
        ensure_external_catalog("vhs", tile, tile_info, path, cfg)
    if not os.path.exists(path):
        log.info("No VHS catalog for tile %s; skipping NIR bands.", tile)
        return mtbl
    vhscat = Table.read(path)
    if len(vhscat) == 0:
        log.info("VHS catalog for tile %s is empty; skipping.", tile)
        return mtbl
    for col, off in _VHS_VEGA_TO_AB.items():
        if col in vhscat.colnames:
            vhscat[col] = vhscat[col] + off
    return matching(
        mtbl, vhscat,
        inra=np.array(mtbl[cfg.ra_column]), indec=np.array(mtbl[cfg.dec_column]),
        refra=np.array(vhscat["RAJ2000"]), refdec=np.array(vhscat["DEJ2000"]),
        sep=cfg.match_radius_arcsec, join_type="left", duplicate="closest",
        ref_prefix="vhs_",
    )


def _match_galex(mtbl: Table, cfg: VACConfig, tile: str, tile_info) -> Table:
    path = cfg.galex_path(tile)
    if not os.path.exists(path):
        ensure_external_catalog("galex", tile, tile_info, path, cfg)
    if not os.path.exists(path):
        log.info("No GALEX catalog for tile %s; skipping UV bands.", tile)
        return mtbl
    galexcat = Table.read(path)
    if len(galexcat) == 0:
        log.info("GALEX catalog for tile %s is empty; skipping.", tile)
        return mtbl
    return matching(
        mtbl, galexcat,
        inra=np.array(mtbl[cfg.ra_column]), indec=np.array(mtbl[cfg.dec_column]),
        refra=np.array(galexcat["RAJ2000"]), refdec=np.array(galexcat["DEJ2000"]),
        sep=cfg.match_radius_arcsec, join_type="left", duplicate="closest",
        ref_prefix="galex_",
    )


def _dedup_brightest(mtbl: Table, cfg: VACConfig) -> Table:
    """Keep one row per REGALADE source: the brightest 7DS dedup band.

    Adds an ``nmatch`` column counting how many catalog rows matched each
    REGALADE source. When the dedup magnitude column is missing or
    ``dedup_by_brightness`` is False, the first occurrence is kept.
    """
    name_col = f"regalade_{cfg.regalade_name_key}"
    names = np.asarray(mtbl[name_col])
    unique_names = np.unique(names)

    dedup_col = f"auto_mag_{cfg.medium_bands[len(cfg.medium_bands) // 2]}"  # ~m650
    have_mag = cfg.dedup_by_brightness and dedup_col in mtbl.colnames
    mags = np.asarray(mtbl[dedup_col]) if have_mag else None

    keep_indices: list[int] = []
    nmatch_values = np.zeros(len(mtbl), dtype=int)
    for name in unique_names:
        idx = np.where(names == name)[0]
        nmatch_values[idx] = len(idx)
        if have_mag:
            keep_indices.append(int(idx[np.argmin(mags[idx])]))
        else:
            keep_indices.append(int(idx[0]))

    mtbl["nmatch"] = nmatch_values
    return mtbl[np.array(keep_indices)]


__all__ = ["build_galaxy_catalog"]
