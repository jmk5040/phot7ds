"""
Flux input-catalog assembly for EAzY / FAST++.

Converts the matched galaxy magnitudes into a Galactic-extinction-corrected
flux table in the EAzY/FAST++ ``.cat`` format (AB zeropoint 25.0), and
writes the companion ``target_ids.fits`` used to merge SED-fitting outputs
back onto the original catalog.
"""
from __future__ import annotations

import logging
import os

import numpy as np
from astropy.io import ascii as ascii_io
from astropy.table import Table

from ..photconv import mag_to_flux, mag_to_flux_err
from .config import VACConfig

log = logging.getLogger(__name__)

# Resolver: EAzY filter name -> (reference column, error column or None).
# ``None`` for the catalog (7DS/SDSS) bands, which are resolved from the
# aperture magnitude columns instead.
_EXTERNAL_COLUMNS: dict[str, tuple[str, str | None]] = {
    "f_W1": ("regalade_W1mag", None),
    "f_W2": ("regalade_W2mag", None),
    "f_VHS_J": ("vhs_Jpmag", "vhs_e_Jpmag"),
    "f_VHS_H": ("vhs_Hpmag", "vhs_e_Hpmag"),
    "f_VHS_K": ("vhs_Kspmag", "vhs_e_Kspmag"),
    "f_FUV": ("galex_FUVmag", "galex_e_FUVmag"),
    "f_NUV": ("galex_NUVmag", "galex_e_NUVmag"),
}


def _tile_center(tile_info) -> tuple[float, float]:
    """Tile center RA/Dec from explicit columns or the four corners."""
    def _get(key):
        try:
            val = tile_info[key]
        except (KeyError, IndexError, ValueError):
            return None
        arr = np.atleast_1d(val)
        return float(arr[0]) if len(arr) else None

    ra, dec = _get("ra"), _get("dec")
    if ra is not None and dec is not None:
        return ra, dec
    ras = [_get(f"ra{i}") for i in (1, 2, 3, 4)]
    decs = [_get(f"dec{i}") for i in (1, 2, 3, 4)]
    if all(v is not None for v in ras + decs):
        return float(np.mean(ras)), float(np.mean(decs))
    raise KeyError("tile_info has neither 'ra'/'dec' nor 'ra1..ra4'/'dec1..dec4'.")


def _central_wavelengths(cfg: VACConfig, filters: list[str]) -> dict[str, float]:
    """Look up each filter's central wavelength via the translate + info files."""
    info = open(cfg.filters_res_info).readlines()
    translate = ascii_io.read(cfg.translate_file)
    lambda_c: dict[str, float] = {}
    for f in filters:
        rows = translate[translate["filter"] == f]
        if len(rows) == 0:
            log.warning("Filter %r not defined in translate file; skipping.", f)
            continue
        linenum = int(str(rows["lines"][0])[1:])
        lambda_c[f] = float(info[linenum - 1].split("lambda_c= ")[-1].split(" ")[0])
    return lambda_c


def _extinction_by_filter(
    cfg: VACConfig, lambda_c: dict[str, float], ra: float, dec: float
) -> dict[str, float]:
    """Galactic extinction (mag) per filter at the tile center (SFD + F99)."""
    import extinction
    import sfdmap

    ebv = sfdmap.ebv(ra, dec, mapdir=str(cfg.sfd_path))
    ebv = float(np.atleast_1d(ebv)[0])
    return {
        band: float(round(extinction.fitzpatrick99(np.array([lam]), 3.1 * ebv)[0], 3))
        for band, lam in lambda_c.items()
    }


def build_flux_catalog(
    magtbl: Table,
    cfg: VACConfig,
    tile: str,
    tile_info,
) -> tuple[Table, Table]:
    """Build the EAzY/FAST++ flux catalog and the target-id table.

    Parameters
    ----------
    magtbl
        Deduped galaxy catalog from :func:`build_galaxy_catalog`.
    cfg
        :class:`VACConfig`.
    tile
        Tile identifier (used in the output file names).
    tile_info
        Single-row tile table (for the extinction sightline center).

    Returns
    -------
    (flux_table, id_table)
        ``flux_table`` is the cleaned EAzY/FAST++ input catalog (also
        written to both the photo-z and SED-fit directories as ``.cat``).
        ``id_table`` carries the identifiers needed to merge results back.
    """
    filters = cfg.filters()
    lambda_c = _central_wavelengths(cfg, filters)
    centra, centdec = _tile_center(tile_info)
    lambda_ext = _extinction_by_filter(cfg, lambda_c, centra, centdec)

    aperture = cfg.aperture
    error_margin = cfg.error_margin

    fluxtbl = Table()
    fluxtbl["#id"] = np.arange(1, len(magtbl) + 1)

    for band in filters:
        if band not in lambda_ext:
            continue
        ext = lambda_ext[band]
        ecol = band.replace("f_", "e_")
        magcol = f"{aperture}_mag_{band.split('_')[-1]}"
        if magcol in magtbl.colnames:
            errcol = magcol.replace("_mag_", "_mag_err_")
            fluxtbl[band] = mag_to_flux(magtbl[magcol] - ext)
            fluxtbl[ecol] = mag_to_flux_err(magtbl[magcol], magtbl[errcol] + error_margin)
        elif band in _EXTERNAL_COLUMNS:
            mcol, errc = _EXTERNAL_COLUMNS[band]
            if mcol not in magtbl.colnames:
                log.warning("Column %s for %s missing; skipping band.", mcol, band)
                continue
            fluxtbl[band] = mag_to_flux(magtbl[mcol] - ext)
            if errc and errc in magtbl.colnames:
                err = magtbl[errc] + error_margin
            else:
                err = np.full(len(magtbl), error_margin)
            fluxtbl[ecol] = mag_to_flux_err(magtbl[mcol], err)
        else:
            log.warning("No column mapping for filter %s; skipping.", band)

    clean_tbl, mask = _apply_validity_cut(fluxtbl, cfg)
    log.info("Flux catalog: %d/%d sources pass the filter-coverage cut.",
             len(clean_tbl), len(fluxtbl))

    # Write the EAzY and FAST++ input catalogs.
    photoz_dir = cfg.photoz_dir(tile)
    sedfit_dir = cfg.sedfit_dir(tile)
    os.makedirs(photoz_dir, exist_ok=True)
    os.makedirs(sedfit_dir, exist_ok=True)
    cat_name = f"{tile}_{cfg.detection_ref}_phot.cat"
    clean_tbl.write(photoz_dir / cat_name, format="ascii", delimiter="\t", overwrite=True)
    clean_tbl.write(sedfit_dir / cat_name, format="ascii", delimiter="\t", overwrite=True)

    id_table = _build_id_table(magtbl, mask, cfg)
    id_name = f"{tile}_{cfg.detection_ref}_target_ids.fits"
    id_table.write(photoz_dir / id_name, format="fits", overwrite=True)
    id_table.write(sedfit_dir / id_name, format="fits", overwrite=True)

    return clean_tbl, id_table


def _apply_validity_cut(fluxtbl: Table, cfg: VACConfig) -> tuple[Table, np.ndarray]:
    """Keep sources with at least ``min_filter_fraction`` measured filters."""
    meas_cols = [c for c in fluxtbl.colnames if c.startswith(("f_", "e_"))]
    n_filters = len(meas_cols) // 2
    least_n = int(n_filters * cfg.min_filter_fraction)
    valcut = len(meas_cols) - least_n * 2  # max allowed (-99) sentinels

    mask = np.ones(len(fluxtbl), dtype=bool)
    for i, row in enumerate(fluxtbl):
        n_missing = sum(row[c] == -99 for c in meas_cols)
        mask[i] = n_missing <= valcut
    return fluxtbl[mask], mask


def _build_id_table(magtbl: Table, mask: np.ndarray, cfg: VACConfig) -> Table:
    """Identifiers for merging EAzY/FAST++ outputs back to the catalog."""
    idtbl = Table()
    if cfg.id_column in magtbl.colnames:
        idtbl[cfg.id_column] = magtbl[cfg.id_column][mask]
    carry = [
        f"regalade_{cfg.regalade_name_key}", "regalade_IdCat",
        "regalade_RAJ2000", "regalade_DEJ2000",
        "regalade_z", "regalade_Dist", "regalade_r_DistInput",
    ]
    for col in carry:
        if col in magtbl.colnames:
            idtbl[col] = magtbl[col][mask]
    for col in magtbl.colnames:
        if "flag" in col.lower():
            idtbl[col] = magtbl[col][mask]
    return idtbl


__all__ = ["build_flux_catalog"]
