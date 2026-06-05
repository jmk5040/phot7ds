"""
Photometric redshifts with eazy-py.

Replaces the legacy compiled-``eazy`` workflow (writing ``zphot.param`` and
shelling out) with the pure-Python :class:`eazy.photoz.PhotoZ`. The r-band
(or K-band) magnitude prior is auto-selected from the available filters, as
in the original pipeline.

The point-estimate redshift is written to a FAST++-compatible ``.zout``
ASCII file so the SED-fitting stage can consume it.
"""
from __future__ import annotations

import logging
import os

import numpy as np
from astropy.io import ascii as ascii_io
from astropy.table import Table

from .config import VACConfig

log = logging.getLogger(__name__)

# eazy-py point-estimate column used downstream (and exported as ``z_phot``).
ZPHOT_COLUMN = "z_phot"


def _select_prior(cfg: VACConfig, cat_path: str) -> tuple[bool, int | None, bool]:
    """Pick an r-band (else K-band) prior filter from the catalog columns.

    Returns ``(apply_prior, prior_filter_id, is_K_prior)``.
    """
    translate = ascii_io.read(cfg.translate_file)
    columns = ascii_io.read(cat_path).colnames

    def _filt_id(colname: str) -> int | None:
        rows = translate[translate["filter"] == colname]
        if len(rows) == 0:
            return None
        return int(str(rows["lines"][0])[1:])

    r_bands = [c for c in columns if c.startswith("f_") and c.endswith("r") and _filt_id(c)]
    k_bands = [c for c in columns if c.startswith("f_") and c.endswith("K") and _filt_id(c)]
    if r_bands:
        return True, _filt_id(r_bands[0]), False
    if k_bands:
        return True, _filt_id(k_bands[0]), True
    return False, None, False


def _build_eazy_params(cfg: VACConfig, tile: str, apply_prior: bool,
                       prior_id: int | None, is_K: bool) -> dict:
    """Assemble the eazy-py parameter dict (overridable via cfg.eazy_params)."""
    lib = cfg.lib_path
    prior_file = lib / "templates" / ("prior_K_extend.dat" if is_K else "prior_R_extend.dat")
    params = {
        "FILTERS_RES": str(cfg.filters_res),
        "FILTER_FORMAT": 1,
        "TEMPLATES_FILE": str(lib / "templates" / "eazy_v1.2_dusty.spectra.param"),
        "TEMPLATE_COMBOS": "a",
        "WAVELENGTH_FILE": str(lib / "templates" / "EAZY_v1.1_lines" / "lambda_v1.1.def"),
        "TEMP_ERR_FILE": str(lib / "templates" / "TEMPLATE_ERROR.eazy_v1.0"),
        "TEMP_ERR_A2": 0.50,
        "SYS_ERR": 0.00,
        "APPLY_IGM": "y",
        "CATALOG_FILE": f"{tile}_{cfg.detection_ref}_phot.cat",
        "MAGNITUDES": "n",
        "NOT_OBS_THRESHOLD": -90,
        "N_MIN_COLORS": 5,
        "OUTPUT_DIRECTORY": "OUTPUT",
        "MAIN_OUTPUT_FILE": f"{tile}_{cfg.detection_ref}_phot",
        "APPLY_PRIOR": "y" if apply_prior else "n",
        "PRIOR_FILE": str(prior_file),
        "PRIOR_ABZP": 25.00,
        "Z_MIN": cfg.z_min,
        "Z_MAX": cfg.z_max,
        "Z_STEP": cfg.z_step,
        "Z_STEP_TYPE": 1,
        "H0": 70.0,
        "OMEGA_M": 0.27,
        "OMEGA_L": 0.73,
    }
    if apply_prior and prior_id is not None:
        params["PRIOR_FILTER"] = int(prior_id)
    params.update(cfg.eazy_params)
    return params


def run_eazy(cfg: VACConfig, tile: str) -> Table:
    """Run eazy-py photo-z for one tile and return the zout table.

    Also writes a FAST++-compatible ``{tile}_{ref}_phot.zout`` (ASCII,
    commented header with at least ``id`` and ``z_phot``) into the SED-fit
    directory.
    """
    from eazy.photoz import PhotoZ

    photoz_dir = cfg.photoz_dir(tile)
    cat_path = str(photoz_dir / f"{tile}_{cfg.detection_ref}_phot.cat")
    if not os.path.exists(cat_path):
        raise FileNotFoundError(f"EAzY input catalog not found: {cat_path}")

    apply_prior, prior_id, is_K = _select_prior(cfg, cat_path)
    log.info("EAzY prior: apply=%s filter_id=%s K=%s", apply_prior, prior_id, is_K)
    params = _build_eazy_params(cfg, tile, apply_prior, prior_id, is_K)

    # eazy-py resolves relative CATALOG_FILE/OUTPUT against the CWD.
    cwd = os.getcwd()
    os.makedirs(photoz_dir / "OUTPUT", exist_ok=True)
    try:
        os.chdir(photoz_dir)
        pz = PhotoZ(
            param_file=None,
            translate_file=str(cfg.translate_file),
            zeropoint_file=None,
            params=params,
            n_proc=cfg.n_proc,
        )
        pz.fit_catalog(n_proc=cfg.n_proc, prior=apply_prior, beta_prior=apply_prior)
        zout, _ = pz.standard_output(
            prior=apply_prior, beta_prior=apply_prior, save_fits=True
        )
    finally:
        os.chdir(cwd)

    zout = Table(zout)
    _write_fastpp_zout(cfg, tile, zout)
    return zout


def _write_fastpp_zout(cfg: VACConfig, tile: str, zout: Table) -> None:
    """Write a minimal ASCII .zout (id + z_phot) for FAST++ to ingest."""
    sedfit_dir = cfg.sedfit_dir(tile)
    os.makedirs(sedfit_dir, exist_ok=True)
    out = Table()
    out["id"] = (
        zout["id"] if "id" in zout.colnames else np.arange(1, len(zout) + 1)
    )
    zcol = ZPHOT_COLUMN if ZPHOT_COLUMN in zout.colnames else _first_z_column(zout)
    out["z_phot"] = np.asarray(zout[zcol], dtype=float)
    out_path = sedfit_dir / f"{tile}_{cfg.detection_ref}_phot.zout"
    out.write(out_path, format="ascii.commented_header", overwrite=True)
    log.info("Wrote FAST++ photo-z input: %s (z column %r)", out_path, zcol)


def _first_z_column(zout: Table) -> str:
    """Fall back to a sensible redshift column if ``z_phot`` is absent."""
    for cand in ("z_phot", "z_map", "z_min_risk", "z_raw_chi2", "z_peak"):
        if cand in zout.colnames:
            return cand
    raise KeyError(f"No usable redshift column in zout: {zout.colnames}")


__all__ = ["run_eazy", "ZPHOT_COLUMN"]
