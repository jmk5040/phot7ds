"""
Photometric redshifts with eazy-py.

Replaces the legacy compiled-``eazy`` workflow (writing ``zphot.param`` and
shelling out) with the pure-Python :class:`eazy.photoz.PhotoZ`. The default
magnitude prior is the 7DS ``m625`` prior (``prior_m6250_extend.dat``); when
the prior band's flux is absent from the catalog the run proceeds without a
prior and a warning is logged.

The point-estimate redshift is written to a FAST++-compatible ``.zout``
ASCII file so the SED-fitting stage can consume it. The redshift column is
named ``z_m2`` when the prior was applied and ``z_a`` otherwise; FAST++ is
pointed at that column via ``NAME_ZPHOT``.
"""
from __future__ import annotations

import logging
import os

import numpy as np
from astropy.io import ascii as ascii_io
from astropy.table import Table

from .config import VACConfig

log = logging.getLogger(__name__)

# eazy-py point-estimate column produced by standard_output().
ZPHOT_COLUMN = "z_phot"


def _select_prior(cfg: VACConfig, cat_path: str) -> tuple[bool, int | None, str]:
    """Decide whether to apply the (default m625) magnitude prior.

    The prior is applied only when the prior band's flux column
    (``f_7DS_<prior_band>``) is present in the EAzY catalog and is defined
    in the translate file. Otherwise a warning is logged and the run
    proceeds without a prior.

    Returns ``(apply_prior, prior_filter_id, prior_filter_name)``.
    """
    translate = ascii_io.read(cfg.translate_file)
    columns = ascii_io.read(cat_path).colnames
    prior_name = f"f_7DS_{cfg.prior_band}"

    rows = translate[translate["filter"] == prior_name]
    prior_id = int(str(rows["lines"][0])[1:]) if len(rows) else None

    if prior_name in columns and prior_id is not None:
        return True, prior_id, prior_name

    log.warning(
        "Prior band %s (%s_mag_%s) absent from catalog; proceeding WITHOUT a "
        "prior (redshift column will be z_a).",
        prior_name, cfg.aperture, cfg.prior_band,
    )
    return False, None, prior_name


def _build_eazy_params(cfg: VACConfig, tile: str, apply_prior: bool,
                       prior_id: int | None) -> dict:
    """Assemble the eazy-py parameter dict (overridable via cfg.eazy_params)."""
    lib = cfg.lib_path
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
        "PRIOR_FILE": str(cfg.prior_path),
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


def run_eazy(cfg: VACConfig, tile: str) -> tuple[Table, dict]:
    """Run eazy-py photo-z for one tile.

    Returns ``(zout_table, eazy_info)``. ``eazy_info`` is a metadata dict
    (prior choice, redshift column name, parameters, target count) used for
    the run log and to tell FAST++ which redshift column to consume.

    Also writes a FAST++-compatible ``{tile}_{ref}_phot.zout`` (ASCII,
    commented header) into the SED-fit directory; the redshift column is
    ``z_m2`` if the prior was applied, else ``z_a``.
    """
    from eazy.photoz import PhotoZ

    photoz_dir = cfg.photoz_dir(tile)
    cat_path = str(photoz_dir / f"{tile}_{cfg.detection_ref}_phot.cat")
    if not os.path.exists(cat_path):
        raise FileNotFoundError(f"EAzY input catalog not found: {cat_path}")

    apply_prior, prior_id, prior_name = _select_prior(cfg, cat_path)
    zphot_column = "z_m2" if apply_prior else "z_a"
    log.info("EAzY prior: apply=%s band=%s filter_id=%s -> redshift column %s",
             apply_prior, prior_name, prior_id, zphot_column)
    params = _build_eazy_params(cfg, tile, apply_prior, prior_id)

    # eazy-py's two stages take *different* serial sentinels: TemplateGrid is
    # serial for n_proc<0, fit_catalog for n_proc==0. Map the single
    # cfg.eazy_n_proc accordingly (<=0 -> run both serially, avoiding the
    # multiprocessing template-grid deadlock/timeout).
    if cfg.eazy_n_proc > 0:
        grid_n_proc = fit_n_proc = cfg.eazy_n_proc
    else:
        grid_n_proc, fit_n_proc = -1, 0
    log.info("EAzY parallelism: template-grid n_proc=%d, fit n_proc=%d",
             grid_n_proc, fit_n_proc)

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
            n_proc=grid_n_proc,
        )
        pz.fit_catalog(n_proc=fit_n_proc, prior=apply_prior, beta_prior=apply_prior)
        zout, _ = pz.standard_output(
            prior=apply_prior,
            beta_prior=apply_prior,
            save_fits=True,
            # percentiles needed for the FAST++ l68/u68, l95/u95, l99/u99 cols
            percentile_limits=[0.5, 2.5, 16, 50, 84, 97.5, 99.5],
        )
    finally:
        os.chdir(cwd)

    zout = Table(zout)
    _write_fastpp_zout(cfg, tile, zout, zphot_column)

    eazy_info = {
        "apply_prior": apply_prior,
        "prior_band": cfg.prior_band,
        "prior_filter": prior_name,
        "prior_filter_id": prior_id,
        "prior_file": str(cfg.prior_path) if apply_prior else None,
        "zphot_column": zphot_column,
        "n_targets": len(zout),
        "grid_n_proc": grid_n_proc,
        "fit_n_proc": fit_n_proc,
        "params": params,
    }
    return zout, eazy_info


# FAST++ confidence-interval column -> eazy-py percentile column.
_INTERVAL_MAP = {
    "l68": "z160", "u68": "z840",
    "l95": "z025", "u95": "z975",
    "l99": "z005", "u99": "z995",
}


def _write_fastpp_zout(cfg: VACConfig, tile: str, zout: Table,
                       zphot_column: str) -> None:
    """Write an ASCII .zout for FAST++ with the redshift + confidence ints.

    The redshift column is named ``zphot_column`` (``z_m2`` with a prior,
    ``z_a`` without) so FAST++ ``NAME_ZPHOT`` can point at it. FAST++ (with
    ``BEST_AT_ZPHOT``) also requires the confidence intervals ``l68/u68``
    (and probes ``l95/u95``, ``l99/u99``); these are taken from eazy-py's
    percentile columns, filling any missing ones with the point estimate.
    """
    sedfit_dir = cfg.sedfit_dir(tile)
    os.makedirs(sedfit_dir, exist_ok=True)
    out = Table()
    out["id"] = (
        zout["id"] if "id" in zout.colnames else np.arange(1, len(zout) + 1)
    )
    zcol = ZPHOT_COLUMN if ZPHOT_COLUMN in zout.colnames else _first_z_column(zout)
    zphot = np.asarray(zout[zcol], dtype=float)
    out[zphot_column] = zphot
    for fastpp_col, eazy_col in _INTERVAL_MAP.items():
        if eazy_col in zout.colnames:
            out[fastpp_col] = np.asarray(zout[eazy_col], dtype=float)
        else:
            log.warning("eazy zout missing %s; filling %s with point estimate.",
                        eazy_col, fastpp_col)
            out[fastpp_col] = zphot
    out_path = sedfit_dir / f"{tile}_{cfg.detection_ref}_phot.zout"
    out.write(out_path, format="ascii.commented_header", overwrite=True)
    log.info("Wrote FAST++ photo-z input: %s (z column %r, with l/u intervals)",
             out_path, zphot_column)


def _first_z_column(zout: Table) -> str:
    """Fall back to a sensible redshift column if ``z_phot`` is absent."""
    for cand in ("z_phot", "z_map", "z_min_risk", "z_raw_chi2", "z_peak"):
        if cand in zout.colnames:
            return cand
    raise KeyError(f"No usable redshift column in zout: {zout.colnames}")


__all__ = ["run_eazy", "ZPHOT_COLUMN", "_select_prior"]
