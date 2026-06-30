"""
Photometric redshifts with the compiled EAzY binary.

This is the default photo-z backend (``VACConfig.photoz_engine == "binary"``).
It mirrors the legacy ``RIS_catalog_fastpp.py`` workflow: write a
``zphot.param`` from the LIB template plus an override dict, drop a
``zphot.translate`` next to it, and shell out to the compiled ``eazy``
executable. The binary is dramatically faster than the pure-Python
:func:`phot7ds.vac.photoz.run_eazy` for the 7DS medium-band filter set
(seconds vs. an effectively unbounded eazy-py ``TemplateGrid`` build).

The binary's native ``.zout`` already carries the FAST++-ready columns
(``z_m2``/``z_a`` plus ``l68/u68``, ``l95/u95``, ``l99/u99``), so it is
copied verbatim into the SED-fit directory for FAST++ to consume. The
redshift column handed to FAST++ is ``z_m2`` when the prior was applied and
``z_a`` otherwise.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess

import numpy as np
from astropy.io import ascii as ascii_io
from astropy.table import Table

from .config import VACConfig
from .photoz import _select_prior

log = logging.getLogger(__name__)


def _tail(path, n: int = 30) -> str:
    """Return the last ``n`` lines of a text file (best-effort)."""
    try:
        with open(path) as fh:
            return "".join(fh.readlines()[-n:])
    except OSError:
        return ""


def _build_param_overrides(cfg: VACConfig, tile: str, apply_prior: bool,
                           prior_id: int | None) -> dict[str, str]:
    """Assemble the EAzY-binary ``zphot.param`` overrides (string values)."""
    lib = cfg.lib_path
    templates = lib / "templates"
    overrides: dict[str, str] = {
        # Filters
        "FILTERS_RES": str(cfg.filters_res),
        "FILTER_FORMAT": "1",
        "SMOOTH_FILTERS": "n",
        "SMOOTH_SIGMA": "100.",
        # Templates
        "TEMPLATES_FILE": str(templates / "eazy_v1.2_dusty.spectra.param"),
        "TEMPLATE_COMBOS": "a",
        "NMF_TOLERANCE": "1.e-4",
        "WAVELENGTH_FILE": str(templates / "EAZY_v1.1_lines" / "lambda_v1.1.def"),
        "TEMP_ERR_FILE": str(templates / "TEMPLATE_ERROR.eazy_v1.0"),
        "TEMP_ERR_A2": "0.50",
        "SYS_ERR": "0.00",
        "APPLY_IGM": "y",
        "LAF_FILE": str(templates / "LAFcoeff.txt"),
        "DLA_FILE": str(templates / "DLAcoeff.txt"),
        "SCALE_2175_BUMP": "0.00",
        "DUMP_TEMPLATE_CACHE": "n",
        "USE_TEMPLATE_CACHE": "n",
        "CACHE_FILE": "photz.tempfilt",
        # Input
        "CATALOG_FILE": f"{tile}_{cfg.detection_ref}_phot.cat",
        "MAGNITUDES": "n",
        "NOT_OBS_THRESHOLD": "-90",
        "N_MIN_COLORS": "5",
        # Output
        "OUTPUT_DIRECTORY": "OUTPUT",
        "MAIN_OUTPUT_FILE": f"{tile}_{cfg.detection_ref}_phot",
        "PRINT_ERRORS": "y",
        "CHI2_SCALE": "1.0",
        "VERBOSE_LOG": "y",
        "OBS_SED_FILE": "n",
        "TEMP_SED_FILE": "n",
        "POFZ_FILE": "n",
        "BINARY_OUTPUT": "y",
        # Redshift / magnitude prior
        "APPLY_PRIOR": "y" if apply_prior else "n",
        "PRIOR_FILE": str(cfg.prior_path),
        "PRIOR_ABZP": "25.00",
        # Redshift grid
        "FIX_ZSPEC": "n",
        "Z_MIN": str(cfg.z_min),
        "Z_MAX": str(cfg.z_max),
        "Z_STEP": str(cfg.z_step),
        "Z_STEP_TYPE": "1",
        # Zeropoint offsets (never iterate; matches the legacy fast workflow)
        "GET_ZP_OFFSETS": "n",
        "ZP_OFFSET_TOL": "1.e-4",
        # Rest-frame colors
        "REST_FILTERS": "---",
        "RF_PADDING": "1000.",
        "RF_ERRORS": "n",
        "Z_COLUMN": "z_peak",
        "USE_ZSPEC_FOR_REST": "y",
        "READ_ZBIN": "n",
        # Cosmology
        "H0": "70.0",
        "OMEGA_M": "0.27",
        "OMEGA_L": "0.73",
    }
    if apply_prior and prior_id is not None:
        overrides["PRIOR_FILTER"] = str(int(prior_id))
    # Allow user overrides (cfg.eazy_params), stringified for the binary.
    for key, value in cfg.eazy_params.items():
        overrides[key] = str(value)
    return overrides


def _write_zphot_param(cfg: VACConfig, dest_path: str,
                       overrides: dict[str, str]) -> None:
    """Write ``zphot.param`` from the LIB template, applying overrides.

    Keeps every non-overridden line (and inline comments) verbatim, so the
    file stays human-readable and any LIB-specific defaults survive.
    """
    with open(cfg.eazy_param_template) as fh:
        contents = fh.readlines()

    written: set[str] = set()
    out_lines: list[str] = []
    for line in contents:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            out_lines.append(line)
            continue
        key = stripped.split()[0]
        if key in overrides:
            inline_comment = ""
            if "#" in line:
                inline_comment = "  #" + line.split("#", 1)[1].rstrip("\n")
            out_lines.append(f"{key:<20} {overrides[key]}{inline_comment}\n")
            written.add(key)
        else:
            out_lines.append(line)

    for key in set(overrides) - written:
        out_lines.append(f"{key:<20} {overrides[key]}\n")

    with open(dest_path, "w") as fh:
        fh.writelines(out_lines)


def _read_binary_zout(zout_path: str) -> Table:
    """Read the EAzY-binary ``.zout`` (``# col1 col2 ...`` header + data)."""
    with open(zout_path) as fh:
        first = fh.readline()
    names = first.lstrip("#").split()
    return Table(ascii_io.read(zout_path, names=names, comment="#"))


def run_eazy_binary(cfg: VACConfig, tile: str) -> tuple[Table, dict]:
    """Run the compiled EAzY binary for one tile.

    Returns ``(zout_table, eazy_info)`` with the same contract as
    :func:`phot7ds.vac.photoz.run_eazy`: ``eazy_info`` carries the prior
    choice and the redshift column name (``z_m2``/``z_a``) consumed by
    FAST++, and the binary's native ``.zout`` is copied into the SED-fit
    directory.
    """
    photoz_dir = cfg.photoz_dir(tile)
    cat_path = str(photoz_dir / f"{tile}_{cfg.detection_ref}_phot.cat")
    if not os.path.exists(cat_path):
        raise FileNotFoundError(f"EAzY input catalog not found: {cat_path}")
    if not cfg.eazy_bin_ok():
        log.warning("No EAzY executable at %s; the photo-z stage cannot run "
                    "with photoz_engine='binary'.", cfg.eazy_bin)
    assert cfg.eazy_bin_ok(), (
        f"EAzY executable not found / not executable at {cfg.eazy_bin}. Build "
        "it (or set VACConfig.eazy_bin), or set photoz_engine='eazy-py'."
    )

    apply_prior, prior_id, prior_name = _select_prior(cfg, cat_path)
    zphot_column = "z_m2" if apply_prior else "z_a"
    log.info("EAzY (binary) prior: apply=%s band=%s filter_id=%s -> column %s",
             apply_prior, prior_name, prior_id, zphot_column)

    overrides = _build_param_overrides(cfg, tile, apply_prior, prior_id)
    # The binary reads zphot.translate (filter-name -> FILTER.RES id) from cwd.
    shutil.copyfile(cfg.translate_file, photoz_dir / "zphot.translate")
    _write_zphot_param(cfg, str(photoz_dir / "zphot.param"), overrides)

    out_subdir = photoz_dir / "OUTPUT"
    os.makedirs(out_subdir, exist_ok=True)

    # Stream EAzY stdout+stderr to a live, preserved log file (the binary can
    # run for a couple of minutes; buffering in memory would hide progress).
    run_log = photoz_dir / f"{tile}_{cfg.detection_ref}_eazy.log"
    cwd = os.getcwd()
    try:
        os.chdir(photoz_dir)
        log.info("Running EAzY binary: %s (cwd=%s); output -> %s",
                 cfg.eazy_bin, photoz_dir, run_log)
        with open(run_log, "w") as fh:
            proc = subprocess.run(
                [str(cfg.eazy_bin)], cwd=str(photoz_dir),
                stdout=fh, stderr=subprocess.STDOUT, text=True,
            )
    finally:
        os.chdir(cwd)
    if proc.returncode != 0:
        raise RuntimeError(
            f"EAzY binary failed (exit {proc.returncode}). See {run_log}\n"
            f"--- last lines ---\n{_tail(run_log)}"
        )

    zout_path = out_subdir / f"{tile}_{cfg.detection_ref}_phot.zout"
    if not os.path.exists(zout_path):
        raise FileNotFoundError(
            f"EAzY produced no .zout: {zout_path}\nstdout:\n{proc.stdout}"
        )
    zout = _read_binary_zout(str(zout_path))
    if zphot_column not in zout.colnames:
        raise KeyError(
            f"Expected redshift column {zphot_column!r} absent from EAzY zout "
            f"(columns: {zout.colnames}). Was the prior applied as expected?"
        )

    # FAST++ consumes the .zout from the SED-fit dir; the binary's native
    # output already has z_m2/z_a + l68/u68/l95/u95/l99/u99, so copy it as-is.
    sedfit_dir = cfg.sedfit_dir(tile)
    os.makedirs(sedfit_dir, exist_ok=True)
    fastpp_zout = sedfit_dir / f"{tile}_{cfg.detection_ref}_phot.zout"
    shutil.copyfile(zout_path, fastpp_zout)
    log.info("Wrote FAST++ photo-z input: %s (z column %r)",
             fastpp_zout, zphot_column)

    eazy_info = {
        "engine": "binary",
        "binary": str(cfg.eazy_bin),
        "apply_prior": apply_prior,
        "prior_band": cfg.prior_band,
        "prior_filter": prior_name,
        "prior_filter_id": prior_id,
        "prior_file": str(cfg.prior_path) if apply_prior else None,
        "zphot_column": zphot_column,
        "n_targets": len(zout),
        "run_log": str(run_log),
        "params": overrides,
    }
    return zout, eazy_info


__all__ = ["run_eazy_binary"]
