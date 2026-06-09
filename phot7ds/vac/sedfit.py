"""
SED fitting with FAST++.

FAST++ is an external compiled binary, so this module writes its
``.param``/``.translate`` inputs from the LIB template (plus override
dict), runs it via :func:`subprocess.run`, and parses the resulting
``.fout`` into an astropy table. It fails gracefully with a clear message
if the binary or the photo-z input is missing.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess

from astropy.io import ascii as ascii_io
from astropy.table import Table

from .config import VACConfig

log = logging.getLogger(__name__)


def _format_value(value) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(str(v) for v in value) + "]"
    text = str(value)
    if text.startswith("[") or text.startswith("'"):
        return text
    return f"'{text}'"


def _build_overrides(cfg: VACConfig, catalog: str, name_zphot: str) -> dict:
    """FAST++ parameter overrides applied on top of the LIB template."""
    share = cfg.fastpp_share
    overrides = {
        "CATALOG": catalog,
        "AB_ZEROPOINT": 25.0,
        "FILTERS_RES": str(cfg.filters_res),
        "FILTER_FORMAT": 1,
        "TEMP_ERR_FILE": str(share / "TEMPLATE_ERROR.fast.v0.2"),
        "LIBRARY_DIR": str(cfg.fastpp_libraries) + "/",
        "NAME_ZPHOT": name_zphot,
        "Z_MIN": cfg.z_min,
        "Z_MAX": cfg.z_max,
        "Z_STEP": cfg.z_step,
        "PARALLEL": "sources",
        "N_THREAD": cfg.n_proc,
    }
    overrides.update(cfg.fastpp_params)
    return overrides


def _write_param(cfg: VACConfig, tile: str, catalog: str, dest_dir,
                 overrides: dict) -> str:
    """Write the FAST++ .param file from the LIB template + overrides."""
    template_lines = open(cfg.fastpp_param_template).readlines()

    written: set[str] = set()
    out_lines: list[str] = []
    for line in template_lines:
        stripped = line.lstrip()
        replaced = False
        for key, value in overrides.items():
            if stripped.startswith(key) and stripped[len(key):len(key) + 1] in (" ", "\t", "="):
                comment = ""
                if "#" in line:
                    comment = "  #" + line.split("#", 1)[1].rstrip("\n")
                out_lines.append(f"{key:<18} = {_format_value(value)}{comment}\n")
                written.add(key)
                replaced = True
                break
        if not replaced:
            out_lines.append(line)

    for key in set(overrides) - written:
        out_lines.append(f"{key:<18} = {_format_value(overrides[key])}\n")

    param_path = os.path.join(dest_dir, f"{catalog}.param")
    with open(param_path, "w") as fh:
        fh.writelines(out_lines)
    return param_path


def run_fastpp(cfg: VACConfig, tile: str, *, name_zphot: str = "z_phot") -> tuple[Table, dict]:
    """Run FAST++ SED fitting for one tile.

    Parameters
    ----------
    name_zphot
        Redshift column FAST++ should read from the ``.zout`` (``z_m2``
        when an eazy prior was applied, else ``z_a``).

    Returns ``(fout_table, fastpp_info)`` where ``fastpp_info`` carries the
    applied parameter overrides for the run log.

    Requires the ``.cat`` and ``.zout`` written by the flux/photo-z stages
    in :meth:`VACConfig.sedfit_dir`.
    """
    if not os.path.exists(cfg.fastpp_bin):
        raise FileNotFoundError(
            f"FAST++ binary not found at {cfg.fastpp_bin}; install it or run "
            "with do_sedfit=False."
        )
    sedfit_dir = cfg.sedfit_dir(tile)
    os.makedirs(sedfit_dir, exist_ok=True)
    catalog = f"{tile}_{cfg.detection_ref}_phot"
    cat_path = sedfit_dir / f"{catalog}.cat"
    if not os.path.exists(cat_path):
        raise FileNotFoundError(f"FAST++ flux catalog missing: {cat_path}")

    # FAST++ reads the translate file as [CATALOG].translate.
    shutil.copyfile(cfg.translate_file, sedfit_dir / f"{catalog}.translate")
    overrides = _build_overrides(cfg, catalog, name_zphot)
    param_path = _write_param(cfg, tile, catalog, str(sedfit_dir), overrides)

    cmd = [str(cfg.fastpp_bin), os.path.basename(param_path)]
    log.info("Running FAST++: %s (cwd=%s, NAME_ZPHOT=%s)",
             " ".join(cmd), sedfit_dir, name_zphot)
    proc = subprocess.run(
        cmd, cwd=str(sedfit_dir), capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"FAST++ failed (exit {proc.returncode}).\nstdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )

    fout_path = sedfit_dir / f"{catalog}.fout"
    if not os.path.exists(fout_path):
        raise FileNotFoundError(f"FAST++ produced no .fout file: {fout_path}")
    fout = ascii_io.read(fout_path, header_start=-1)
    log.info("FAST++ produced %d SED fits.", len(fout))

    fastpp_info = {
        "name_zphot": name_zphot,
        "binary": str(cfg.fastpp_bin),
        "n_fits": len(fout),
        "params": overrides,
    }
    return Table(fout), fastpp_info


__all__ = ["run_fastpp"]
