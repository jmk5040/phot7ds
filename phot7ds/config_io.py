"""
Bootstrapping helpers for external config and reference files.

These helpers solve two recurring chores in the example pipeline:

1. SE++ (``sourcextractor++``) and SWarp ship default configs that can be
   dumped on demand; :func:`ensure_sepp_config` / :func:`ensure_swarp_config`
   create them when absent. Existing files are never overwritten.
2. The tile table (``7DT_tiles.ascii``) and the Gaia XP reference catalog
   are **survey artefacts** the user must provide. :func:`require_tile_table`
   and :func:`require_gaiaxp_reference` raise :class:`FileNotFoundError`
   with a helpful message when missing.

All paths are :class:`pathlib.Path` objects.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


# --- Auto-generated configs -------------------------------------------------


def _resolve_tool(name: str, override: str | None = None) -> str:
    """Return the binary path for ``name`` or raise ``RuntimeError``."""
    if override:
        return override
    found = shutil.which(name)
    if found is None:
        raise RuntimeError(
            f"{name!r} not found on PATH. Install it or pass an explicit "
            f"binary path."
        )
    return found


def ensure_sepp_config(
    path: str | Path,
    *,
    overwrite: bool = False,
    sepp_binary: str | None = None,
) -> Path:
    """Make sure a SourceExtractor++ ``--config-file`` exists.

    If ``path`` does not exist (or ``overwrite=True``), dumps the default
    via ``sourcextractor++ --dump-default-config > path``. Otherwise the
    existing file is left untouched and only its path is returned.
    """
    out = Path(path)
    if out.exists() and not overwrite:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    binary = _resolve_tool("sourcextractor++", sepp_binary)
    log.info("Generating SE++ default config -> %s", out)
    with open(out, "w") as fh:
        subprocess.run([binary, "--dump-default-config"], check=True, stdout=fh)
    return out


def ensure_swarp_config(
    path: str | Path,
    *,
    overwrite: bool = False,
    swarp_binary: str | None = None,
) -> Path:
    """Make sure a SWarp config file exists.

    If ``path`` does not exist (or ``overwrite=True``), dumps the default
    via ``SWarp -dd > path``. Otherwise the existing file is returned as-is.
    """
    out = Path(path)
    if out.exists() and not overwrite:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    binary = _resolve_tool("SWarp", swarp_binary)
    log.info("Generating SWarp default config -> %s", out)
    with open(out, "w") as fh:
        subprocess.run([binary, "-dd"], check=True, stdout=fh)
    return out


# --- Required survey artefacts ---------------------------------------------


def require_tile_table(path: str | Path) -> Path:
    """Return ``path`` if it exists; otherwise raise with a helpful hint.

    ``7DT_tiles.ascii`` (or the equivalent FITS table) is essential because
    it carries the tile corner / centre coordinates used everywhere in the
    pipeline. It cannot be auto-generated.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            "Tile table not found at: {p}\n\n"
            "phot7ds expects a CSV-like table with columns "
            "'tile', 'ra', 'dec', 'ra1'..'ra4', 'dec1'..'dec4'. The 7DS "
            "team distributes 7DT_tiles.ascii in the survey config bundle; "
            "copy it into the path above before running.".format(p=p)
        )
    return p


def require_gaiaxp_reference(
    reference_dir: str | Path,
    *,
    tile: str,
    filename_template: str = "gaiaxp_dr3_synphot_{tile}.csv",
) -> Path:
    """Locate the Gaia XP reference catalog for ``tile`` or raise.

    Parameters
    ----------
    reference_dir
        Directory holding per-tile Gaia XP CSVs.
    tile
        Tile identifier substituted into ``filename_template``.
    filename_template
        Format string with one ``{tile}`` placeholder.
    """
    out = Path(reference_dir) / filename_template.format(tile=tile)
    if not out.exists():
        raise FileNotFoundError(
            "Gaia XP reference catalog not found at: {p}\n\n"
            "phot7ds requires a per-tile synphot CSV with at least "
            "'ra', 'dec' and 'mag_<band>' columns. Drop the file in the "
            "above location before running, or pass an explicit "
            "``reference_catalog=`` argument to run_photometry().".format(
                p=out
            )
        )
    return out


__all__ = [
    "ensure_sepp_config",
    "ensure_swarp_config",
    "require_tile_table",
    "require_gaiaxp_reference",
]
