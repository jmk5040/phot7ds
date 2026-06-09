"""
On-demand download of external reference catalogs via VizieR.

Thin wrapper around the project's ``Query/Vizier_Query.py`` helper. The
module is imported dynamically from :attr:`VACConfig.vizier_query_path` so
phot7ds stays importable without astroquery installed; the import only
happens when an auto-download is actually requested.

Only catalogs absent at their expected per-tile path are fetched.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path
from types import ModuleType

from .config import VACConfig

log = logging.getLogger(__name__)

# VAC external-catalog key -> Vizier_Query preset key.
_PRESET_KEYS = {
    "regalade": "regalade",
    "vhs": "vhs",
    "galex": "galex",
}

_vizier_module: ModuleType | None = None


def _load_vizier_query(cfg: VACConfig) -> ModuleType:
    """Import ``Vizier_Query.py`` from the configured path (cached)."""
    global _vizier_module
    if _vizier_module is not None:
        return _vizier_module
    path = Path(cfg.vizier_query_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Vizier_Query helper not found at {path}. Set "
            "VACConfig.vizier_query_path or disable auto_download."
        )
    spec = importlib.util.spec_from_file_location("phot7ds_vizier_query", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Vizier_Query from {path}.")
    module = importlib.util.module_from_spec(spec)
    # Register before exec so module-level @dataclass can resolve its module.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _vizier_module = module
    return module


def ensure_external_catalog(
    catalog_key: str,
    tile: str,
    tile_info,
    output_path: str | Path,
    cfg: VACConfig,
) -> bool:
    """Ensure the external catalog exists at ``output_path``.

    Returns ``True`` if the file is present (already there or freshly
    downloaded), ``False`` if it is still absent (download disabled or
    failed). Never raises for a failed download - the caller decides how to
    handle a missing optional catalog.
    """
    output_path = Path(output_path)
    if output_path.exists():
        return True
    if not cfg.auto_download:
        return False
    preset_key = _PRESET_KEYS.get(catalog_key)
    if preset_key is None:
        log.warning("No VizieR preset for %r; cannot auto-download.", catalog_key)
        return False

    try:
        vq = _load_vizier_query(cfg)
        preset = vq.CATALOG_PRESETS[preset_key]
        log.info("Auto-downloading %s for tile %s -> %s",
                 preset.label, tile, output_path.parent)
        os.makedirs(output_path.parent, exist_ok=True)
        outpath, n = vq.download_catalog_for_tile(
            tile_info, tile, preset, output_dir=output_path.parent, overwrite=False,
        )
        log.info("Downloaded %s (%d rows): %s", preset.label, n, outpath)
        return Path(outpath).exists()
    except Exception as exc:  # network / VizieR / parsing errors are non-fatal
        log.warning("Auto-download of %s for tile %s failed: %s",
                    catalog_key, tile, exc)
        return False


__all__ = ["ensure_external_catalog"]
