"""
On-demand download of external reference catalogs via VizieR.

Self-contained VizieR querying for the value-added catalog pipeline: a
catalog is queried inside the tile bounding box and trimmed to the tile
polygon, then written as ``{tile}_{suffix}.fits``. Only catalogs that are
absent at their expected per-tile path are fetched.

``astroquery`` is imported lazily (inside the query function) so that
``import phot7ds.vac`` keeps working without it; the import only happens
when an auto-download actually runs.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from astropy.io import fits
from astropy.table import Table

from ..tile_geometry import trim_to_tile_polygon
from .config import VACConfig

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CatalogPreset:
    """A VizieR catalog preset (id, output naming, optional column list)."""

    key: str
    vizier_id: str
    label: str
    columns: tuple[str, ...] | None = None


# VAC external-catalog key -> VizieR preset. These are the references the VAC
# cross-matcher knows how to consume; extend as needed.
CATALOG_PRESETS: dict[str, CatalogPreset] = {
    "regalade": CatalogPreset(
        key="regalade",
        vizier_id="J/A+A/706/A284/regalade",
        label="REGALADE",
    ),
    "vhs": CatalogPreset(
        key="vhs",
        vizier_id="II/367/vhs_dr5",
        label="VHS DR5",
        columns=("*", "e_Ypmag", "e_Jpmag", "e_Hpmag", "e_Kspmag"),
    ),
    "galex": CatalogPreset(
        key="galex",
        vizier_id="II/335/galex_ais",
        label="GALEX AIS",
    ),
}


def _detect_radec_columns(tab: Table) -> tuple[str, str]:
    """Identify RA/Dec column names in a VizieR result table."""
    candidates_ra = ["_RAJ2000", "RAJ2000", "RA_ICRS", "_RA.icrs", "RAdeg", "RA"]
    candidates_dec = ["_DEJ2000", "DEJ2000", "DE_ICRS", "_DE.icrs", "DEdeg", "DEC", "Dec"]
    ra_col = next((c for c in candidates_ra if c in tab.colnames), None)
    dec_col = next((c for c in candidates_dec if c in tab.colnames), None)
    if ra_col is None or dec_col is None:
        raise ValueError(
            f"Could not identify RA/Dec columns in table: {tab.colnames}"
        )
    return ra_col, dec_col


def _sanitize_table_for_fits(tab: Table) -> Table:
    """Drop verbose VizieR metadata that can break FITS header writing."""
    clean = tab.copy()
    clean.meta.clear()
    for col in clean.itercols():
        col.description = None
        if hasattr(col, "meta"):
            col.meta.clear()
    return clean


def _tile_corners(tile_info: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(ra_corners, dec_corners)`` from a tile row's ra1..ra4/dec1..dec4."""
    def _get(key: str) -> float:
        if isinstance(tile_info, Table):
            val = tile_info[key]
            return float(val[0] if hasattr(val, "__len__") and len(val) else val)
        return float(tile_info[key])

    ra = np.array([_get(f"ra{i}") for i in (1, 2, 3, 4)], dtype=float)
    dec = np.array([_get(f"dec{i}") for i in (1, 2, 3, 4)], dtype=float)
    return ra, dec


def query_vizier_catalog_for_tile(
    tile_info: Any,
    tile: str,
    catalog_id: str,
    *,
    columns: Sequence[str] | None = None,
) -> Table:
    """Query VizieR inside the tile bounding box, then trim to the polygon."""
    # Lazy import: keeps `import phot7ds.vac` working without astroquery.
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astroquery.vizier import Vizier

    ra_corners, dec_corners = _tile_corners(tile_info)
    ra_min, ra_max = float(np.min(ra_corners)), float(np.max(ra_corners))
    dec_min, dec_max = float(np.min(dec_corners)), float(np.max(dec_corners))
    center = SkyCoord(
        ra=(ra_min + ra_max) * 0.5 * u.deg,
        dec=(dec_min + dec_max) * 0.5 * u.deg,
    )
    width = (ra_max - ra_min) * u.deg
    height = (dec_max - dec_min) * u.deg

    if columns is None:
        vizier = Vizier(row_limit=-1)
    else:
        vizier = Vizier(columns=list(columns), row_limit=-1)

    result = vizier.query_region(center, width=width, height=height, catalog=catalog_id)
    if len(result) == 0:
        log.warning("%s: no entries in pre-query box (%s)", tile, catalog_id)
        return Table()

    tab = result[0]
    if len(tab) == 0:
        log.warning("%s: empty table (%s)", tile, catalog_id)
        return tab

    ra_col, dec_col = _detect_radec_columns(tab)
    tab = trim_to_tile_polygon(
        tile_info, tab, margin=0.0, rakey=ra_col, deckey=dec_col
    )
    log.info("%s: matched %d sources from %s", tile, len(tab), catalog_id)
    return tab


def download_catalog_for_tile(
    tile_info: Any,
    tile: str,
    preset: CatalogPreset,
    *,
    output_dir: str | Path,
    output_path: str | Path | None = None,
    columns: Sequence[str] | None = None,
    overwrite: bool = False,
) -> tuple[Path, int]:
    """Query VizieR for ``preset`` and write the per-tile FITS catalog.

    ``output_path`` (when given) sets the exact destination file; otherwise
    the file is ``{tile}_{preset.key}.fits`` under ``output_dir``. Returns
    ``(path, n_rows)``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outpath = (
        Path(output_path) if output_path is not None
        else output_dir / f"{tile}_{preset.key}.fits"
    )

    if outpath.exists() and not overwrite:
        with fits.open(outpath) as hdul:
            n = int(hdul[1].header.get("NAXIS2", 0)) if len(hdul) > 1 else 0
        log.info("%s: reusing existing catalog (%d rows): %s", tile, n, outpath)
        return outpath, n

    cols = list(columns) if columns is not None else preset.columns
    tab = query_vizier_catalog_for_tile(
        tile_info, tile, preset.vizier_id, columns=cols,
    )
    _sanitize_table_for_fits(tab).write(outpath, overwrite=True)
    log.info("Saved %s (%d rows)", outpath, len(tab))
    return outpath, len(tab)


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
    preset = CATALOG_PRESETS.get(catalog_key)
    if preset is None:
        log.warning("No VizieR preset for %r; cannot auto-download.", catalog_key)
        return False

    try:
        log.info("Auto-downloading %s for tile %s -> %s",
                 preset.label, tile, output_path)
        outpath, n = download_catalog_for_tile(
            tile_info, tile, preset,
            output_dir=output_path.parent,
            output_path=output_path,
            overwrite=False,
        )
        log.info("Downloaded %s (%d rows): %s", preset.label, n, outpath)
        return Path(outpath).exists()
    except Exception as exc:  # network / VizieR / parsing errors are non-fatal
        log.warning("Auto-download of %s for tile %s failed: %s",
                    catalog_key, tile, exc)
        return False


__all__ = [
    "CatalogPreset",
    "CATALOG_PRESETS",
    "query_vizier_catalog_for_tile",
    "download_catalog_for_tile",
    "ensure_external_catalog",
]
