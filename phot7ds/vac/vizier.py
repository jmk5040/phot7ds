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

import contextlib
import logging
import signal
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

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
    # VizieR's default VHS column set stops at the 2" aperture, so the 5.7"
    # (ap6) columns must be named explicitly. Of the two ap6 flavours
    # published, 'Japc6' (JAPERMAGNOAPERCORR6) is the extended-source
    # aperture magnitude with no aperture correction, which is what pairs
    # with a fixed 7DS aperture; 'Jap6' is corrected to total assuming a
    # *point* source and would put the total-light mismatch straight back in.
    # VizieR carries one error per aperture ('e_Jap6'), shared by both.
    # '*' keeps this a superset of the plain VHS staging, so one matched
    # table can serve both magnitude sets.
    "vhs_ap6": CatalogPreset(
        key="vhs_ap6",
        vizier_id="II/367/vhs_dr5",
        label="VHS DR5 (+ap6)",
        columns=("*", "e_Ypmag", "e_Jpmag", "e_Hpmag", "e_Kspmag",
                 "Yapc6", "Japc6", "Hapc6", "Ksapc6",
                 "e_Yap6", "e_Jap6", "e_Hap6", "e_Ksap6"),
    ),
    "galex": CatalogPreset(
        key="galex",
        vizier_id="II/335/galex_ais",
        label="GALEX AIS",
    ),
    # Not consumed by the flux assembly (WISE rides on the REGALADE columns),
    # but stageable for downstream use.
    "catwise": CatalogPreset(
        key="catwise",
        vizier_id="II/365/catwise",
        label="CatWISE2020",
        columns=("_RAJ2000", "_DEJ2000", "RAJ2000", "DEJ2000",
                 "W1mproPM", "e_W1mproPM", "W2mproPM", "e_W2mproPM"),
    ),
}


def vhs_preset_key(cfg: VACConfig) -> str:
    """VizieR preset key for the VHS staging that ``cfg`` reads."""
    return "vhs_ap6" if cfg.vhs_mag_set == "ap6" else "vhs"


@contextlib.contextmanager
def time_limit(seconds: int):
    """Abort the enclosed block after ``seconds`` via ``SIGALRM``.

    astroquery's VizieR client does not reliably honour its own ``timeout``
    and can stall indefinitely on a half-closed connection, which is enough
    to hang a whole batch. Only usable from the main thread; a no-op where
    ``SIGALRM`` is unavailable.
    """
    if not hasattr(signal, "SIGALRM") or threading.current_thread() is not threading.main_thread():
        yield
        return

    def _raise(signum, frame):
        raise TimeoutError(f"timed out after {seconds}s")

    previous = signal.signal(signal.SIGALRM, _raise)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


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


def tile_query_box(
    tile_info: Any, *, margin_deg: float = 0.0
) -> tuple[float, float, float, float]:
    """Return ``(ra_center, dec_center, width, height)`` in degrees.

    ``width`` is the tile's *angular* extent, i.e. the RA span deprojected by
    ``cos(dec)``. This matters near the poles: a 1.4°-wide tile at dec = -83°
    spans ~12° in raw RA, and querying VizieR with that raw span pulls in
    roughly an order of magnitude too much sky — enough to stall a dense
    catalog like VHS DR5. ``margin_deg`` pads the box; the polygon trim that
    follows removes the excess.
    """
    ra_corners, dec_corners = _tile_corners(tile_info)
    dec_c = float(np.mean(dec_corners))
    ra_c = float(np.mean(ra_corners))
    width = float(np.max(ra_corners) - np.min(ra_corners)) * np.cos(np.radians(dec_c))
    height = float(np.max(dec_corners) - np.min(dec_corners))
    return ra_c, dec_c, width + margin_deg, height + margin_deg


def query_vizier_catalog_for_tile(
    tile_info: Any,
    tile: str,
    catalog_id: str,
    *,
    columns: Sequence[str] | None = None,
    timeout: int | None = None,
    margin_deg: float = 0.0,
) -> Table:
    """Query VizieR inside the tile bounding box, then trim to the polygon."""
    # Lazy import: keeps `import phot7ds.vac` working without astroquery.
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astroquery.vizier import Vizier

    ra_c, dec_c, width_deg, height_deg = tile_query_box(
        tile_info, margin_deg=margin_deg
    )
    center = SkyCoord(ra=ra_c * u.deg, dec=dec_c * u.deg)
    width = width_deg * u.deg
    height = height_deg * u.deg

    kwargs: dict[str, Any] = {"row_limit": -1}
    if timeout is not None:
        kwargs["timeout"] = timeout
    if columns is not None:
        kwargs["columns"] = list(columns)
    vizier = Vizier(**kwargs)

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
    timeout: int | None = None,
    margin_deg: float = 0.0,
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
        tile_info, tile, preset.vizier_id, columns=cols, timeout=timeout,
        margin_deg=margin_deg,
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

    log.info("Auto-downloading %s for tile %s -> %s",
             preset.label, tile, output_path)
    return _fetch_with_retries(
        tile, tile_info, preset, output_path,
        timeout=cfg.vizier_timeout, attempts=cfg.vizier_attempts,
    )


def _fetch_with_retries(
    tile: str,
    tile_info: Any,
    preset: CatalogPreset,
    output_path: Path,
    *,
    timeout: int,
    attempts: int,
    margin_deg: float = 0.0,
) -> bool:
    """Download one reference under a hard time limit, retrying on failure.

    Never raises: the caller decides how to handle a missing optional
    catalog. Returns True when the file exists afterwards.
    """
    for attempt in range(1, max(1, attempts) + 1):
        try:
            with time_limit(timeout):
                outpath, n = download_catalog_for_tile(
                    tile_info, tile, preset,
                    output_dir=output_path.parent,
                    output_path=output_path,
                    overwrite=False,
                    timeout=timeout,
                    margin_deg=margin_deg,
                )
            log.info("Downloaded %s for %s (%d rows): %s",
                     preset.label, tile, n, outpath)
            return Path(outpath).exists()
        except Exception as exc:  # network / VizieR / parsing / timeout
            log.warning("Download of %s for tile %s failed (attempt %d/%d): "
                        "%s: %s", preset.label, tile, attempt, attempts,
                        type(exc).__name__, exc)
    return output_path.exists()


def prefetch_references(
    tiles: Iterable[str],
    tile_table: Table | str | Path,
    cfg: VACConfig,
    *,
    keys: Sequence[str] = ("vhs", "galex"),
    timeout: int | None = None,
    attempts: int | None = None,
    margin_deg: float = 0.05,
) -> dict[str, dict[str, bool]]:
    """Stage the per-tile external references before any fitting begins.

    Downloading up front, under a bounded time limit, keeps a flaky VizieR
    query from stalling a long batch part-way through: a reference that
    cannot be fetched is simply left out of the fit rather than blocking the
    tile. Set ``VACConfig.auto_download=False`` afterwards so the fitting
    stages never reach for the network themselves.

    ``keys`` are :data:`CATALOG_PRESETS` keys; ``"vhs"`` is redirected to the
    staging that ``cfg.vhs_mag_set`` actually reads. Returns
    ``{tile: {key: staged?}}``.
    """
    from .pipeline import select_tile_row

    if not isinstance(tile_table, Table):
        tile_table = Table.read(str(tile_table))
    timeout = cfg.vizier_timeout if timeout is None else timeout
    attempts = cfg.vizier_attempts if attempts is None else attempts

    results: dict[str, dict[str, bool]] = {}
    for tile in tiles:
        staged: dict[str, bool] = {}
        try:
            tile_info = select_tile_row(tile_table, tile)
        except KeyError:
            log.warning("prefetch %s: absent from the tile table, skipping", tile)
            results[tile] = staged
            continue
        for key in keys:
            preset_key = vhs_preset_key(cfg) if key == "vhs" else key
            preset = CATALOG_PRESETS.get(preset_key)
            if preset is None:
                log.warning("prefetch %s: no VizieR preset for %r", tile, key)
                staged[key] = False
                continue
            path = cfg.reference_path(key, tile)
            if path.exists():
                log.info("prefetch %s %s: already staged", tile, preset.label)
                staged[key] = True
                continue
            staged[key] = _fetch_with_retries(
                tile, tile_info, preset, path,
                timeout=timeout, attempts=attempts, margin_deg=margin_deg,
            )
        results[tile] = staged
    return results


__all__ = [
    "CatalogPreset",
    "CATALOG_PRESETS",
    "vhs_preset_key",
    "time_limit",
    "tile_query_box",
    "query_vizier_catalog_for_tile",
    "download_catalog_for_tile",
    "ensure_external_catalog",
    "prefetch_references",
]
