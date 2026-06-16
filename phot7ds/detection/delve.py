"""
Build a DELVE-DR3 detection (or mask) image for a 7DT tile.

The tile field of view is partitioned into a grid of patches; each patch is
queried via the NOIRLab SIA service for the requested band/product type,
downloaded, and finally co-added with SWarp into a single mosaic image at the
target tile center, pixel scale and image size.
"""
from __future__ import annotations

import logging
import math
import os
import random
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Iterable, Literal

import astropy.units as u
import numpy as np
import requests
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from pyvo.dal import SIAService

log = logging.getLogger(__name__)

DELVE_SIA_URL = "https://datalab.noirlab.edu/sia/delve_dr3"

# HTTP statuses we consider worth a retry (transient on the server side).
_RETRYABLE_HTTP_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})


def _retryable_exception(exc: BaseException) -> bool:
    """Return True if ``exc`` looks like a transient network/HTTP failure."""
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return True
    if isinstance(exc, requests.HTTPError):
        resp = getattr(exc, "response", None)
        if resp is not None and resp.status_code in _RETRYABLE_HTTP_STATUSES:
            return True
    # Astropy / pyvo wrap server errors as plain exceptions; match by text.
    text = str(exc).lower()
    return any(token in text for token in (
        "502", "503", "504", "bad gateway", "gateway timeout",
        "service unavailable", "connection reset", "remote disconnect",
        "temporary failure", "timed out",
    ))


def _backoff_seconds(attempt: int, base: float, cap: float = 60.0) -> float:
    """Exponential backoff with jitter."""
    delay = min(base * (2 ** max(0, attempt - 1)), cap)
    return delay * (0.5 + random.random())


def _get_tile_value(tile_info: Any, key: str):
    if isinstance(tile_info, Table):
        if len(tile_info) == 0:
            raise ValueError("tile_info is empty")
        return tile_info[key][0]
    return tile_info[key]


def _sexagesimal(value: float, *, hours: bool) -> tuple[int, int, float]:
    """Convert a non-negative decimal value to (h|d, m, s) with carry.

    The seconds component is rounded to 2 decimals; if the rounding pushes
    it to 60.00 the carry propagates upwards. This avoids invalid strings
    like ``14:39:60.00`` that earlier produced ``14:39:60.00`` instead of
    ``14:40:00.00`` in coordinate writers.
    """
    assert value >= 0
    primary = int(value)
    minutes = int((value - primary) * 60)
    seconds = ((value - primary) * 60 - minutes) * 60
    seconds = round(seconds, 2)
    if seconds >= 60.0:
        seconds -= 60.0
        minutes += 1
    if minutes >= 60:
        minutes -= 60
        primary += 1
    if hours and primary >= 24:
        primary -= 24
    return primary, minutes, seconds


def deg_to_hms_dms(ra_deg: float, dec_deg: float) -> tuple[str, str]:
    """Convert decimal degrees to SWarp / FITS sexagesimal strings.

    RA is returned as ``HH:MM:SS.ss`` (hours); Dec as ``[+|-]DD:MM:SS.ss``.
    Matches the convention used in ``Utils_7DT.deg_to_hms_dms`` and
    propagates seconds/minutes carry-over (so a rounded ``60.00`` second
    component never appears in the output).
    """
    ra_deg = float(ra_deg)
    dec_deg = float(dec_deg)

    rh, rm, rs = _sexagesimal(ra_deg / 15.0, hours=True)
    ra_str = f"{rh:02d}:{rm:02d}:{rs:05.2f}"

    sign = "+" if dec_deg >= 0 else "-"
    dd, dm, ds = _sexagesimal(abs(dec_deg), hours=False)
    dec_str = f"{sign}{dd:02d}:{dm:02d}:{ds:05.2f}"

    return ra_str, dec_str


def _resolve_swarp_center(
    tile_info: Any,
    ra_center: str | float | None,
    dec_center: str | float | None,
) -> tuple[str, str]:
    """Return SWarp ``-CENTER`` sexagesimal strings."""
    if ra_center is not None and dec_center is not None:
        return str(ra_center), str(dec_center)
    if ra_center is not None or dec_center is not None:
        raise ValueError("pass both ra_center and dec_center, or neither")
    ra_deg = round(float(_get_tile_value(tile_info, "ra")), 4)
    dec_deg = round(float(_get_tile_value(tile_info, "dec")), 4)
    ra_str, dec_str = deg_to_hms_dms(ra_deg, dec_deg)
    log.info(
        "SWarp center from tile table: RA=%s Dec=%s (%.4f deg, %.4f deg)",
        ra_str,
        dec_str,
        ra_deg,
        dec_deg,
    )
    return ra_str, dec_str


def _auto_grid_counts(
    ra_span: float,
    dec_span: float,
    patch_size_deg: float,
    overlap: float,
) -> tuple[int, int]:
    """Number of columns/rows so patches overlap in *coordinate* degrees.

    The patch footprint is taken conservatively as ``patch_size_deg``
    coordinate degrees on each axis (no ``cos(dec)`` credit in RA), so the
    grid is guaranteed to overlap even at high declination where the RA
    coordinate span is stretched by ``1/cos(dec)``. ``overlap`` is the
    fractional overlap between adjacent patches (0 = touching, 0.3 = 30%).
    """
    step = max(1e-6, patch_size_deg * (1.0 - overlap))
    n_cols = max(1, math.ceil(ra_span / step))
    n_rows = max(1, math.ceil(dec_span / step))
    return n_cols, n_rows


def build_patch_centers(
    tile_info: Any,
    *,
    n_cols: int | None = 9,
    n_rows: int | None = 6,
    patch_size_deg: float | None = None,
    overlap: float = 0.0,
) -> list[tuple[float, float]]:
    """Generate ``(ra, dec)`` patch centers tiling the field of view.

    The four corner positions ``(ra1..ra4, dec1..dec4)`` of ``tile_info``
    define the bounding rectangle, which is divided into a ``n_cols x n_rows``
    grid of equal-area patches.

    Pass ``n_cols=None`` and/or ``n_rows=None`` to size the grid
    automatically from ``patch_size_deg`` and ``overlap`` so the patches are
    guaranteed to overlap in coordinate degrees. This matters at high
    declination: the RA coordinate span is stretched by ``1/cos(dec)``, so a
    fixed column count that works near the equator leaves thin RA gaps in
    the mosaic.
    """
    ra = [float(_get_tile_value(tile_info, f"ra{i}")) for i in (1, 2, 3, 4)]
    dec = [float(_get_tile_value(tile_info, f"dec{i}")) for i in (1, 2, 3, 4)]
    ra_min, ra_max = min(ra), max(ra)
    dec_min, dec_max = min(dec), max(dec)
    ra_span = ra_max - ra_min
    dec_span = dec_max - dec_min

    if n_cols is None or n_rows is None:
        if patch_size_deg is None:
            raise ValueError("patch_size_deg is required when n_cols/n_rows is None")
        auto_cols, auto_rows = _auto_grid_counts(ra_span, dec_span, patch_size_deg, overlap)
        if n_cols is None:
            n_cols = auto_cols
        if n_rows is None:
            n_rows = auto_rows

    ra_centers = [ra_min + (2 * i + 1) * ra_span / (2 * n_cols) for i in range(n_cols)]
    dec_centers = [dec_max - (2 * i + 1) * dec_span / (2 * n_rows) for i in range(n_rows)]

    centers: list[tuple[float, float]] = []
    for r in range(n_rows):
        for c in range(n_cols):
            centers.append((ra_centers[c], dec_centers[r]))
    return centers


def download_delve_patches(
    centers: Iterable[tuple[float, float]],
    *,
    output_path: str,
    tile: str,
    imgtype: str,
    detection_band: str = "det",
    patch_size_deg: float = 0.25,
    sia_url: str = DELVE_SIA_URL,
    ncores: int = 12,
    request_timeout: float = 120,
    max_retries: int = 5,
    retry_backoff_sec: float = 2.0,
    filename_prefix: str = "patch",
    all_matches: bool = False,
) -> list[str]:
    """Query the DELVE SIA service for each ``(ra, dec)`` center and download.

    Returns the sorted list of successfully downloaded patch FITS paths.
    Transient SIA/HTTP failures are retried with exponential backoff; only
    patches that match the requested ``imgtype`` / ``detection_band`` (and
    are not ``_nobkg``) are kept. Used by both the full mosaic builder and
    the gap-filler.

    With ``all_matches=False`` (default) the first matching brick per center
    is downloaded. With ``all_matches=True`` *every* overlapping matching
    brick is downloaded (deduplicated by access URL across centers); this is
    needed near brick boundaries / tile edges where the first-returned brick
    does not cover the queried position but a neighbouring one does. The
    extra bricks are MAX-combined downstream by SWarp.
    """
    os.makedirs(output_path, exist_ok=True)
    centers = list(centers)
    thread_local = threading.local()
    seen_urls: set[str] = set()
    seen_lock = threading.Lock()

    def _clients() -> tuple[SIAService, requests.Session]:
        if not hasattr(thread_local, "sia_service"):
            thread_local.sia_service = SIAService(sia_url)
        if not hasattr(thread_local, "http_session"):
            thread_local.http_session = requests.Session()
        return thread_local.sia_service, thread_local.http_session

    def _download(url: str, dest: str) -> None:
        _, http_session = _clients()
        response = http_session.get(url, stream=True, timeout=(15, request_timeout))
        response.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)

    def _worker(args: tuple[int, float, float]) -> tuple[int, list[str], str]:
        patch_idx, ra, dec = args
        last_error = "Patch download failed after retries"
        for attempt in range(1, max_retries + 1):
            try:
                sia_service, _ = _clients()
                position = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
                results = sia_service.search(pos=position, size=patch_size_deg * u.deg)
                table = results.to_table()
                if len(table) == 0:
                    return patch_idx, [], "No SIA rows returned."
                prodtype = np.asarray(table["prodtype"]).astype(str)
                bandpass = np.asarray(table["obs_bandpass"]).astype(str)
                publisher_did = np.asarray(table["obs_publisher_did"]).astype(str)
                row_mask = (
                    (prodtype == imgtype)
                    & (bandpass == detection_band)
                    & (np.char.find(publisher_did, "_nobkg") < 0)
                )
                if not np.any(row_mask):
                    return patch_idx, [], "No matching DELVE patch found."
                rows = table[row_mask]
                if not all_matches:
                    rows = rows[:1]
                out: list[str] = []
                for sub, match in enumerate(rows):
                    url = str(match["access_url"])
                    with seen_lock:
                        if url in seen_urls:
                            continue
                        seen_urls.add(url)
                    exptime = int(float(match["exptime"]))
                    patch_img = (
                        f"{output_path}/DELVE{imgtype.upper()}_{tile}_{detection_band}_"
                        f"{filename_prefix}{patch_idx:03d}_{sub:02d}_{ra:.4f}{dec:.4f}_"
                        f"{patch_size_deg:.2f}x{patch_size_deg:.2f}_{exptime}sec.fits"
                    )
                    _download(url, patch_img)
                    out.append(patch_img)
                return patch_idx, out, "ok"
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt == max_retries or not _retryable_exception(exc):
                    return patch_idx, [], f"Patch download failed: {last_error}"
                delay = _backoff_seconds(attempt, retry_backoff_sec)
                log.warning(
                    "[%s] patch %03d attempt %d/%d failed (%s); retry in %.1fs",
                    tile, patch_idx, attempt, max_retries, last_error, delay,
                )
                time.sleep(delay)
        return patch_idx, [], last_error

    tasks = [(idx, ra, dec) for idx, (ra, dec) in enumerate(centers, 1)]
    n_patches = len(tasks)
    n_workers = max(1, min(ncores, n_patches))
    log.info("[%s] launching %d workers for %d patches", tile, n_workers, n_patches)

    downloaded: list[str] = []
    n_ok = 0
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_worker, t): t[0] for t in tasks}
        for future in as_completed(futures):
            idx = futures[future]
            _, paths, message = future.result()
            if paths:
                n_ok += 1
                log.info("[%s] patch %d/%d OK: %d brick(s)", tile, idx, n_patches, len(paths))
                downloaded.extend(paths)
            elif message == "ok":
                log.info("[%s] patch %d/%d OK: (all bricks already fetched)",
                         tile, idx, n_patches)
            else:
                log.warning("[%s] patch %d/%d FAIL: %s", tile, idx, n_patches, message)

    log.info("[%s] patch download summary: %d/%d centers, %d brick files",
             tile, n_ok, n_patches, len(downloaded))
    return sorted(downloaded)


def build_delve_detection_image(
    *,
    tile_info: Any,
    ra_center: str | float | None = None,
    dec_center: str | float | None = None,
    imgtype: Literal["image", "mask"],
    output_path: str,
    swarp_cfg_path: str,
    detection_band: str = "det",
    n_cols: int | None = 9,
    n_rows: int | None = 6,
    patch_size_deg: float = 0.25,
    overlap: float = 0.0,
    ncores: int = 12,
    sia_url: str = DELVE_SIA_URL,
    combine_type: str = "MAX",
    pixscale_arcsec: float = 0.505,
    image_size_x: int = 10200,
    image_size_y: int = 6800,
    cleanup_patches: bool = True,
    request_timeout: float = 120,
    max_retries: int = 5,
    retry_backoff_sec: float = 2.0,
    min_patch_fraction: float = 0.6,
) -> tuple[str, str]:
    """Build a DELVE detection (or mask) mosaic for one tile.

    Parameters
    ----------
    tile_info
        Single-row :class:`~astropy.table.Table` (or dict-like) with corners
        ``ra1/dec1 .. ra4/dec4``, the ``tile`` identifier, and (when
        ``ra_center`` / ``dec_center`` are omitted) ``ra`` / ``dec`` in
        decimal degrees.
    ra_center, dec_center
        Center passed to SWarp as sexagesimal strings (e.g. FITS ``OBJCTRA`` /
        ``OBJCTDEC``). If either is ``None``, both are taken from
        ``round(tile_info['ra'], 4)`` and ``round(tile_info['dec'], 4)`` and
        converted to ``HH:MM:SS`` / ``±DD:MM:SS``.
    imgtype
        ``'image'`` (science) or ``'mask'`` (bad-pixel mask).
    output_path
        Output directory for this tile (will be created).
    swarp_cfg_path
        Path to the SWarp config (``default.swarp``).
    detection_band
        DELVE bandpass identifier (default ``'det'``).
    n_cols, n_rows, patch_size_deg, overlap
        Patch grid geometry. Pass ``n_cols=None`` / ``n_rows=None`` to size
        the grid automatically from ``patch_size_deg`` and ``overlap`` so the
        patches overlap in coordinate degrees (recommended for high-|dec|
        tiles, where a fixed column count leaves thin RA gaps). ``overlap``
        is the fractional overlap between neighbouring patches.
    ncores
        Concurrency: number of worker threads for SIA downloads (also passed
        to SWarp ``-NTHREADS``).
    combine_type
        SWarp combine type for ``imgtype='image'``. Masks always use ``MAX``.
    pixscale_arcsec
        Output pixel scale.
    image_size_x, image_size_y
        Output mosaic dimensions in pixels.
    cleanup_patches
        Remove downloaded patch images after the mosaic is built.
    request_timeout
        Read timeout (seconds) for the per-patch HTTP download.
    max_retries
        Retry attempts per patch on transient SIA / HTTP failures.
        Retries use exponential backoff with jitter (see
        ``retry_backoff_sec``); only transient errors are retried (502,
        503, 504, 429, connection / timeout). Non-retryable errors fail
        the patch immediately.
    retry_backoff_sec
        Base of the exponential backoff between retries (seconds).
    min_patch_fraction
        Minimum fraction of patches that must download successfully
        before SWarp is run. If fewer succeed, :class:`RuntimeError` is
        raised so partial mosaics do not silently fall through.

    Returns
    -------
    detection_image, detection_weight
        Paths to the output FITS image and weight file.
    """
    if imgtype not in ("image", "mask"):
        raise ValueError("imgtype must be 'image' or 'mask'")

    ra_center, dec_center = _resolve_swarp_center(tile_info, ra_center, dec_center)

    tile = str(_get_tile_value(tile_info, "tile"))
    os.makedirs(output_path, exist_ok=True)

    centers = build_patch_centers(
        tile_info, n_cols=n_cols, n_rows=n_rows,
        patch_size_deg=patch_size_deg, overlap=overlap,
    )
    n_patches = len(centers)

    patch_imgs = download_delve_patches(
        centers,
        output_path=output_path,
        tile=tile,
        imgtype=imgtype,
        detection_band=detection_band,
        patch_size_deg=patch_size_deg,
        sia_url=sia_url,
        ncores=ncores,
        request_timeout=request_timeout,
        max_retries=max_retries,
        retry_backoff_sec=retry_backoff_sec,
    )
    if not patch_imgs:
        raise RuntimeError(f"[{tile}] no patch images downloaded")

    min_required = max(1, int(min_patch_fraction * n_patches))
    if len(patch_imgs) < min_required:
        raise RuntimeError(
            f"[{tile}] only {len(patch_imgs)}/{n_patches} patches downloaded "
            f"(< {min_patch_fraction*100:.0f}%); aborting before SWarp to avoid "
            "a partial mosaic. Re-run when the SIA service is more stable, "
            "lower ncores, or raise max_retries / lower min_patch_fraction."
        )

    exptime = 0.0
    gain = 0.0
    saturate = 0.0
    for img in patch_imgs:
        hdr = fits.getheader(img)
        exptime += float(hdr.get("EXPTIME", 1.0))
        gain += float(hdr.get("GAIN", 1.0))
        saturate += float(hdr.get("SATURATE", 1.0))
    exptime = int(exptime / len(patch_imgs))
    gain = float(gain / len(patch_imgs))
    saturate = float(saturate / len(patch_imgs))

    list_file = (
        f"{output_path}/DELVE{imgtype.upper()}_{tile}_{detection_band}_"
        f"{combine_type}_{exptime}sec_coadd.list"
    )
    with open(list_file, "w") as f:
        for img in patch_imgs:
            f.write(img + "\n")

    detection_img = f"{output_path}/{tile}_DELVE_DR3_{imgtype.upper()}_det.fits"
    detection_wgt = f"{output_path}/{tile}_DELVE_DR3_{imgtype.upper()}_det_weight.fits"

    swarp_args = _build_swarp_args(
        list_file=list_file,
        swarp_cfg_path=swarp_cfg_path,
        imgtype=imgtype,
        combine_type=combine_type,
        ra_center=ra_center,
        dec_center=dec_center,
        image_size_x=image_size_x,
        image_size_y=image_size_y,
        pixscale_arcsec=pixscale_arcsec,
        output_path=output_path,
        ncores=ncores,
        gain=gain,
        saturate=saturate,
        detection_img=detection_img,
        detection_wgt=detection_wgt,
    )

    proc = subprocess.run(swarp_args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"swarp failed for {tile}: {proc.stderr}")
    if not os.path.exists(detection_img):
        raise RuntimeError(f"swarp finished but output missing: {detection_img}")

    with fits.open(detection_img, memmap=True) as hdul:
        hdul[0].data = hdul[0].data.astype(np.float32)
        hdul[0].header["GAIN"] = gain
        hdul[0].header["EXPTIME"] = exptime
        hdul[0].header["SATURATE"] = saturate
        hdul.writeto(detection_img, overwrite=True)

    if cleanup_patches:
        for img in patch_imgs:
            if os.path.exists(img):
                os.remove(img)
    # Clean the list file and (unused) weight file. Both may be absent on
    # an unusual SWarp run; treat the missing case as benign.
    for stale in (
        list_file,
        f"{output_path}/{tile}_DELVE_DR3_{imgtype.upper()}_det_weight.fits",
    ):
        try:
            os.remove(stale)
        except FileNotFoundError:
            pass

    return detection_img, detection_wgt


def _build_swarp_args(
    *,
    list_file: str,
    swarp_cfg_path: str,
    imgtype: str,
    combine_type: str,
    ra_center: str | float,
    dec_center: str | float,
    image_size_x: int,
    image_size_y: int,
    pixscale_arcsec: float,
    output_path: str,
    ncores: int,
    gain: float,
    saturate: float,
    detection_img: str,
    detection_wgt: str,
) -> list[str]:
    common = [
        "SWarp", f"@{list_file}",
        "-c", swarp_cfg_path,
        "-FSCALASTRO_TYPE", "FIXED",
        "-FSCALE_KEYWORD", "NONE",
        "-PIXELSCALE_TYPE", "MANUAL",
        "-PIXEL_SCALE", f"{pixscale_arcsec:.4f}",
        "-CENTER_TYPE", "MANUAL",
        "-CENTER", f"{ra_center},{dec_center}",
        "-IMAGE_SIZE", f"{image_size_x},{image_size_y}",
        "-FSCALE_DEFAULT", "1.0",
        "-GAIN_DEFAULT", f"{gain}",
        "-SATLEV_DEFAULT", f"{saturate}",
        "-RESAMPLE", "Y",
        "-RESAMPLE_DIR", output_path,
        "-DELETE_TMPFILES", "Y",
        "-WRITE_XML", "N",
        "-WRITE_FILEINFO", "Y",
        "-VERBOSE_TYPE", "NORMAL",
        "-WEIGHTOUT_NAME", detection_wgt,
        "-NTHREADS", f"{ncores}",
        "-COPY_KEYWORDS", "MJD-OBS,EXPTIME,GAIN,SATURATE,BAND",
        "-IMAGEOUT_NAME", detection_img,
    ]
    specific = [
        "-COMBINE_TYPE", "MAX",
        "-RESAMPLING_TYPE", "NEAREST",
        "-SUBTRACT_BACK", "N",
        "-INTERPOLATE", "N",
    ]
    # if imgtype == "mask":
    #     specific = [
    #         "-COMBINE_TYPE", "MAX",
    #         "-RESAMPLING_TYPE", "NEAREST",
    #         "-SUBTRACT_BACK", "N",
    #         "-INTERPOLATE", "N",
    #     ]
    #     # Masks: union (MAX), nearest-neighbour, no background subtraction or sigma clipping.
    # else:
    #     specific = [
    #         "-COMBINE_TYPE", combine_type,
    #         "-RESAMPLING_TYPE", "LANCZOS3",
    #         "-SUBTRACT_BACK", "N",
    #         "-INTERPOLATE", "Y",
    #         "-CLIP_SIGMA", "4.0",
    #     ]
    return common + specific


def _gap_query_centers(
    gap: np.ndarray,
    wcs: WCS,
    patch_size_deg: float,
    overlap: float,
    *,
    stride: int = 4,
) -> list[tuple[float, float]]:
    """Minimal set of ``(ra, dec)`` patch centers covering the gap pixels.

    Empty (``gap``) pixels are projected to the sky and binned onto a grid
    of step ``patch_size_deg * (1 - overlap)`` coordinate degrees; one query
    center is emitted per occupied bin, so a thin gap stripe maps to a small
    number of overlapping patch queries.
    """
    ys, xs = np.where(gap)
    if stride > 1:
        ys, xs = ys[::stride], xs[::stride]
    if xs.size == 0:
        return []
    ra, dec = wcs.all_pix2world(xs.astype(float), ys.astype(float), 0)
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)
    good = np.isfinite(ra) & np.isfinite(dec)
    ra, dec = ra[good], dec[good]
    if ra.size == 0:
        return []
    step = max(1e-6, patch_size_deg * (1.0 - overlap))
    ra0, dec0 = float(ra.min()), float(dec.min())
    bi = np.floor((ra - ra0) / step).astype(np.int64)
    bj = np.floor((dec - dec0) / step).astype(np.int64)
    # One query per occupied bin, placed at the *centroid of the gap pixels*
    # in that bin (not the geometric bin center). DELVE gaps sit on brick
    # boundaries, so a query offset from the seam returns the neighbouring
    # brick whose edge coincides with the gap; centering on the gap pixels
    # makes the SIA return a brick that actually covers them.
    key = bi * 100000 + bj
    order = np.argsort(key, kind="stable")
    key_s, ra_s, dec_s = key[order], ra[order], dec[order]
    uk, start = np.unique(key_s, return_index=True)
    cnt = np.diff(np.append(start, key_s.size))
    ra_c = np.add.reduceat(ra_s, start) / cnt
    dec_c = np.add.reduceat(dec_s, start) / cnt
    return list(zip(ra_c.tolist(), dec_c.tolist()))


def _swarp_gap_mosaic(
    patch_imgs: list[str],
    *,
    hdr: fits.Header,
    swarp_cfg_path: str,
    patch_dir: str,
    tag: str,
    ncores: int,
) -> tuple[np.ndarray, np.ndarray]:
    """SWarp ``patch_imgs`` onto the frame described by ``hdr``.

    Returns ``(data, covered)`` where ``covered`` is the boolean coverage
    map derived from the SWarp weight image (so a valid value of 0 in a mask
    mosaic is not mistaken for "no coverage").
    """
    nx = int(hdr["NAXIS1"])
    ny = int(hdr["NAXIS2"])
    if "CD1_1" in hdr:
        pixscale = abs(float(hdr["CD1_1"])) * 3600.0
    else:
        pixscale = abs(float(hdr["CDELT1"])) * 3600.0
    ra_c, dec_c = deg_to_hms_dms(float(hdr["CRVAL1"]), float(hdr["CRVAL2"]))

    list_file = os.path.join(patch_dir, f"{tag}.list")
    with open(list_file, "w") as fh:
        fh.write("\n".join(patch_imgs) + "\n")
    gap_img = os.path.join(patch_dir, f"{tag}.fits")
    gap_wgt = os.path.join(patch_dir, f"{tag}_weight.fits")

    swarp_args = [
        "SWarp", f"@{list_file}",
        "-c", swarp_cfg_path,
        "-COMBINE_TYPE", "MAX",
        "-RESAMPLING_TYPE", "NEAREST",
        "-SUBTRACT_BACK", "N",
        "-FSCALASTRO_TYPE", "FIXED",
        "-FSCALE_KEYWORD", "NONE",
        "-FSCALE_DEFAULT", "1.0",
        "-PIXELSCALE_TYPE", "MANUAL",
        "-PIXEL_SCALE", f"{pixscale:.4f}",
        "-CENTER_TYPE", "MANUAL",
        "-CENTER", f"{ra_c},{dec_c}",
        "-IMAGE_SIZE", f"{nx},{ny}",
        "-INTERPOLATE", "N",
        "-RESAMPLE", "Y",
        "-RESAMPLE_DIR", patch_dir,
        "-DELETE_TMPFILES", "Y",
        "-WRITE_XML", "N",
        "-VERBOSE_TYPE", "QUIET",
        "-WEIGHTOUT_NAME", gap_wgt,
        "-NTHREADS", str(ncores),
        "-IMAGEOUT_NAME", gap_img,
    ]
    proc = subprocess.run(swarp_args, capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(gap_img):
        raise RuntimeError(f"gap-fill SWarp failed: {proc.stderr}")

    gap_data = fits.getdata(gap_img)
    if os.path.exists(gap_wgt):
        covered = fits.getdata(gap_wgt) > 0
    else:
        covered = np.isfinite(gap_data) & (gap_data != 0)
    for p in (list_file, gap_img, gap_wgt):
        try:
            os.remove(p)
        except FileNotFoundError:
            pass
    return gap_data, covered


def fill_delve_detection_gaps(
    *,
    image_path: str,
    swarp_cfg_path: str,
    imgtype: Literal["image", "mask"],
    coverage_reference: str | None = None,
    detection_band: str = "det",
    patch_size_deg: float = 0.25,
    overlap: float = 0.5,
    max_passes: int = 5,
    sia_url: str = DELVE_SIA_URL,
    ncores: int = 12,
    request_timeout: float = 120,
    max_retries: int = 5,
    retry_backoff_sec: float = 2.0,
    coverage_tol: float = 1e-4,
    cleanup: bool = True,
) -> dict:
    """Fill coverage gaps in an existing DELVE mosaic, in place.

    Gaps (empty pixels) are located from ``coverage_reference`` (defaults to
    ``image_path``; for a *mask* pass the sibling science-image path so the
    well-defined image coverage drives the gap definition). DELVE patches are
    queried only for the gap regions, SWarped onto the *same* frame as
    ``image_path``, and merged into the empty pixels.

    The fill iterates up to ``max_passes`` times: each pass re-derives query
    centers from the *remaining* gap, so patch-edge slivers left by one pass
    are recentered and covered by the next. The file is rewritten in place
    (original header preserved).

    Returns a summary dict (gap fraction before/after, query centers,
    downloaded patches, passes run).
    """
    if imgtype not in ("image", "mask"):
        raise ValueError("imgtype must be 'image' or 'mask'")

    image_path = str(image_path)
    ref_path = str(coverage_reference) if coverage_reference else image_path
    tile = os.path.basename(image_path).split("_")[0]
    out_dir = os.path.dirname(image_path) or "."
    patch_dir = os.path.join(out_dir, f"{tile}_gapfill_{imgtype}")

    # Frame + data of the file we will rewrite.
    with fits.open(image_path) as hdul:
        data = hdul[0].data
        hdr = hdul[0].header.copy()
        wcs = WCS(hdr)

    # "want" = the region we ultimately want covered, from the reference.
    with fits.open(ref_path) as hdul:
        ref_data = hdul[0].data
    want = ~(np.isfinite(ref_data) & (ref_data != 0))
    gap_frac = float(want.mean())

    summary = {
        "image": image_path,
        "tile": tile,
        "gap_fraction_before": gap_frac,
        "n_centers": 0,
        "n_patches": 0,
        "passes": 0,
        "gap_fraction_after": gap_frac,
        "filled": False,
    }
    if gap_frac <= coverage_tol or not want.any():
        log.info("[%s] no gaps to fill in %s (gap fraction %.4f)",
                 tile, os.path.basename(image_path), gap_frac)
        return summary

    filled_region = np.zeros_like(want)
    total_centers = 0
    total_patches = 0

    for pass_idx in range(1, max_passes + 1):
        remaining = want & ~filled_region
        rem_frac = float(remaining.mean())
        if rem_frac <= coverage_tol or not remaining.any():
            break
        centers = _gap_query_centers(remaining, wcs, patch_size_deg, overlap)
        if not centers:
            break
        log.info("[%s] %s pass %d: gap %.4f -> %d gap-patch queries",
                 tile, os.path.basename(image_path), pass_idx, rem_frac, len(centers))
        patch_imgs = download_delve_patches(
            centers,
            output_path=patch_dir,
            tile=tile,
            imgtype=imgtype,
            detection_band=detection_band,
            patch_size_deg=patch_size_deg,
            sia_url=sia_url,
            ncores=ncores,
            request_timeout=request_timeout,
            max_retries=max_retries,
            retry_backoff_sec=retry_backoff_sec,
            filename_prefix=f"gap{pass_idx}_",
            all_matches=True,
        )
        total_centers += len(centers)
        total_patches += len(patch_imgs)
        if not patch_imgs:
            log.warning("[%s] pass %d: no gap patches downloaded; stopping",
                        tile, pass_idx)
            break

        gap_data, gap_wcov = _swarp_gap_mosaic(
            patch_imgs,
            hdr=hdr,
            swarp_cfg_path=swarp_cfg_path,
            patch_dir=patch_dir,
            tag=f"{tile}_gapfill_{imgtype}_pass{pass_idx}",
            ncores=ncores,
        )
        if cleanup:
            for p in patch_imgs:
                try:
                    os.remove(p)
                except FileNotFoundError:
                    pass

        # Coverage of the gap mosaic. For science images a pixel only counts
        # as filled when it carries real (non-zero) data: SWarp can leave
        # weight>0 with data==0 at resampled brick edges, and trusting the
        # weight there would mark truly-empty pixels as "filled" so later
        # passes skip them. For masks a value of 0 is legitimate, so the
        # weight map is the only reliable coverage indicator.
        if imgtype == "image":
            new_cov = np.isfinite(gap_data) & (gap_data != 0)
        else:
            new_cov = gap_wcov
        fill_mask = remaining & new_cov
        n_new = int(fill_mask.sum())
        if n_new == 0:
            log.info("[%s] pass %d added no coverage; stopping", tile, pass_idx)
            summary["passes"] = pass_idx
            break
        data[fill_mask] = gap_data[fill_mask].astype(data.dtype)
        filled_region |= fill_mask
        summary["passes"] = pass_idx
        log.info("[%s] pass %d filled %d pixels", tile, pass_idx, n_new)

    if total_patches and summary["passes"]:
        fits.writeto(image_path, data, hdr, overwrite=True)
        summary["filled"] = True
    summary["n_centers"] = total_centers
    summary["n_patches"] = total_patches
    summary["gap_fraction_after"] = float((want & ~filled_region).mean())
    log.info("[%s] %s gap fraction %.4f -> %.4f (%d passes, %d patches)",
             tile, os.path.basename(image_path), gap_frac,
             summary["gap_fraction_after"], summary["passes"], total_patches)

    if cleanup:
        try:
            os.rmdir(patch_dir)
        except OSError:
            pass

    return summary


__all__ = [
    "DELVE_SIA_URL",
    "deg_to_hms_dms",
    "build_delve_detection_image",
    "build_patch_centers",
    "download_delve_patches",
    "fill_delve_detection_gaps",
]
