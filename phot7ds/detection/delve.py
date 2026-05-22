"""
Build a DELVE-DR3 detection (or mask) image for a 7DT tile.

The tile field of view is partitioned into a grid of patches; each patch is
queried via the NOIRLab SIA service for the requested band/product type,
downloaded, and finally co-added with SWarp into a single mosaic image at the
target tile center, pixel scale and image size.
"""
from __future__ import annotations

import logging
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from typing import Any, Iterable, Literal

import astropy.units as u
import numpy as np
import requests
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from pyvo.dal import SIAService

log = logging.getLogger(__name__)

DELVE_SIA_URL = "https://datalab.noirlab.edu/sia/delve_dr3"


def _get_tile_value(tile_info: Any, key: str):
    if isinstance(tile_info, Table):
        if len(tile_info) == 0:
            raise ValueError("tile_info is empty")
        return tile_info[key][0]
    return tile_info[key]


def deg_to_hms_dms(ra_deg: float, dec_deg: float) -> tuple[str, str]:
    """Convert decimal degrees to SWarp / FITS sexagesimal strings.

    RA is returned as ``HH:MM:SS.ss`` (hours); Dec as ``[+|-]DD:MM:SS.ss``.
    Matches the convention used in ``Utils_7DT.deg_to_hms_dms``.
    """
    ra_deg = float(ra_deg)
    dec_deg = float(dec_deg)

    ra_hours = ra_deg / 15.0
    rh = int(ra_hours)
    rm = int((ra_hours - rh) * 60)
    rs = ((ra_hours - rh) * 60 - rm) * 60
    ra_str = f"{rh:02d}:{rm:02d}:{rs:05.2f}"

    sign = "+" if dec_deg >= 0 else "-"
    dec_abs = abs(dec_deg)
    dd = int(dec_abs)
    dm = int((dec_abs - dd) * 60)
    ds = ((dec_abs - dd) * 60 - dm) * 60
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


def build_patch_centers(
    tile_info: Any,
    *,
    n_cols: int = 9,
    n_rows: int = 6,
) -> list[tuple[float, float]]:
    """Generate ``(ra, dec)`` patch centers tiling the field of view.

    The four corner positions ``(ra1..ra4, dec1..dec4)`` of ``tile_info``
    define the bounding rectangle, which is divided into a ``n_cols x n_rows``
    grid of equal-area patches.
    """
    ra = [float(_get_tile_value(tile_info, f"ra{i}")) for i in (1, 2, 3, 4)]
    dec = [float(_get_tile_value(tile_info, f"dec{i}")) for i in (1, 2, 3, 4)]
    ra_min, ra_max = min(ra), max(ra)
    dec_min, dec_max = min(dec), max(dec)
    ra_span = ra_max - ra_min
    dec_span = dec_max - dec_min

    ra_centers = [ra_min + (2 * i + 1) * ra_span / (2 * n_cols) for i in range(n_cols)]
    dec_centers = [dec_max - (2 * i + 1) * dec_span / (2 * n_rows) for i in range(n_rows)]

    centers: list[tuple[float, float]] = []
    for r in range(n_rows):
        for c in range(n_cols):
            centers.append((ra_centers[c], dec_centers[r]))
    return centers


def build_delve_detection_image(
    *,
    tile_info: Any,
    ra_center: str | float | None = None,
    dec_center: str | float | None = None,
    imgtype: Literal["image", "mask"],
    output_path: str,
    swarp_cfg_path: str,
    detection_band: str = "det",
    n_cols: int = 9,
    n_rows: int = 6,
    patch_size_deg: float = 0.25,
    ncores: int = 12,
    sia_url: str = DELVE_SIA_URL,
    combine_type: str = "MAX",
    pixscale_arcsec: float = 0.505,
    image_size_x: int = 10200,
    image_size_y: int = 6800,
    cleanup_patches: bool = True,
    request_timeout: float = 120,
    max_retries: int = 3,
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
    n_cols, n_rows, patch_size_deg
        Patch grid geometry.
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

    centers = build_patch_centers(tile_info, n_cols=n_cols, n_rows=n_rows)
    n_patches = len(centers)
    thread_local = threading.local()

    def _clients() -> tuple[SIAService, requests.Session]:
        if not hasattr(thread_local, "sia_service"):
            thread_local.sia_service = SIAService(sia_url)
        if not hasattr(thread_local, "http_session"):
            thread_local.http_session = requests.Session()
        return thread_local.sia_service, thread_local.http_session

    def _query_and_download(args: tuple[int, float, float]) -> tuple[int, str, bool]:
        patch_idx, ra, dec = args
        for attempt in range(1, max_retries + 1):
            try:
                sia_service, http_session = _clients()
                position = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
                size = patch_size_deg * u.deg
                results = sia_service.search(pos=position, size=size)
                table = results.to_table()
                if len(table) == 0:
                    return patch_idx, "No SIA rows returned.", False

                prodtype = np.asarray(table["prodtype"]).astype(str)
                bandpass = np.asarray(table["obs_bandpass"]).astype(str)
                publisher_did = np.asarray(table["obs_publisher_did"]).astype(str)
                row_mask = (
                    (prodtype == imgtype)
                    & (bandpass == detection_band)
                    & (np.char.find(publisher_did, "_nobkg") < 0)
                )
                if not np.any(row_mask):
                    return patch_idx, "No matching DELVE patch found.", False

                match = table[row_mask][0]
                download_url = str(match["access_url"])
                exptime = int(float(match["exptime"]))
                patch_img = (
                    f"{output_path}/DELVE{imgtype.upper()}_{tile}_{detection_band}_"
                    f"patch{patch_idx:02d}_{ra:.4f}{dec:.4f}_"
                    f"{patch_size_deg:.2f}x{patch_size_deg:.2f}_{exptime}sec.fits"
                )
                response = http_session.get(
                    download_url, stream=True, timeout=(15, request_timeout)
                )
                response.raise_for_status()
                with open(patch_img, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)
                return patch_idx, os.path.basename(patch_img), True
            except Exception as exc:
                if attempt == max_retries:
                    return patch_idx, f"Patch download failed: {exc}", False
                continue
        return patch_idx, "Patch download failed after retries", False

    tasks = [(idx, ra, dec) for idx, (ra, dec) in enumerate(centers, 1)]
    n_workers = max(1, min(ncores, n_patches))
    log.info("[%s] launching %d workers for %d patches", tile, n_workers, n_patches)

    n_success = 0
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_query_and_download, t): t[0] for t in tasks}
        for future in as_completed(futures):
            idx = futures[future]
            _, message, success = future.result()
            log.info("[%s] patch %d/%d: %s", tile, idx, n_patches, message)
            if success:
                n_success += 1

    log.info("[%s] patch download summary: %d/%d", tile, n_success, n_patches)

    patch_imgs = sorted(
        glob(
            f"{output_path}/DELVE{imgtype.upper()}_{tile}_{detection_band}_patch*.fits"
        )
    )
    if not patch_imgs:
        raise RuntimeError(f"[{tile}] no patch images downloaded")

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
    # clean the list file and weight file
    os.remove(list_file)
    os.remove(f"{output_path}/{tile}_DELVE_DR3_{imgtype.upper()}_det_weight.fits")

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


__all__ = [
    "DELVE_SIA_URL",
    "deg_to_hms_dms",
    "build_delve_detection_image",
    "build_patch_centers",
]
