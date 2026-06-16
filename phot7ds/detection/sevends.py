"""
Build a 7DS native multi-band "white" detection image with SWarp.

All single-band coadds of a tile already share the same WCS grid, so SWarp
simply stacks them onto that frame. The companion ``*_coadd_weight.fits``
maps are used as inverse-variance weights so noisier bands/pixels
contribute less to the combined image.

Combine modes (``combine_type``):

* ``"WEIGHTED"`` - weighted mean -> a true "white" image (default).
* ``"CHI-MEAN"`` - chi-mean -> optimal multi-band *detection* image for
  source finding (feeds SExtractor / SE++ directly).
* ``"MEDIAN"`` - unweighted median; weight maps are neither required nor
  used, so images without a ``*_weight.fits`` sibling are stacked too.
* ``"AVERAGE"`` - unweighted mean (weight maps still required/used as
  with ``"WEIGHTED"``).

The output mirrors the DELVE builder's contract: it returns
``(detection_image, detection_weight)`` and records provenance in the
FITS header, including the full list of stacked input images.
"""
from __future__ import annotations

import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from glob import glob

from astropy.io import fits

log = logging.getLogger(__name__)

DEFAULT_BROAD_BANDS = ("g", "r", "i")
MEDIUM_BAND_PREFIX = "m"
DEFAULT_SEEING_KEYS = ("SEEING", "FWHM")


def _band_token(image_path: str) -> str:
    """Return the band field of a ``T#####_<band>_7DT..._coadd.fits`` name."""
    return os.path.basename(image_path).split("_")[1]


def _seeing_value(image_path: str, seeing_keys: tuple[str, ...]) -> float | None:
    """Return the first finite seeing/FWHM header value, or ``None``.

    Smaller is sharper. ``None`` means no usable keyword was found, in
    which case the per-band selection is skipped for this image.
    """
    try:
        hdr = fits.getheader(image_path)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("Could not read header for %s: %s", os.path.basename(image_path), exc)
        return None
    for key in seeing_keys:
        if key in hdr:
            try:
                val = float(hdr[key])
            except (TypeError, ValueError):
                continue
            if val > 0 and val == val:  # finite & positive
                return val
    return None


def _read_seeings(
    images: list[str], seeing_keys: tuple[str, ...], n_workers: int = 1
) -> list[float | None]:
    """Read the seeing value for each image (serial by default).

    Header reads are I/O-bound; pass ``n_workers > 1`` to parallelize for
    pathological directories with very many frames per band. Order is
    preserved to stay aligned with ``images``.
    """
    if n_workers <= 1:
        return [_seeing_value(p, seeing_keys) for p in images]
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        return list(pool.map(lambda p: _seeing_value(p, seeing_keys), images))


def _select_sharpest_per_band(
    images: list[str],
    weights: list[str | None],
    seeing_keys: tuple[str, ...],
    n_workers: int = 1,
) -> tuple[list[str], list[str | None]]:
    """Keep one representative (sharpest-seeing) image per band.

    Bands whose images have no usable seeing keyword are kept in full
    (selection is skipped for them) so coverage is never silently lost.
    Reads are serial by default; raise ``n_workers`` for huge directories.
    """
    seeings = _read_seeings(images, seeing_keys, n_workers=n_workers)

    by_band: dict[str, list[tuple[str, str | None, float | None]]] = {}
    for img, wgt, see in zip(images, weights, seeings):
        by_band.setdefault(_band_token(img), []).append((img, wgt, see))

    sel_images: list[str] = []
    sel_weights: list[str | None] = []
    for band in sorted(by_band):
        entries = by_band[band]
        with_seeing = [e for e in entries if e[2] is not None]
        if not with_seeing:
            # No seeing info for this band -> skip selection, keep them all.
            log.warning(
                "Band %s: no %s keyword on %d image(s); keeping all (no selection).",
                band, "/".join(seeing_keys), len(entries),
            )
            for img, wgt, _ in entries:
                sel_images.append(img)
                sel_weights.append(wgt)
            continue
        best = min(with_seeing, key=lambda e: e[2])
        log.info(
            "Band %s: selected %s (seeing=%.3f) out of %d image(s).",
            band, os.path.basename(best[0]), best[2], len(entries),
        )
        sel_images.append(best[0])
        sel_weights.append(best[1])
    return sel_images, sel_weights


def collect_band_inputs(
    image_dir: str,
    *,
    image_glob: str = "*_coadd.fits",
    weight_suffix: str = "_weight",
    medium_only: bool = False,
    one_per_band: bool = False,
    seeing_keys: tuple[str, ...] = DEFAULT_SEEING_KEYS,
    n_workers: int = 1,
    require_weights: bool = True,
) -> tuple[list[str], list[str | None]]:
    """Return aligned ``(images, weights)`` path lists for a tile directory.

    With ``require_weights=True`` (default) only science coadds that have a
    sibling weight map are kept, so the image/weight ordering stays
    consistent for SWarp. With ``require_weights=False`` (e.g. for an
    unweighted ``MEDIAN`` combine) images without a weight map are kept
    too, with ``None`` in the corresponding ``weights`` slot.

    When ``medium_only`` is set, broad-band images (``g``/``r``/``i``) are
    dropped and only medium bands (``m###``) are stacked.

    When ``one_per_band`` is set, only a single representative image is
    kept per band: the one with the smallest (sharpest) seeing, read from
    the first available header keyword in ``seeing_keys`` (``SEEING`` then
    ``FWHM`` by default). If a band has no usable seeing keyword on any of
    its images, selection is skipped for that band and all its images are
    kept.
    """
    images: list[str] = []
    weights: list[str | None] = []
    for img in sorted(glob(os.path.join(image_dir, image_glob))):
        if img.endswith(f"{weight_suffix}.fits"):
            continue  # defensive: never treat a weight map as a science image
        if medium_only and not _band_token(img).startswith(MEDIUM_BAND_PREFIX):
            continue
        stem, ext = os.path.splitext(img)
        wgt = f"{stem}{weight_suffix}{ext}"
        if not os.path.exists(wgt):
            if require_weights:
                log.warning("No weight map for %s; skipping", os.path.basename(img))
                continue
            log.info("No weight map for %s; keeping (weights not required)",
                     os.path.basename(img))
            wgt = None
        images.append(img)
        weights.append(wgt)

    if one_per_band and images:
        images, weights = _select_sharpest_per_band(
            images, weights, seeing_keys, n_workers=n_workers
        )

    return images, weights


def _reference_frame(image_path: str) -> tuple[float, float, float, int, int]:
    """Read ``(ra, dec, pixscale_arcsec, nx, ny)`` from a reference header."""
    h = fits.getheader(image_path)
    ra = float(h["CRVAL1"])
    dec = float(h["CRVAL2"])
    if "CD1_1" in h:
        pixscale = abs(float(h["CD1_1"])) * 3600.0
    elif "CDELT1" in h:
        pixscale = abs(float(h["CDELT1"])) * 3600.0
    else:
        raise KeyError(f"No CD1_1 / CDELT1 in {os.path.basename(image_path)}")
    return ra, dec, pixscale, int(h["NAXIS1"]), int(h["NAXIS2"])

#%%
def build_7ds_detection_image(
    *,
    image_dir: str,
    output_path: str,
    swarp_cfg_path: str,
    tile: str | None = None,
    output_name: str | None = None,
    medium_only: bool = False,
    one_per_band: bool = False,
    seeing_keys: tuple[str, ...] = DEFAULT_SEEING_KEYS,
    combine_type: str = "WEIGHTED",
    resampling_type: str = "LANCZOS3",
    subtract_back: bool = False,
    ra_center: float | str | None = None,
    dec_center: float | str | None = None,
    pixscale_arcsec: float | None = None,
    image_size_x: int | None = None,
    image_size_y: int | None = None,
    ncores: int = 8,
    seeing_workers: int = 1,
    image_glob: str = "*_coadd.fits",
    weight_suffix: str = "_weight",
    cleanup: bool = True,
    overwrite: bool = True,
) -> tuple[str, str]:
    """Stack a tile's single-band coadds into a white detection image.

    Parameters
    ----------
    image_dir
        Directory containing the per-band ``*_coadd.fits`` science images
        and their ``*_coadd_weight.fits`` weight maps (e.g. the symlink
        directory ``/data/.../RIS/data/<tile>``).
    output_path
        Output directory (created if missing).
    swarp_cfg_path
        Path to a SWarp config file (use
        :func:`phot7ds.ensure_swarp_config` to create one on demand).
    tile
        Tile identifier. Inferred from the first image name
        (``T#####_...``) when omitted.
    output_name
        Output image file name. Defaults to
        ``"{tile}_7DS_EDR_IMAGE_det.fits"`` (mirrors the DELVE builder).
    medium_only
        Stack only medium bands (``m###``); skip ``g``/``r``/``i``.
    one_per_band
        Keep a single representative image per band - the sharpest by
        seeing - instead of stacking every image in the directory. Useful
        when a tile has many repeat visits per band. Bands with no usable
        seeing keyword are kept in full (selection skipped).
    seeing_keys
        Header keywords tried in order to read the seeing/FWHM (smaller =
        sharper). Defaults to ``("SEEING", "FWHM")``.
    combine_type
        SWarp ``COMBINE_TYPE`` (see module docstring). Default
        ``"WEIGHTED"``. With ``"MEDIAN"`` the stack is unweighted: the
        ``*_weight.fits`` maps are neither required nor used
        (``WEIGHT_TYPE NONE``), so images without a weight map are
        stacked too.
    resampling_type
        SWarp ``RESAMPLING_TYPE`` (default ``"LANCZOS3"``).
    subtract_back
        Whether SWarp re-subtracts the background. ``False`` by default
        because 7DS coadds are already background-subtracted.
    ra_center, dec_center
        Output frame center. Taken from the reference image ``CRVAL1`` /
        ``CRVAL2`` when omitted (decimal degrees).
    pixscale_arcsec, image_size_x, image_size_y
        Output grid. Taken from the reference image header when omitted,
        which keeps the white image pixel-aligned with the inputs.
    ncores
        SWarp ``NTHREADS``.
    seeing_workers
        Threads used to read seeing headers (for ``one_per_band`` selection
        and header recording). Defaults to ``1`` (serial); raise it only
        for tiles with a very large number of frames.
    image_glob, weight_suffix
        Override the science-image glob / weight-map suffix conventions.
    cleanup
        Remove the temporary SWarp input list afterwards.
    overwrite
        Reuse an existing output image when ``False`` and it is present.

    Returns
    -------
    detection_image, detection_weight
        Paths to the output FITS image and its weight map.
    """
    # MEDIAN is an unweighted combine: weight maps are neither required
    # (images without one are stacked too) nor passed to SWarp.
    use_weights = combine_type.upper() != "MEDIAN"

    images, weights = collect_band_inputs(
        image_dir,
        image_glob=image_glob,
        weight_suffix=weight_suffix,
        medium_only=medium_only,
        one_per_band=one_per_band,
        seeing_keys=seeing_keys,
        n_workers=seeing_workers,
        require_weights=use_weights,
    )
    if not images:
        raise FileNotFoundError(
            f"No {'image/weight pairs' if use_weights else 'images'} found in "
            f"{image_dir} (medium_only={medium_only}, glob={image_glob!r})"
        )

    if tile is None:
        tile = os.path.basename(images[0]).split("_")[0]
    if output_name is None:
        output_name = f"{tile}_7DS_EDR_IMAGE_det.fits"

    os.makedirs(output_path, exist_ok=True)
    out_img = os.path.join(output_path, output_name)
    out_wgt = f"{os.path.splitext(out_img)[0]}_weight.fits"

    if not overwrite and os.path.exists(out_img):
        log.info("Reusing existing 7DS detection image: %s", out_img)
        return out_img, out_wgt

    bands = [_band_token(p) for p in images]
    log.info(
        "Building 7DS white detection image for %s: %d bands (%s), COMBINE_TYPE=%s",
        tile, len(bands), "medium-only" if medium_only else "all", combine_type,
    )

    # Lock the output frame to the shared tile grid from a reference header.
    ref_ra, ref_dec, ref_pix, ref_nx, ref_ny = _reference_frame(images[0])
    ra_c = ref_ra if ra_center is None else ra_center
    dec_c = ref_dec if dec_center is None else dec_center
    pix = ref_pix if pixscale_arcsec is None else pixscale_arcsec
    nx = ref_nx if image_size_x is None else image_size_x
    ny = ref_ny if image_size_y is None else image_size_y

    list_file = os.path.join(output_path, f"{tile}_7DS_input.list")
    with open(list_file, "w") as fh:
        fh.write("\n".join(images) + "\n")

    swarp_args = [
        "SWarp", f"@{list_file}",
        "-c", swarp_cfg_path,
        "-IMAGEOUT_NAME", out_img,
        "-WEIGHTOUT_NAME", out_wgt,
    ]
    if use_weights:
        swarp_args += [
            "-WEIGHT_TYPE", "MAP_WEIGHT",
            "-WEIGHT_IMAGE", ",".join(weights),
        ]
    else:
        swarp_args += ["-WEIGHT_TYPE", "NONE"]
    swarp_args += [
        "-COMBINE", "Y",
        "-COMBINE_TYPE", combine_type,
        "-RESAMPLE", "Y",
        "-RESAMPLE_DIR", output_path,
        "-RESAMPLING_TYPE", resampling_type,
        "-SUBTRACT_BACK", "Y" if subtract_back else "N",
        "-CENTER_TYPE", "MANUAL",
        "-CENTER", f"{ra_c},{dec_c}",
        "-PIXELSCALE_TYPE", "MANUAL",
        "-PIXEL_SCALE", f"{float(pix):.6f}",
        "-IMAGE_SIZE", f"{nx},{ny}",
        # Keep counts as-is; do not rescale fluxes between bands.
        "-FSCALASTRO_TYPE", "NONE",
        "-FSCALE_KEYWORD", "NONE",
        "-DELETE_TMPFILES", "Y",
        "-WRITE_XML", "N",
        "-VERBOSE_TYPE", "NORMAL",
        "-NTHREADS", str(ncores),
    ]

    proc = subprocess.run(swarp_args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"SWarp failed for {tile}: {proc.stderr}")
    if not os.path.exists(out_img):
        raise RuntimeError(f"SWarp finished but output missing: {out_img}")

    # Seeing of each stacked frame, recorded in the output header.
    seeings = _read_seeings(images, seeing_keys, n_workers=seeing_workers)
    _annotate_header(out_img, tile=tile, bands=bands, images=images,
                     seeings=seeings, combine_type=combine_type,
                     medium_only=medium_only, one_per_band=one_per_band)

    if cleanup and os.path.exists(list_file):
        os.remove(list_file)

    log.info("Built 7DS white detection image: %s", out_img)
    return out_img, out_wgt


def _annotate_header(
    image_path: str,
    *,
    tile: str,
    bands: list[str],
    images: list[str],
    seeings: list[float | None],
    combine_type: str,
    medium_only: bool,
    one_per_band: bool = False,
) -> None:
    """Record provenance + the full input image list (with seeing) in the header."""
    with fits.open(image_path, mode="update") as hdul:
        hdr = hdul[0].header
        hdr["DETTYPE"] = ("7DS-white", "7DS multi-band detection image")
        hdr["COMBTYPE"] = (combine_type, "SWarp combine type")
        hdr["MEDONLY"] = (bool(medium_only), "Medium-band-only stack")
        hdr["ONEPRBND"] = (bool(one_per_band), "One sharpest image per band")
        hdr["NBANDS"] = (len(bands), "Number of bands combined")
        # Long comma-list (astropy uses CONTINUE cards transparently).
        hdr["BANDS"] = (",".join(bands), "Bands combined")
        hdr["NIMAGES"] = (len(images), "Number of input images stacked")
        # Per-frame seeing summary over the frames that report it.
        good = [s for s in seeings if s is not None]
        if good:
            hdr["SEEMIN"] = (round(min(good), 4), "Min seeing of stacked frames [arcsec]")
            hdr["SEEMAX"] = (round(max(good), 4), "Max seeing of stacked frames [arcsec]")
            hdr["SEEMED"] = (
                round(float(sorted(good)[len(good) // 2]), 4),
                "Median seeing of stacked frames [arcsec]",
            )
        # One pair of cards per stacked image: DETIMGnn holds the file name
        # and DETSEEnn the matching seeing (kept separate because long file
        # names leave no room for seeing in the DETIMG card comment).
        for i, (img, see) in enumerate(zip(images, seeings), start=1):
            hdr[f"DETIMG{i:02d}"] = (os.path.basename(img), f"input image {i}")
            if see is not None:
                hdr[f"DETSEE{i:02d}"] = (round(float(see), 4), "seeing [arcsec]")
        hdul.flush()


__all__ = [
    "build_7ds_detection_image",
    "collect_band_inputs",
]
