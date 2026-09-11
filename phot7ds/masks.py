"""
Per-band mask flags from py7DT count-map MEFs.

The 7DT image pipeline writes, next to every coadd, a multi-extension FITS
``<coadd>_counts.fits`` whose extensions are ``uint8`` *count maps*: how many
input frames were affected by a given condition at each pixel
(``NGEOM``, ``NUSED``, ``NBAD``, ``NSAT``, ``NTRAIL``, ``NOUTLIER``, and the
reserved ``NHOT``/``NDEAD``/``NSTRAY``). SourceExtractor++ cannot read those
files directly -- the extensions are tile-compressed and ``[EXTNAME]`` selectors
are silently ignored -- and feeding six planes per band would add hundreds of
catalog columns. This module therefore folds each MEF into **one bitmask per
band**, staged in tmpfs for the duration of the SE++ run:

===  ==========  =================================================
bit  name        set where
===  ==========  =================================================
1    OUTLIER     ``NOUTLIER > 0`` (clipped-coaddition rejection)
2    BADPIX      ``NBAD > 0``     (detector bad-pixel mask)
4    STRAY       ``NSTRAY > 0``   (reserved by py7DT)
8    SATELLITE   ``NTRAIL > 0``   (satellite-trail mask)
16   SATURATED   ``NSAT > 0``
32   HOT         ``NHOT > 0``     (reserved)
64   DEAD        ``NDEAD > 0``    (reserved)
128  NODATA      no data in *this band*: coadd pixel is 0 or non-finite,
                 or ``NUSED == 0`` in the band's MEF
===  ==========  =================================================

Bits 1-64 follow ``pipeline.imcoadd.const.MaskBit`` of the 7DT pipeline
(https://github.com/7DimensionalTelescope/pipeline); 128 is a phot7ds
extension. Milder conditions have lower bits, so ``mask_flags < 4`` keeps
sources touched only by an outlier or a detector bad pixel.

SE++ evaluates each flag image over the source's detection isophote with
``flag-type or``, producing ``isophotal_image_flags_<band>`` (the OR of the
bitmask) and ``isophotal_image_flags_pixel_count_<band>`` (number of isophote
pixels with a non-zero mask). :func:`rename_mask_columns` turns these into
``mask_flags_<band>`` / ``mask_npix_<band>``.

When a band has no count-map MEF the bitmask still carries bit 128 from the
coadd itself, so every band gets a ``mask_flags`` column and the legacy union
coverage column is no longer needed.
"""
from __future__ import annotations

import logging
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from astropy.io import fits
from astropy.table import Table

log = logging.getLogger(__name__)

# py7DT pipeline/imcoadd/const.py::MaskBit
MASKBIT: dict[str, int] = {
    "OUTLIER": 1,
    "BADPIX": 2,
    "STRAY": 4,
    "SATELLITE": 8,
    "SATURATED": 16,
    "HOT": 32,
    "DEAD": 64,
}
NODATA_BIT = 128
BIT_NAMES: dict[int, str] = {**{v: k for k, v in MASKBIT.items()}, NODATA_BIT: "NODATA"}

# py7DT pipeline/imcoadd/counts.py: count plane (EXTNAME) -> bit it feeds.
PLANE_TO_BIT: dict[str, int] = {
    "NOUTLIER": MASKBIT["OUTLIER"],
    "NBAD": MASKBIT["BADPIX"],
    "NSTRAY": MASKBIT["STRAY"],
    "NTRAIL": MASKBIT["SATELLITE"],
    "NSAT": MASKBIT["SATURATED"],
    "NHOT": MASKBIT["HOT"],
    "NDEAD": MASKBIT["DEAD"],
}
NODATA_PLANE = "NUSED"

COUNT_MASK_SUFFIX = "_counts.fits"
MASK_FLAGS_PREFIX = "mask_flags_"
MASK_NPIX_PREFIX = "mask_npix_"
SEPP_FLAGS_PREFIX = "isophotal_image_flags_"
SEPP_NPIX_PREFIX = "isophotal_image_flags_pixel_count_"
STAGING_PREFIX = "phot7ds_masks_"


def mask_flags_column(band: str) -> str:
    return f"{MASK_FLAGS_PREFIX}{band}"


def mask_npix_column(band: str) -> str:
    return f"{MASK_NPIX_PREFIX}{band}"


def maskbits_description() -> str:
    """``'1=OUTLIER,2=BADPIX,...,128=NODATA'``."""
    return ",".join(f"{bit}={name}" for bit, name in sorted(BIT_NAMES.items()))


# ---------------------------------------------------------------------------
# Locating count-map MEFs
# ---------------------------------------------------------------------------
def _image_stem(path: str | Path) -> str:
    name = Path(path).name
    for ext in (".fits.fz", ".fits.gz", ".fits"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return Path(name).stem


def find_count_mask(
    science_image: str | Path,
    *,
    mask_dir: str | Path | None = None,
    suffix: str = COUNT_MASK_SUFFIX,
) -> str | None:
    """Return the count-map MEF for ``science_image`` or ``None``.

    Looks for ``<image stem><suffix>`` in ``mask_dir`` (if given), next to the
    image, and -- when the image is a symlink -- next to its target. The
    pipeline convention is ``<coadd>_counts.fits`` in the coadd directory.
    """
    p = Path(science_image)
    name = f"{_image_stem(p)}{suffix}"
    candidates: list[Path] = []
    if mask_dir:
        candidates.append(Path(mask_dir) / name)
    candidates.append(p.with_name(name))
    real = Path(os.path.realpath(p))
    if real.parent != p.parent:
        candidates.append(real.with_name(name))
    for c in candidates:
        if c.is_file():
            return str(c)
    return None


def _band_of_mef(path: str) -> str | None:
    try:
        with fits.open(path) as hdul:
            for hdu in hdul:
                band = hdu.header.get("FILTER")
                if band:
                    return str(band).strip().replace("-", "_")
    except Exception:
        return None
    return None


def resolve_count_masks(
    science_images: Sequence[str],
    band_names: Sequence[str],
    count_masks: Mapping[str, str] | Sequence[str] | str | Path | None = None,
    *,
    suffix: str = COUNT_MASK_SUFFIX,
) -> dict[str, str | None]:
    """Map every band to its count-map MEF (or ``None``).

    Parameters
    ----------
    science_images, band_names
        Parallel sequences (one band name per image), as produced by
        :func:`phot7ds.images.extract_band_names_and_saturation`.
    count_masks
        * ``None`` -- auto-discover ``<coadd stem><suffix>`` next to each
          image (see :func:`find_count_mask`);
        * a directory -- look for the same names inside it;
        * a mapping ``{band: path}``;
        * a sequence of paths -- matched to bands by the ``FILTER`` header of
          the MEF, falling back to a ``_<band>_`` token in the filename.

    Bands without a MEF map to ``None``; the caller then builds a
    coverage-only bitmask (bit 128 alone).
    """
    bands = list(band_names)
    images = list(science_images)
    out: dict[str, str | None] = {b: None for b in bands}

    if isinstance(count_masks, Mapping):
        for band, path in count_masks.items():
            if band in out and path:
                out[band] = str(path)
            elif band not in out:
                log.warning("count_masks: band %r is not among the measurement bands; ignored.", band)
    elif isinstance(count_masks, (str, Path)) and Path(count_masks).is_dir():
        for band, img in zip(bands, images):
            out[band] = find_count_mask(img, mask_dir=count_masks, suffix=suffix)
    elif count_masks is None:
        for band, img in zip(bands, images):
            out[band] = find_count_mask(img, suffix=suffix)
    else:
        paths = [str(count_masks)] if isinstance(count_masks, (str, Path)) else [str(p) for p in count_masks]
        for path in paths:
            band = _band_of_mef(path)
            if band not in out:
                band = next((b for b in bands if f"_{b}_" in Path(path).name), None)
            if band is None:
                log.warning("count_masks: could not match %s to a band; ignored.", path)
                continue
            out[band] = path

    n_found = sum(1 for v in out.values() if v)
    log.info("Count-map MEFs: %d/%d bands (%s without: coverage-only bitmask)",
             n_found, len(out), ",".join(b for b, v in out.items() if not v) or "none")
    return out


# ---------------------------------------------------------------------------
# Building one bitmask
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BandMaskInfo:
    """What went into one band's bitmask."""

    band: str
    image: str
    count_mask: str | None
    bitmask: str
    status: str                                   # "mef" or "coverage-only[: reason]"
    planes: tuple[str, ...] = ()                  # count planes folded in
    missing_planes: tuple[str, ...] = ()          # known planes absent from the MEF
    nodata_sources: tuple[str, ...] = ()
    per_bit_pixels: dict[int, int] = field(default_factory=dict)
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "band": self.band, "image": self.image, "count_mask": self.count_mask,
            "status": self.status, "planes": list(self.planes),
            "missing_planes": list(self.missing_planes),
            "nodata_sources": list(self.nodata_sources),
            "per_bit_pixels": {str(k): v for k, v in self.per_bit_pixels.items()},
            "seconds": self.seconds,
        }


def _fold_count_mask(
    count_mask: str, shape: tuple[int, ...], *, use_nused: bool
) -> tuple[np.ndarray, np.ndarray | None, list[str], list[str]]:
    """Read the MEF into ``(defect bits, NUSED==0 mask or None, planes, missing)``.

    Works on private arrays so a failure part-way leaves the caller's
    bitmask untouched.
    """
    bm = np.zeros(shape, dtype=np.uint8)
    nused0: np.ndarray | None = None
    planes: list[str] = []
    with fits.open(count_mask) as hdul:
        for hdu in hdul[1:] if len(hdul) > 1 else hdul:
            name = str(hdu.name).upper()
            if hdu.data is None:
                continue
            if name == NODATA_PLANE:
                if use_nused:
                    data = np.asarray(hdu.data)
                    if data.shape != shape:
                        raise ValueError(f"[{name}] shape {data.shape} != image {shape}")
                    nused0 = data == 0
                continue
            bit = PLANE_TO_BIT.get(name)
            if bit is None:
                continue
            data = np.asarray(hdu.data)
            if data.shape != shape:
                raise ValueError(f"[{name}] shape {data.shape} != image {shape}")
            if not np.issubdtype(data.dtype, np.integer):
                raise ValueError(f"[{name}] is {data.dtype}, expected an integer count map")
            bm[data > 0] |= np.uint8(bit)
            planes.append(name)
    if not planes and nused0 is None:
        raise ValueError("no known count planes (NBAD, NSAT, NTRAIL, NOUTLIER, ...) found")
    missing = [p for p in PLANE_TO_BIT if p not in planes]
    return bm, nused0, planes, missing


def build_band_bitmask(
    band: str,
    image: str,
    count_mask: str | None,
    out_path: str,
    expected_shape: tuple[int, int] | None = None,
    *,
    use_nused: bool = True,
    nodata_bit: int = NODATA_BIT,
) -> BandMaskInfo:
    """Write the ``uint8`` bitmask for one band and describe it.

    Bit 128 is always derived from the band's coadd (``== 0`` or non-finite);
    the count-map MEF adds the defect bits and, with ``use_nused``, its
    ``NUSED == 0`` pixels. If the MEF is missing or cannot be used (unreadable,
    wrong shape, no known planes) the bitmask degrades to coverage-only and
    the reason is recorded in ``status`` -- the run does not stop.

    Raises :class:`ValueError` only when the *coadd* is not on the detection
    grid (``expected_shape``), because then no flag image can be built.
    """
    t0 = time.time()
    with fits.open(image, memmap=True) as hdul:
        img = hdul[0].data
        if img is None:
            raise ValueError(f"{image}: primary HDU has no data")
        if expected_shape is not None and tuple(img.shape) != tuple(expected_shape):
            raise ValueError(f"{image}: shape {img.shape} != detection grid {tuple(expected_shape)}")
        nodata = ~np.isfinite(img)
        nodata |= img == 0
        shape = img.shape
    bm = np.zeros(shape, dtype=np.uint8)
    nodata_sources = ["coadd==0|nonfinite"]

    planes: list[str] = []
    missing: list[str] = list(PLANE_TO_BIT)
    status = "coverage-only"
    if count_mask:
        try:
            bm_mef, nused0, planes, missing = _fold_count_mask(count_mask, shape, use_nused=use_nused)
            bm |= bm_mef
            if nused0 is not None:
                nodata |= nused0
                nodata_sources.append("NUSED==0")
            status = "mef"
        except Exception as exc:  # degrade, do not abort the run
            log.warning("Band %s: count mask %s ignored (%s); coverage-only bitmask.", band, count_mask, exc)
            planes, missing = [], list(PLANE_TO_BIT)
            status = f"coverage-only: {exc}"
            count_mask = None

    per_bit = {int(bit): int(np.count_nonzero(bm & bit)) for bit in PLANE_TO_BIT.values() if np.any(bm & bit)}
    bm[nodata] |= np.uint8(nodata_bit)
    per_bit[int(nodata_bit)] = int(np.count_nonzero(nodata))

    hdr = fits.Header()
    hdr["BUNIT"] = ("bitmask", f"OR of mask bits; {nodata_bit} = no data in this band")
    hdr["BAND"] = (band, "measurement band this mask belongs to")
    # Long file names leave no room for a comment on the same card.
    hdr["SRCIMG"] = Path(image).name[:68]
    hdr.comments["SRCIMG"] = "coadd (no-data source)" if len(Path(image).name) < 40 else ""
    if count_mask:
        hdr["SRCMEF"] = Path(count_mask).name[:68]
        hdr.comments["SRCMEF"] = "count-map MEF" if len(Path(count_mask).name) < 48 else ""
    for name, bit in MASKBIT.items():
        hdr[f"MB{name[:6]}"] = (bit, f"{name} bit")
    hdr["MBNODATA"] = (nodata_bit, "no data: " + "|".join(nodata_sources))
    hdr["PLANES"] = (",".join(planes), "count planes folded in")
    hdr["NOTEVAL"] = (",".join(missing)[:60], "planes not in MEF")
    kind, _, reason = status.partition(": ")
    hdr["MSKSTAT"] = (kind, "mef | coverage-only")
    if reason:
        hdr["MSKWHY"] = reason[:68]
    fits.PrimaryHDU(bm, hdr).writeto(out_path, overwrite=True)

    return BandMaskInfo(
        band=band, image=str(image), count_mask=count_mask, bitmask=str(out_path),
        status=status, planes=tuple(planes), missing_planes=tuple(missing),
        nodata_sources=tuple(nodata_sources), per_bit_pixels=per_bit,
        seconds=round(time.time() - t0, 2),
    )


def _build_job(args: tuple) -> BandMaskInfo:
    band, image, count_mask, out_path, expected_shape, use_nused = args
    return build_band_bitmask(band, image, count_mask, out_path, expected_shape, use_nused=use_nused)


# ---------------------------------------------------------------------------
# Staging in tmpfs
# ---------------------------------------------------------------------------
def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def sweep_stale_staging(root: str | Path, prefix: str = STAGING_PREFIX) -> list[str]:
    """Remove ``<root>/<prefix>*_<pid>`` directories whose owner process is gone.

    A crashed or killed run cannot execute its ``finally`` block; the next run
    calls this so tmpfs does not fill up with orphaned bitmasks.
    """
    root = Path(root)
    removed: list[str] = []
    if not root.is_dir():
        return removed
    for d in root.glob(f"{prefix}*"):
        if not d.is_dir():
            continue
        try:
            pid = int(d.name.rsplit("_", 1)[-1])
        except ValueError:
            continue
        if pid == os.getpid() or _pid_alive(pid):
            continue
        shutil.rmtree(d, ignore_errors=True)
        removed.append(str(d))
    if removed:
        log.info("Removed %d stale mask staging dir(s) under %s", len(removed), root)
    return removed


class MaskStaging:
    """Build per-band bitmasks into a private directory and delete it afterwards.

    ``staging_dir`` (default ``/dev/shm``) is used when it exists and is
    writable; otherwise ``fallback_dir`` (normally the run's work directory).
    Use as a context manager, or call :meth:`cleanup` in a ``finally``.
    """

    def __init__(
        self,
        run_name: str,
        *,
        staging_dir: str | Path | None = "/dev/shm",
        fallback_dir: str | Path | None = None,
        workers: int = 8,
        sweep_stale: bool = True,
    ):
        root: Path | None = None
        if staging_dir:
            cand = Path(staging_dir)
            if cand.is_dir() and os.access(cand, os.W_OK):
                root = cand
                if sweep_stale:
                    sweep_stale_staging(root)
            else:
                log.warning("Mask staging dir %s unavailable; using %s", staging_dir, fallback_dir or ".")
        if root is None:
            root = Path(fallback_dir) if fallback_dir else Path(".")
        self.root = root
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(run_name))[:80]
        self.dir = root / f"{STAGING_PREFIX}{safe}_{os.getpid()}"
        self.workers = max(1, int(workers))
        self.infos: dict[str, BandMaskInfo] = {}
        self.seconds: float | None = None

    @property
    def location(self) -> str:
        return "tmpfs" if str(self.root).startswith("/dev/shm") else "disk"

    def stage(
        self,
        band_images: Mapping[str, str],
        count_masks: Mapping[str, str | None],
        expected_shape: tuple[int, int] | None,
        *,
        use_nused: bool = True,
    ) -> dict[str, BandMaskInfo]:
        """Build one bitmask per band; bands off the detection grid are skipped."""
        self.dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        jobs = [
            (band, img, count_masks.get(band), str(self.dir / f"{band}_mask.fits"), expected_shape, use_nused)
            for band, img in band_images.items()
        ]
        results: list[BandMaskInfo | BaseException] = []
        if self.workers > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(max_workers=min(self.workers, len(jobs))) as ex:
                futures = [ex.submit(_build_job, j) for j in jobs]
                for fut in futures:
                    try:
                        results.append(fut.result())
                    except Exception as exc:
                        results.append(exc)
        else:
            for j in jobs:
                try:
                    results.append(_build_job(j))
                except Exception as exc:
                    results.append(exc)
        for job, res in zip(jobs, results):
            band = job[0]
            if isinstance(res, BaseException):
                log.warning("Band %s: no mask flags (%s)", band, res)
                continue
            self.infos[band] = res
        self.seconds = time.time() - t0
        n_mef = sum(1 for i in self.infos.values() if i.status == "mef")
        total_mb = sum(os.path.getsize(i.bitmask) for i in self.infos.values()) / 1e6
        log.info(
            "Staged %d/%d band bitmasks (%d with count masks) in %s [%s]: %.0f MB, %.1f s",
            len(self.infos), len(jobs), n_mef, self.dir, self.location, total_mb, self.seconds,
        )
        return dict(self.infos)

    def cleanup(self) -> None:
        if self.dir.exists():
            shutil.rmtree(self.dir, ignore_errors=True)
            log.info("Removed mask staging dir %s", self.dir)

    def __enter__(self) -> "MaskStaging":
        return self

    def __exit__(self, *exc) -> bool:
        self.cleanup()
        return False


# ---------------------------------------------------------------------------
# Catalog side
# ---------------------------------------------------------------------------
def rename_mask_columns(cat: Table, bands: Sequence[str]) -> dict[str, str]:
    """Rename SE++ ``isophotal_image_flags_<band>`` / ``..._pixel_count_<band>``
    to ``mask_flags_<band>`` / ``mask_npix_<band>`` in place."""
    renamed: dict[str, str] = {}
    for band in bands:
        for old, new in (
            (f"{SEPP_FLAGS_PREFIX}{band}", mask_flags_column(band)),
            (f"{SEPP_NPIX_PREFIX}{band}", mask_npix_column(band)),
        ):
            if old in cat.colnames and new not in cat.colnames:
                cat.rename_column(old, new)
                renamed[old] = new
    return renamed


def mask_flag_columns(cat: Table) -> list[str]:
    return [c for c in cat.colnames if c.startswith(MASK_FLAGS_PREFIX)]


def any_band_nodata(cat: Table, bands: Sequence[str] | None = None, *, nodata_bit: int = NODATA_BIT) -> np.ndarray:
    """Boolean mask: source has the no-data bit in at least one band.

    Placeholder cells (negative sentinel from the canonical schema) are
    treated as "no information", not as flagged.
    """
    cols = [mask_flags_column(b) for b in bands] if bands is not None else mask_flag_columns(cat)
    out = np.zeros(len(cat), dtype=bool)
    for c in cols:
        if c not in cat.colnames:
            continue
        vals = np.asarray(cat[c])
        if np.ma.isMaskedArray(vals):
            vals = vals.filled(-1)
        vals = np.asarray(vals, dtype=np.float64)
        valid = np.isfinite(vals) & (vals >= 0)
        flagged = np.zeros_like(out)
        flagged[valid] = (vals[valid].astype(np.int64) & nodata_bit) != 0
        out |= flagged
    return out


__all__ = [
    "MASKBIT",
    "NODATA_BIT",
    "BIT_NAMES",
    "PLANE_TO_BIT",
    "COUNT_MASK_SUFFIX",
    "MASK_FLAGS_PREFIX",
    "MASK_NPIX_PREFIX",
    "BandMaskInfo",
    "MaskStaging",
    "build_band_bitmask",
    "find_count_mask",
    "resolve_count_masks",
    "rename_mask_columns",
    "mask_flag_columns",
    "mask_flags_column",
    "mask_npix_column",
    "maskbits_description",
    "any_band_nodata",
    "sweep_stale_staging",
]
