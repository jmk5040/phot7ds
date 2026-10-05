"""
Image organisation and mask generation.

* :func:`identify_image_filter` names an image's filter (``FILTER`` card,
  else the filename); :func:`select_images_by_filter` keeps the images whose
  filter is in a given list (the filter registry), optionally one per filter,
  and reports the rest; :func:`organize_images_by_filter` groups paths by
  filter in that list's order.
* :func:`build_coverage_mask` builds a coverage mask that flags pixels which
  are zero in the detection image or in any science image.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from astropy.io import fits

log = logging.getLogger(__name__)


#: Reasons recorded for an image left out of the measurement set.
DROP_UNKNOWN_FILTER = "unknown_filter"
DROP_NOT_IN_REGISTRY = "not_in_registry"
DROP_DUPLICATE = "duplicate"

_FILTER_LIKE_RE = re.compile(r"^(?:[a-z]|m\d{3,4}w?)$", re.IGNORECASE)


def _match_name(label: str, names: Sequence[str]) -> str | None:
    lowered = label.strip().lower()
    for name in names:
        if name.lower() == lowered:
            return name
    return None


@dataclass(frozen=True)
class ImageFilter:
    """Filter identification of one image (see :func:`identify_image_filter`).

    ``band`` is the matched name from ``names`` (``None`` if unmatched);
    ``label`` the raw filter label found; ``source`` where it came from
    (``'header'``, ``'filename'`` or ``None``).
    """

    path: str
    band: str | None
    label: str | None
    source: str | None

    @property
    def drop_reason(self) -> str | None:
        if self.band is not None:
            return None
        return DROP_NOT_IN_REGISTRY if self.label else DROP_UNKNOWN_FILTER


def identify_image_filter(
    img_path: str,
    names: Sequence[str],
    filter_source: Literal["header", "filename"] = "header",
) -> ImageFilter:
    """Identify the filter of ``img_path`` among ``names``.

    With ``filter_source='header'`` the ``FILTER`` card decides whenever it
    is present (a value not in ``names`` is reported, not overridden by the
    filename). Without a readable ``FILTER`` card, or with
    ``filter_source='filename'``, the basename is split on ``_``/``.``/``-``
    and the first token equal (case-insensitively) to one of ``names`` wins;
    a filter-like token (``m329w``, ``y``) that is not in ``names`` is
    reported as the label.
    """
    names = list(names)
    if filter_source == "header":
        raw = None
        try:
            raw = fits.getheader(img_path).get("FILTER")
        except Exception as exc:
            log.debug("FITS header read failed for %s: %s", img_path, exc)
        if raw is not None and str(raw).strip():
            label = str(raw).strip()
            band = _match_name(label, names)
            if band is None:
                band = _match_name(re.split(r"[-_]", label)[0], names)
            return ImageFilter(img_path, band, label, "header")

    tokens = [t for t in re.split(r"[_.\-]", os.path.basename(img_path)) if t]
    for tok in tokens:
        band = _match_name(tok, names)
        if band is not None:
            return ImageFilter(img_path, band, tok, "filename")
    for tok in tokens[1:]:
        if _FILTER_LIKE_RE.match(tok):
            return ImageFilter(img_path, None, tok, "filename")
    return ImageFilter(img_path, None, None, None)


def select_images_by_filter(
    image_list: Sequence[str],
    names: Sequence[str],
    *,
    deduplicate: bool = True,
    filter_source: Literal["header", "filename"] = "header",
) -> tuple[list[ImageFilter], list[dict[str, str | None]]]:
    """Keep the images whose filter is in ``names``, ordered as ``names``.

    Images are sorted by basename within a filter; with ``deduplicate`` only
    the first one per filter is kept. Returns ``(kept, dropped)``, where each
    dropped entry is ``{"image", "filter", "reason"}`` with ``reason`` one of
    ``unknown_filter`` (no filter found), ``not_in_registry`` (a filter not in
    ``names``) or ``duplicate`` (``filter`` then names the kept image's band
    and ``kept`` the kept image).
    """
    order = {name: i for i, name in enumerate(names)}
    ids = [identify_image_filter(p, names, filter_source) for p in image_list]
    dropped: list[dict[str, str | None]] = [
        {"image": str(f.path), "filter": f.label, "reason": f.drop_reason}
        for f in ids if f.band is None
    ]
    known = sorted(
        (f for f in ids if f.band is not None),
        key=lambda f: (order[f.band], os.path.basename(f.path), f.path),
    )
    kept: list[ImageFilter] = []
    first: dict[str, ImageFilter] = {}
    for f in known:
        if deduplicate and f.band in first:
            dropped.append({
                "image": str(f.path), "filter": f.band, "reason": DROP_DUPLICATE,
                "kept": str(first[f.band].path),
            })
            continue
        first.setdefault(f.band, f)
        kept.append(f)
    return kept, dropped


def organize_images_by_filter(
    image_list: Sequence[str],
    bands_dict: dict[str, float],
    filter_source: Literal["header", "filename"] = "header",
    output_form: Literal["dict", "list"] = "dict",
    keep_duplicates: bool = True,
):
    """Group science image paths by filter according to ``bands_dict`` order.

    Parameters
    ----------
    image_list
        Flat list of image file paths.
    bands_dict
        Mapping ``{filter_name: central_wavelength}``. The *order* of keys
        in this dict defines the output order (see :func:`get_filter_definitions`).
    filter_source
        ``'header'`` reads the ``FILTER`` FITS keyword (falling back to the
        filename when absent); ``'filename'`` uses the basename only. See
        :func:`identify_image_filter`.
    output_form
        - ``'dict'``: ``{filter: path | [paths] | None}``. Missing filters
          carry ``None``. Duplicates collapse based on ``keep_duplicates``.
        - ``'list'``: ordered list, missing filters omitted.
    keep_duplicates
        For ``output_form='dict'`` only. If False, only the first sorted path
        per filter is kept (single string instead of a list).

    Images whose filter is not in ``bands_dict`` are left out (logged at
    INFO); use :func:`select_images_by_filter` to get them back with reasons.

    Returns
    -------
    dict or list
        See ``output_form``.
    """
    filter_order = list(bands_dict.keys())
    kept, dropped = select_images_by_filter(
        image_list, filter_order, deduplicate=False, filter_source=filter_source,
    )
    for d in dropped:
        log.info("Image without a known filter (%s): %s", d["filter"] or "none", d["image"])
    image_filter_map: dict[str, list[str]] = {}
    for f in kept:
        image_filter_map.setdefault(f.band, []).append(f.path)

    if output_form == "dict":
        result: dict[str, str | list[str] | None] = {}
        for filter_name in filter_order:
            paths = image_filter_map.get(filter_name)
            if not paths:
                result[filter_name] = None
                continue
            # Sort by basename (ignore directory) so dedup picks the
            # alphabetically-first file name.
            paths.sort(key=os.path.basename)
            if len(paths) > 1:
                result[filter_name] = paths if keep_duplicates else paths[0]
            else:
                result[filter_name] = paths[0]
        return result

    if output_form == "list":
        ordered: list[str] = []
        for filter_name in filter_order:
            paths = image_filter_map.get(filter_name)
            if not paths:
                continue
            paths.sort(key=os.path.basename)
            ordered.extend(paths)
        return ordered

    raise ValueError(f"output_form must be 'dict' or 'list', got {output_form!r}")


#: Header keywords tried in order for a measurement image's saturation level
#: (7DS coadds carry ``SATURATE``; ``SATLV`` is the single-frame 7DT key).
SATURATION_KEYWORDS: tuple[str, ...] = ("SATURATE", "SATLV")
#: Header keywords tried in order for a measurement image's gain [e-/ADU].
#: 7DS coadds carry the effective gain of the stack in ``EGAIN`` and no
#: ``GAIN``.
GAIN_KEYWORDS: tuple[str, ...] = ("EGAIN", "GAIN")


def _first_header_value(
    hdr: fits.Header, keywords: Sequence[str]
) -> tuple[float | None, str | None]:
    for key in keywords:
        value = hdr.get(key)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value, key
    return None, None


def extract_band_names_and_saturation(
    sciimgs: Sequence[str],
    default_saturation: float = 10000,
    saturation_keywords: Sequence[str] = SATURATION_KEYWORDS,
    bands: Sequence[str] | None = None,
) -> tuple[list[str], list[float]]:
    """Extract per-image filter names and saturation values from FITS headers.

    ``bands`` (1:1 with ``sciimgs``, e.g. the registry names found by
    :func:`select_images_by_filter`) replaces the ``FILTER`` card as the band
    name. Duplicate filter names get index suffixes (``-1``, ``-2``, ...) so
    SE++ column names stay unique.

    Returns
    -------
    band_names
        Filter names, possibly with ``-N`` disambiguation suffixes.
    saturation_values
        Saturation level for each image: the first of ``saturation_keywords``
        (``SATURATE``, then ``SATLV``) present in the header, else
        ``default_saturation`` (with a warning).
    """
    if bands is not None and len(bands) != len(sciimgs):
        raise ValueError(f"bands has {len(bands)} entries for {len(sciimgs)} images")
    band_names_raw: list[str] = []
    saturation_values: list[float] = []
    for j, img in enumerate(sciimgs):
        try:
            hdr = fits.getheader(img)
            band = str(hdr.get("FILTER", "UNKNOWN")).replace("-", "_")
            saturation, _ = _first_header_value(hdr, saturation_keywords)
        except Exception:
            band = "UNKNOWN"
            saturation = None
        if bands is not None:
            band = bands[j]
        if saturation is None:
            log.warning(
                "No saturation keyword (%s) in %s; using %g",
                "/".join(saturation_keywords), os.path.basename(img), default_saturation,
            )
            saturation = float(default_saturation)
        band_names_raw.append(band)
        saturation_values.append(saturation)

    counts: dict[str, int] = {}
    for b in band_names_raw:
        counts[b] = counts.get(b, 0) + 1

    band_names: list[str] = []
    occurrences: dict[str, int] = {}
    for b in band_names_raw:
        if counts[b] > 1:
            occurrences[b] = occurrences.get(b, -1) + 1
            if occurrences[b] > 0:
                b = f"{b}-{occurrences[b]}"
        band_names.append(b)
    return band_names, saturation_values


def extract_gain_values(
    sciimgs: Sequence[str],
    default_gain: float = 0.0,
    gain_keywords: Sequence[str] = GAIN_KEYWORDS,
) -> list[float]:
    """Per-image gain [e-/ADU] for the measurement images.

    Reads the first of ``gain_keywords`` (``EGAIN``, then ``GAIN``) present in
    each header. A missing value falls back to ``default_gain`` with a
    warning; the default ``0`` is SE++'s own "no gain" value (the Poisson
    term of the flux error is dropped).
    """
    gains: list[float] = []
    for img in sciimgs:
        try:
            gain, _ = _first_header_value(fits.getheader(img), gain_keywords)
        except Exception:
            gain = None
        if gain is None:
            log.warning(
                "No gain keyword (%s) in %s; using %g",
                "/".join(gain_keywords), os.path.basename(img), default_gain,
            )
            gain = float(default_gain)
        gains.append(gain)
    return gains


def read_detection_gain(
    detection_image: str,
    default_gain: float = 0.0,
    gain_keywords: Sequence[str] = GAIN_KEYWORDS,
) -> tuple[float, str | None]:
    """Gain [e-/ADU] SE++ should use for the detection image.

    Same keyword order as the measurement images: a 7DS white stack carries
    the effective gain in ``EGAIN``; a DELVE mosaic carries only SWarp's
    ``GAIN`` and falls through to it. A missing value falls back to
    ``default_gain`` (``0`` = SE++'s "no Poisson term") with a warning.

    Returns ``(gain, keyword)``; ``keyword`` is ``None`` for the fallback.
    """
    try:
        gain, key = _first_header_value(fits.getheader(detection_image), gain_keywords)
    except Exception:
        gain, key = None, None
    if gain is None:
        log.warning(
            "No gain keyword (%s) in detection image %s; using %g",
            "/".join(gain_keywords), os.path.basename(detection_image), default_gain,
        )
        return float(default_gain), None
    return gain, key


def _primary_data_shape(image_path: str) -> tuple[int, ...]:
    """Return the primary-HDU pixel shape ``(ny, nx)`` without loading data."""
    with fits.open(image_path, memmap=True) as hdul:
        return tuple(hdul[0].data.shape)


def build_coverage_mask(
    detection_image: str,
    science_images: Sequence[str],
    output_path: str,
    overwrite: bool = False,
    max_masked_fraction: float = 0.5,
) -> tuple[str | None, float | None]:
    """Build a coverage mask flagging zero-valued pixels.

    The output mask is the union (sum, clipped to ``uint8``) of:

    * zero pixels in the detection image,
    * zero pixels in each science image.

    Because the mask is a pixel-wise union, it is only meaningful when the
    detection image and every science image share the same array shape (the
    7DS standard coadd grid). When any science image has a different shape
    -- e.g. unstandardised single-frame images -- the mask is **skipped**
    rather than raising: the function logs a warning and returns
    ``(None, None)``. Photometry then proceeds without a coverage flag image
    (SE++ still aligns measurement images by WCS).

    Parameters
    ----------
    detection_image
        Path to detection FITS image.
    science_images
        Iterable of science FITS image paths.
    output_path
        Path to write the mask FITS file.
    overwrite
        If False and ``output_path`` exists, reuse it without rewriting.
    max_masked_fraction
        If the masked-pixel ratio exceeds this, raise :class:`ValueError`.

    Returns
    -------
    output_path
        Path to the mask FITS file, or ``None`` when the mask was skipped
        because of a shape mismatch.
    masked_ratio
        Fraction of pixels that ended up flagged, or ``None`` when skipped.
    """
    if not overwrite and os.path.exists(output_path):
        with fits.open(output_path) as hdul:
            ratio = float(hdul[0].header.get("MSKRATIO", np.nan))
        log.info("Coverage mask exists, reusing: %s", output_path)
        return output_path, ratio

    det_shape = _primary_data_shape(detection_image)

    # Shape-compatibility guard: skip (don't fail) when images are not on the
    # common detection grid, so non-standard inputs still get a catalog.
    mismatched: list[tuple[str, tuple[int, ...]]] = []
    for sciimg in science_images:
        sci_shape = _primary_data_shape(sciimg)
        if sci_shape != det_shape:
            mismatched.append((sciimg, sci_shape))
    if mismatched:
        sample = mismatched[0]
        log.warning(
            "Skipping coverage mask: %d/%d science image(s) do not match the "
            "detection grid %s (e.g. %s has shape %s). Photometry will run "
            "without a coverage flag image.",
            len(mismatched), len(list(science_images)), det_shape,
            os.path.basename(sample[0]), sample[1],
        )
        return None, None

    def _nodata(data: np.ndarray, out: np.ndarray) -> np.ndarray:
        # Zero *or* non-finite: NaN/inf pixels are equally "no data" and
        # would otherwise poison the photometry without raising a flag.
        np.equal(data, 0, out=out)
        out |= ~np.isfinite(data)
        return out

    with fits.open(detection_image, memmap=True) as hdul_det:
        det_data = hdul_det[0].data
        maskdata = np.zeros(det_data.shape, dtype=np.uint16)
        zero_mask = np.zeros(det_data.shape, dtype=bool)
        _nodata(det_data, zero_mask)
        maskdata[zero_mask] = 1

    for sciimg in science_images:
        with fits.open(sciimg, memmap=True) as hdul_sci:
            _nodata(hdul_sci[0].data, zero_mask)
        maskdata += zero_mask

    masked_ratio = float(np.count_nonzero(maskdata) / maskdata.size)
    hdr = fits.getheader(detection_image)
    hdr["MSKRATIO"] = (
        round(masked_ratio, 3),
        "Ratio of pixels masked as 1 (covered by all images)",
    )
    hdr["DETIMG"] = (
        os.path.basename(detection_image), "Detection image filename"
    )
    for j, sciimg in enumerate(science_images):
        hdr[f"SCIMG{j:03d}"] = (
            os.path.basename(sciimg), f"Science image #{j:03d} basename"
        )

    fits.PrimaryHDU(
        data=np.clip(maskdata, 0, 255).astype(np.uint8), header=hdr
    ).writeto(output_path, overwrite=True)

    if masked_ratio > max_masked_fraction:
        raise ValueError(
            f"Coverage mask flagged {masked_ratio*100:.1f}% of pixels, exceeding "
            f"max_masked_fraction={max_masked_fraction:.2f}. Check the detection image."
        )

    log.info("Coverage mask: %s (%.1f%% pixels masked)", output_path, masked_ratio * 100)
    return output_path, masked_ratio


__all__ = [
    "DROP_DUPLICATE",
    "DROP_NOT_IN_REGISTRY",
    "DROP_UNKNOWN_FILTER",
    "ImageFilter",
    "identify_image_filter",
    "select_images_by_filter",
    "organize_images_by_filter",
    "extract_band_names_and_saturation",
    "extract_gain_values",
    "read_detection_gain",
    "GAIN_KEYWORDS",
    "SATURATION_KEYWORDS",
    "build_coverage_mask",
]
