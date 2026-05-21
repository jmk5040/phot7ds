"""
Image organisation and mask generation.

* :func:`organize_images_by_filter` groups a flat list of science image paths
  by filter name (read from the FITS header or the filename) using a band
  dictionary as the master order. Optionally deduplicates within a band.
* :func:`build_coverage_mask` builds a coverage mask that flags pixels which
  are zero in the detection image or in any science image.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Literal, Sequence

import numpy as np
from astropy.io import fits

log = logging.getLogger(__name__)


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
        ``'header'`` reads the ``FILTER`` FITS keyword; ``'filename'`` infers
        from the path.
    output_form
        - ``'dict'``: ``{filter: path | [paths] | None}``. Missing filters
          carry ``None``. Duplicates collapse based on ``keep_duplicates``.
        - ``'list'``: ordered list, missing filters omitted.
    keep_duplicates
        For ``output_form='dict'`` only. If False, only the first sorted path
        per filter is kept (single string instead of a list).

    Returns
    -------
    dict or list
        See ``output_form``.
    """
    filter_order = list(bands_dict.keys())
    image_filter_map: dict[str, list[str]] = {}

    for img_path in image_list:
        filter_name: str | None = None

        if filter_source == "header":
            try:
                hdr = fits.getheader(img_path)
                raw = hdr.get("FILTER", None)
                if raw:
                    filter_name = raw.replace("-", "_").strip()
            except Exception as exc:
                log.debug("FITS header read failed for %s: %s", img_path, exc)

        if filter_name is None or filter_source == "filename":
            for band in filter_order:
                pattern = r"(?:^|[/_])" + re.escape(band) + r"(?:[/_]|$)"
                if re.search(pattern, img_path, re.IGNORECASE):
                    filter_name = band
                    break
            if filter_name is None:
                match = re.search(r"[/_]([gm]\d+)[/_]", img_path, re.IGNORECASE)
                if match:
                    extracted = match.group(1).lower()
                    if extracted.startswith("m"):
                        num = re.search(r"\d+", extracted)
                        if num:
                            filter_name = f"m{num.group()}"
                    elif extracted in ("g", "r", "i"):
                        filter_name = extracted

        if filter_name is None:
            filter_name = "UNKNOWN"
        filter_base = filter_name.split("_")[0]
        if filter_base in filter_order:
            filter_name = filter_base
        elif filter_name not in filter_order:
            filter_name = "UNKNOWN"

        image_filter_map.setdefault(filter_name, []).append(img_path)

    if output_form == "dict":
        result: dict[str, str | list[str] | None] = {}
        for filter_name in filter_order:
            paths = image_filter_map.get(filter_name)
            if not paths:
                result[filter_name] = None
                continue
            paths.sort()
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
            paths.sort()
            ordered.extend(paths)
        return ordered

    raise ValueError(f"output_form must be 'dict' or 'list', got {output_form!r}")


def extract_band_names_and_saturation(
    sciimgs: Sequence[str], default_saturation: float = 10000
) -> tuple[list[str], list[float]]:
    """Extract per-image filter names and saturation values from FITS headers.

    Duplicate filter names get index suffixes (``-1``, ``-2``, ...) so SE++
    column names stay unique.

    Returns
    -------
    band_names
        Filter names, possibly with ``-N`` disambiguation suffixes.
    saturation_values
        Saturation level for each image (read from ``SATLV``; falls back to
        ``default_saturation`` if missing).
    """
    band_names_raw: list[str] = []
    saturation_values: list[float] = []
    for img in sciimgs:
        try:
            hdr = fits.getheader(img)
            band = str(hdr.get("FILTER", "UNKNOWN")).replace("-", "_")
            saturation = float(hdr.get("SATLV", default_saturation))
        except Exception:
            band = "UNKNOWN"
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


def build_coverage_mask(
    detection_image: str,
    science_images: Sequence[str],
    output_path: str,
    overwrite: bool = False,
    max_masked_fraction: float = 0.5,
) -> tuple[str, float]:
    """Build a coverage mask flagging zero-valued pixels.

    The output mask is the union (sum, clipped to ``uint8``) of:

    * zero pixels in the detection image,
    * zero pixels in each science image.

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
        Path to the mask FITS file.
    masked_ratio
        Fraction of pixels that ended up flagged.
    """
    if not overwrite and os.path.exists(output_path):
        with fits.open(output_path) as hdul:
            ratio = float(hdul[0].header.get("MSKRATIO", np.nan))
        log.info("Coverage mask exists, reusing: %s", output_path)
        return output_path, ratio

    with fits.open(detection_image, memmap=True) as hdul_det:
        det_data = hdul_det[0].data
        maskdata = np.zeros(det_data.shape, dtype=np.uint16)
        zero_mask = np.zeros(det_data.shape, dtype=bool)
        np.equal(det_data, 0, out=zero_mask)
        maskdata[zero_mask] = 1

    for sciimg in science_images:
        with fits.open(sciimg, memmap=True) as hdul_sci:
            np.equal(hdul_sci[0].data, 0, out=zero_mask)
        maskdata += zero_mask

    masked_ratio = float(np.count_nonzero(maskdata) / maskdata.size)
    hdr = fits.getheader(detection_image)
    hdr["MSKRATIO"] = round(masked_ratio, 3)
    hdr["DETIMG"] = os.path.basename(detection_image)
    for j, sciimg in enumerate(science_images):
        hdr[f"SCIMG{j:03d}"] = os.path.basename(sciimg)

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
    "organize_images_by_filter",
    "extract_band_names_and_saturation",
    "build_coverage_mask",
]
