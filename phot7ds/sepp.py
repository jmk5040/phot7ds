"""
SourceExtractor++ (``sourcextractor++``) helpers.

This module knows how to:

1. Generate a minimal SE++ Python config file (:func:`generate_sepp_python_config`)
   that loads the measurement images and adds aperture photometry.
2. Build the ``sourcextractor++`` shell command (:func:`build_sepp_command`).
3. Run the command (:func:`run_sepp`).
4. Post-process the FITS catalog produced by SE++ to split the array-valued
   columns into per-band scalar columns (:func:`split_array_columns_to_per_filter`).
"""
from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from astropy.table import Table

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FlagImage:
    """One SE++ external flag image (``--flag-image-<label>``).

    ``label`` becomes the column suffix (``isophotal_image_flags_<label>``,
    ``isophotal_image_flags_pixel_count_<label>``). ``flag_type`` is one of
    SE++'s ``or``, ``and``, ``min``, ``max``, ``most``; ``or`` is the right
    choice for bitmasks. The image must be an integer FITS image on the
    detection grid; tile-compressed extensions are not readable by SE++.
    """

    label: str
    path: str
    flag_type: str = "or"


_FLAG_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_FLAG_TYPES = ("or", "and", "min", "max", "most")


# The SE++ output properties this pipeline relies on. Order matters: extra
# properties can be appended, but removing one will break downstream code.
DEFAULT_OUTPUT_PROPERTIES: tuple[str, ...] = (
    "SourceIDs",
    "GroupInfo",
    "WorldCentroid",
    "PixelCentroid",
    "ErrorEllipse",
    "PeakValue",
    "SNRRatio",
    "KronRadius",
    "SourceFlags",
    "ExternalFlags",
    "FluxRadius",
    "ShapeParameters",
    "AutoPhotometry",
    "AperturePhotometry",
)


def generate_sepp_python_config(
    config_file: str,
    sciimgs: Sequence[str],
    band_names: Sequence[str],
    saturation_values: Sequence[float],
    gain: float | Sequence[float],
    aperture_photometry: bool = True,
    aper_radius_pix: Sequence[float] | float | None = None,
    fixed_apertures: Sequence[float] | float | None = None,
) -> str:
    """Write a minimal SE++ Python configuration file.

    The file only declares the measurement images and (optionally) the
    aperture columns; *detection* properties are requested via the
    ``--output-properties`` command-line flag.

    Parameters
    ----------
    config_file
        Path to the Python config file to create.
    sciimgs
        Measurement (science) image paths.
    band_names
        Filter names corresponding 1:1 to ``sciimgs``. Duplicates must have
        already been disambiguated via index suffix
        (see :func:`phot7ds.images.extract_band_names_and_saturation`).
    saturation_values
        Saturation level for each science image (see
        :func:`phot7ds.images.extract_band_names_and_saturation`).
    gain
        Gain [e-/ADU] of each science image, 1:1 with ``sciimgs`` (see
        :func:`phot7ds.images.extract_gain_values`). A scalar is applied to
        every image.
    aperture_photometry
        Whether to add aperture photometry blocks.
    aper_radius_pix
        Aperture radius/radii in pixels (required if ``aperture_photometry=True``).
    fixed_apertures
        Aperture sizes (typically arcsec, used for column naming) in matching
        order with ``aper_radius_pix``.

    Returns
    -------
    str
        ``config_file`` (echoed for chaining).
    """
    if isinstance(gain, (list, tuple, np.ndarray)):
        gain_values = [float(g) for g in gain]
    else:
        gain_values = [float(gain)] * len(sciimgs)
    if len(gain_values) != len(sciimgs):
        raise ValueError(
            f"gain ({len(gain_values)}) and sciimgs ({len(sciimgs)}) must match"
        )
    if len(saturation_values) != len(sciimgs):
        raise ValueError(
            f"saturation_values ({len(saturation_values)}) and sciimgs "
            f"({len(sciimgs)}) must match"
        )

    with open(config_file, "w") as f:
        f.write("from sourcextractor.config import *\n\n")
        f.write("measurement_image_paths = [\n")
        for img in sciimgs:
            f.write(f"    '{os.path.abspath(img)}',\n")
        f.write("]\n\n")
        f.write("measurement_image_group = load_fits_images(measurement_image_paths)\n")
        f.write("measurement_group = MeasurementGroup(measurement_image_group)\n\n")
        f.write("# Per-image gain [e-/ADU] and saturation [ADU] of the measurement\n")
        f.write("# images (from their own EGAIN / SATURATE header keys).\n")
        f.write("gain_values = [\n")
        for g in gain_values:
            f.write(f"    {g!r},\n")
        f.write("]\n")
        f.write("saturation_values = [\n")
        for sat in saturation_values:
            f.write(f"    {float(sat)!r},\n")
        f.write("]\n\n")
        f.write("for idx, img in enumerate(measurement_group):\n")
        f.write("    img.gain = gain_values[idx]\n")
        f.write("    img.saturation = saturation_values[idx]\n\n")

        if aperture_photometry:
            if aper_radius_pix is None:
                raise ValueError("aper_radius_pix is required when aperture_photometry=True")
            if fixed_apertures is None:
                raise ValueError("fixed_apertures is required when aperture_photometry=True")
            if not isinstance(aper_radius_pix, (list, tuple, np.ndarray)):
                aper_radius_pix = [aper_radius_pix]
            if not isinstance(fixed_apertures, (list, tuple, np.ndarray)):
                fixed_apertures = [fixed_apertures]
            if len(aper_radius_pix) != len(fixed_apertures):
                raise ValueError(
                    f"aper_radius_pix ({len(aper_radius_pix)}) and "
                    f"fixed_apertures ({len(fixed_apertures)}) must match"
                )

            f.write("aperture_radii = [\n")
            for radius in aper_radius_pix:
                f.write(f"    {float(radius):.2f},\n")
            f.write("]\n\n")
            f.write("band_names = [\n")
            for band in band_names:
                f.write(f"    '{band}',\n")
            f.write("]\n\n")
            f.write("fixed_apertures = [\n")
            for aper in fixed_apertures:
                f.write(f"    {aper},\n")
            f.write("]\n\n")
            f.write("for idx, img in enumerate(measurement_group):\n")
            f.write("    band = band_names[idx] if idx < len(band_names) else f'BAND{idx}'\n")
            f.write("    # SE++ supports one aperture-photometry call per image; multiple\n")
            f.write("    # radii produce an array-valued column which we split in\n")
            f.write("    # post-processing.\n")
            f.write("    aper = add_aperture_photometry(img, aperture_radii)[0]\n")
            f.write("    add_output_column('aper_' + band.lower(), aper)\n")

    log.info("Wrote SE++ Python config: %s", config_file)
    return config_file


def build_sepp_command(
    python_config_file: str,
    sepp_config_file: str,
    detection_image: str,
    detection_gain: float,
    detection_saturate: float,
    catalog_path: str,
    *,
    coverage_mask: str | None = None,
    badpix_mask: str | None = None,
    flag_images: Sequence[FlagImage] | None = None,
    detection_threshold: float = 1.5,
    detection_minimum_area: int = 9,
    auto_kron_min_radius: float = 3.5,
    auto_kron_factor: float = 2.5,
    background_cell_size: int = 256,
    smoothing_box_size: int = 3,
    partition_threshold_count: int = 32,
    partition_minimum_area: int = 9,
    partition_minimum_contrast: float = 1e-5,
    flux_fractions: Sequence[float] = (0.5, 0.9),
    clean_param: int = 8,
    segmentation_filter: str | None = None,
    thread_count: int = 4,
    output_properties: Sequence[str] = DEFAULT_OUTPUT_PROPERTIES,
    log_file: str | None = None,
    log_level: str = "INFO",
) -> str:
    """Build the ``sourcextractor++`` shell command as a single string.

    External flag images come from three sources, all emitted as
    ``--flag-image-<label> <path> --flag-type-<label> <type>``:

    * ``coverage_mask`` -> label ``cover`` (legacy union coverage mask),
    * ``badpix_mask``   -> label ``badpix`` (detection-image bad-pixel mask),
    * ``flag_images``   -> arbitrary :class:`FlagImage` entries, e.g. one
      per-band bitmask per measurement band (see :mod:`phot7ds.masks`).

    Labels must be unique. When no flag image is given at all,
    ``ExternalFlags`` is removed from the output properties, because SE++
    aborts when that property is requested without a flag image.

    Returns the command; pass it to :func:`run_sepp` to execute.
    """
    flags: list[FlagImage] = []
    if coverage_mask:
        flags.append(FlagImage("cover", str(coverage_mask), "or"))
    if badpix_mask:
        flags.append(FlagImage("badpix", str(badpix_mask), "or"))
    flags.extend(flag_images or ())
    seen: set[str] = set()
    for fi in flags:
        if not _FLAG_LABEL_RE.match(fi.label):
            raise ValueError(f"Invalid flag-image label {fi.label!r}")
        if fi.flag_type not in _FLAG_TYPES:
            raise ValueError(f"Invalid flag-type {fi.flag_type!r} for {fi.label!r}; use one of {_FLAG_TYPES}")
        if fi.label in seen:
            raise ValueError(f"Duplicate flag-image label {fi.label!r}")
        seen.add(fi.label)

    prop_list = [p.strip() for p in output_properties if p.strip()]
    if not flags and "ExternalFlags" in prop_list:
        prop_list.remove("ExternalFlags")
    props = ",".join(prop_list)
    fractions = ",".join(str(f) for f in flux_fractions)

    parts: list[str] = [
        "sourcextractor++",
        f"--python-config-file {shlex.quote(python_config_file)}",
        f"--config-file {shlex.quote(sepp_config_file)}",
        f"--detection-image {shlex.quote(detection_image)}",
        f"--detection-image-gain {detection_gain}",
        f"--detection-image-saturation {detection_saturate}",
        f"--detection-threshold {detection_threshold}",
        f"--detection-minimum-area {detection_minimum_area}",
        f"--auto-kron-min-radius {auto_kron_min_radius}",
        f"--auto-kron-factor {auto_kron_factor}",
        f"--background-cell-size {background_cell_size}",
        f"--smoothing-box-size {smoothing_box_size}",
        f"--partition-threshold-count {partition_threshold_count}",
        f"--partition-minimum-area {partition_minimum_area}",
        f"--partition-minimum-contrast {partition_minimum_contrast}",
        f"--flux-fraction {fractions}",
        "--use-cleaning 0",
        f"--cleaning-minimum-area {clean_param}",
        f"--thread-count {thread_count}",
        f"--output-properties {props}",
        f"--output-catalog-filename {shlex.quote(catalog_path)}",
        "--output-catalog-format FITS",
        f"--log-level {log_level}",
    ]
    if segmentation_filter:
        parts.append(f"--segmentation-filter {shlex.quote(segmentation_filter)}")
    for fi in flags:
        parts.append(f"--flag-image-{fi.label} {shlex.quote(fi.path)}")
        parts.append(f"--flag-type-{fi.label} {fi.flag_type}")
    if log_file:
        parts.append(f"--log-file {shlex.quote(log_file)}")
    return " ".join(parts)


def run_sepp(cmd: str, *, check: bool = True) -> int:
    """Run a ``sourcextractor++`` command. Returns the exit code.

    Raises :class:`RuntimeError` on non-zero exit if ``check=True``.
    """
    log.info("Running SE++:\n%s", cmd)
    exit_code = os.system(cmd)
    if check and exit_code != 0:
        raise RuntimeError(f"sourcextractor++ failed with exit code {exit_code}")
    return exit_code


_AUTO_COL_RE = re.compile(r"^(aper\d+)_(.+)_(flux|flux_err|mag|mag_err|flags)$")
_RENAME_RE = re.compile(r"^(aper\d+)_(.+)_(flux|flux_err|mag|mag_err|flags)$")


def split_array_columns_to_per_filter(
    catalog_path: str,
    band_names: Sequence[str],
    flux_fractions: Sequence[float] = (0.5, 0.9),
    overwrite: bool = True,
    fixed_apertures: Sequence[float] | float | None = None,
    array_cols: Sequence[str] = (
        "auto_flux",
        "auto_flux_err",
        "auto_mag",
        "auto_mag_err",
        "auto_flags",
        "flux_radius",
    ),
) -> Table | None:
    """Read a SE++ catalog and split array columns into per-filter columns.

    SE++ writes some properties as array columns shaped
    ``(n_sources, n_bands[, n_extra])``. This function:

    *     Splits ``auto_*`` (2D) and ``flux_radius`` (3D) columns into per-band
      scalar columns, e.g. ``auto_mag_g``, ``flux_rad_50_m400``.
    * Splits the multi-aperture columns ``aper_<band>`` into per-aperture
      columns, e.g. ``aper05_mag_g``, ``aper10_mag_g``. Aperture labels
      are zero-padded to two digits.
    * Renames any remaining ``aperN_<band>_<quantity>`` to
      ``aperN_<quantity>_<band>`` so the band name is always the last token.

    Parameters
    ----------
    catalog_path
        Path to the FITS catalog written by SE++.
    band_names
        Filter names with ``-N`` disambiguation, in the order the
        measurement images were registered.
    flux_fractions
        Flux fractions for ``flux_radius`` splitting (matches the
        ``--flux-fraction`` flag passed to SE++).
    overwrite
        If True, write the updated table back to ``catalog_path``.
    fixed_apertures
        Aperture sizes (used purely for naming; matched to the radii passed
        to SE++ in :func:`generate_sepp_python_config`).
    array_cols
        Array columns produced by SE++ to attempt splitting on.

    Returns
    -------
    Table or None
        The post-processed table, or ``None`` if the catalog file is missing
        or unreadable.
    """
    if not os.path.exists(catalog_path):
        log.error("Catalog not found: %s", catalog_path)
        return None

    try:
        cat = Table.read(catalog_path, format="fits")
    except Exception as exc:
        log.error("Error reading catalog %s: %s", catalog_path, exc)
        return None

    if cat is None or len(cat) == 0:
        log.error("Catalog is empty: %s", catalog_path)
        return None

    log.info("Catalog rows=%d  cols=%d", len(cat), len(cat.colnames))

    n_bands = len(band_names)
    array_cols = [c for c in array_cols if c in cat.colnames]
    for col_name in array_cols:
        col_data = cat[col_name]
        # Special handling for flux_radius shape (n_src, n_bands, n_fracs).
        if col_name == "flux_radius":
            if hasattr(col_data, "shape") and len(col_data.shape) == 3:
                _, n_bands_actual, n_flux_fracs = col_data.shape
                if n_bands_actual == n_bands:
                    for band_idx, band in enumerate(band_names):
                        band_lower = band.lower()
                        for frac_idx, frac_label in enumerate(flux_fractions):
                            if frac_idx < n_flux_fracs:
                                cat[f"flux_rad_{int(frac_label*100)}_{band_lower}"] = (
                                    col_data[:, band_idx, frac_idx]
                                )
                    cat.remove_column(col_name)
                    log.info(
                        "Split %s -> %d filters x %d flux fractions",
                        col_name, n_bands, n_flux_fracs,
                    )
                else:
                    log.warning(
                        "%s has %d bands, expected %d", col_name, n_bands_actual, n_bands
                    )
            else:
                log.warning(
                    "%s has unexpected shape, skipping", col_name
                )
            continue

        if not hasattr(col_data, "shape") or len(col_data.shape) <= 1:
            continue

        if col_name.startswith("auto_"):
            base_name = "auto"
            quantity = col_name[5:].lower()
        else:
            base_name = col_name.lower()
            quantity = ""

        arr2d: np.ndarray | None = None
        if len(col_data.shape) == 2 and col_data.shape[1] == n_bands:
            arr2d = np.asarray(col_data)
        elif (
            len(col_data.shape) == 3
            and col_data.shape[1] == n_bands
            and col_data.shape[2] == 1
        ):
            arr2d = np.asarray(col_data).squeeze(axis=2)

        if arr2d is None:
            log.warning("Could not split %s (shape %s)", col_name, col_data.shape)
            continue

        for idx, band in enumerate(band_names):
            band_lower = band.lower()
            new_col = f"{base_name}_{quantity}_{band_lower}" if quantity else f"{base_name}_{band_lower}"
            cat[new_col] = arr2d[:, idx]
        cat.remove_column(col_name)
        log.info("Split %s into %d per-filter columns", col_name, n_bands)

    # Split multi-aperture columns generated as add_output_column('aper_<band>', aper).
    if fixed_apertures is not None:
        if not isinstance(fixed_apertures, (list, tuple, np.ndarray)):
            fixed_apertures = [fixed_apertures]
        fixed_apertures = list(fixed_apertures)

        def _aper_label(size: float | int | str) -> str:
            """Return a zero-padded aperture label (``5`` -> ``"05"``)."""
            if isinstance(size, str):
                token = size.strip().lower().lstrip("aper")
                if token.isdigit():
                    return f"{int(token):02d}"
                return token or str(size)
            try:
                val = float(size)
            except (TypeError, ValueError):
                return str(size)
            if val.is_integer() or abs(val - round(val)) < 1e-6:
                return f"{int(round(val)):02d}"
            return f"{val:g}"

        to_split: list[tuple[str, str, str]] = []
        for col in cat.colnames:
            m = re.match(r"^aper_(.+)_(flux|flux_err|mag|mag_err|flags)$", col.lower())
            if m:
                band, quantity = m.groups()
                to_split.append((col, band, quantity))

        for col_name, band, quantity in to_split:
            col_data = np.asarray(cat[col_name])
            split_data: np.ndarray | None = None
            aper_labels: list[str] = []
            if len(fixed_apertures) == 1:
                size = fixed_apertures[0]
                label = _aper_label(size)
                if col_data.ndim == 1:
                    split_data = col_data[:, np.newaxis]
                    aper_labels = [label]
                elif col_data.ndim == 2 and col_data.shape[1] == 1:
                    split_data = col_data
                    aper_labels = [label]

            if split_data is None and col_data.ndim == 2 and col_data.shape[1] == len(
                fixed_apertures
            ):
                split_data = col_data
                aper_labels = [_aper_label(s) for s in fixed_apertures]
            elif split_data is None and col_data.ndim == 3:
                if col_data.shape[1] == 1 and col_data.shape[2] == len(fixed_apertures):
                    split_data = col_data[:, 0, :]
                    aper_labels = [_aper_label(s) for s in fixed_apertures]
                elif col_data.shape[1] == len(fixed_apertures) and col_data.shape[2] == 1:
                    split_data = col_data[:, :, 0]
                    aper_labels = [_aper_label(s) for s in fixed_apertures]

            if split_data is None:
                log.warning("Could not split %s with shape %s", col_name, col_data.shape)
                continue
            for aper_idx, label in enumerate(aper_labels):
                cat[f"aper{label}_{quantity}_{band}"] = split_data[:, aper_idx]
            cat.remove_column(col_name)
            log.info(
                "Split %s into %d aperture columns (labels=%s)",
                col_name, len(aper_labels), ",".join(aper_labels),
            )

    # Final pass: ensure aperN_<band>_<quantity> -> aperN_<quantity>_<band>.
    for col in list(cat.colnames):
        m = _RENAME_RE.match(col.lower())
        if not m:
            continue
        aper_size, band, quantity = m.groups()
        new_name = f"{aper_size}_{quantity}_{band}"
        if new_name != col and new_name not in cat.colnames:
            cat.rename_column(col, new_name)

    if overwrite:
        cat.write(catalog_path, format="fits", overwrite=True)
        log.info("Wrote split catalog: %s", catalog_path)
    return cat


__all__ = [
    "DEFAULT_OUTPUT_PROPERTIES",
    "FlagImage",
    "generate_sepp_python_config",
    "build_sepp_command",
    "run_sepp",
    "split_array_columns_to_per_filter",
]
