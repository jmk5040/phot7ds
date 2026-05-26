"""
Single-image-set photometry pipeline.

The public entry point is :func:`run_photometry`. It takes a list of science
image paths, a detection image, a SourceExtractor++ config and a reference
catalog, and writes a single zero-point-calibrated FITS catalog plus a run
log to ``output_dir``.

All tuning knobs are keyword-only arguments with sensible defaults. For
convenience you may bundle settings into a :class:`PhotometryConfig` and
pass it as ``config=``; any kwarg explicitly passed to
:func:`run_photometry` always overrides the equivalent field on ``config``.
"""
from __future__ import annotations

import dataclasses
import getpass
import json
import logging
import platform
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from astropy.io import fits

from ._logging import configure_logging
from .calibration import calibrate_zeropoints, load_gaiaxp_reference
from .config import PhotometryConfig
from .depth import (
    depth_results_to_meta,
    estimate_depths,
    format_depth_table,
    zeropoints_to_meta,
)
from .filters import DEFAULT_BANDS, get_filter_definitions
from .images import (
    build_coverage_mask,
    extract_band_names_and_saturation,
    organize_images_by_filter,
)
from .presets import PRESET_TUNING_FIELDS, resolve_preset
from .schema import (
    build_canonical_schema,
    standardize_catalog as apply_standard_catalog,
    strip_nonfits_units,
)
from .sepp import (
    build_sepp_command,
    generate_sepp_python_config,
    run_sepp,
    split_array_columns_to_per_filter,
)

log = logging.getLogger(__name__)


@dataclass
class PhotometryResult:
    """Outputs of a single :func:`run_photometry` call.

    Attributes
    ----------
    catalog_path
        Path to the final zero-point-calibrated FITS catalog. The basename
        is whatever the user supplied (only ``.fits`` is enforced); for
        example ``test_zp.fits`` stays ``test_zp.fits``. The raw SE++
        output is written alongside it as ``{run_name}_raw.fits``.
    manifest_path
        Path to a JSON file recording the inputs used.
    log_file
        Path to the run log (combined Python + SE++ output).
    n_sources
        Number of sources in the final catalog.
    """

    catalog_path: str
    manifest_path: str
    log_file: str
    n_sources: int


# Fields that are pipeline kwargs but NOT on PhotometryConfig (per-run only).
_RUN_ONLY_KWARGS = {
    "science_images",
    "detection_image",
    "reference_catalog",
    "output_dir",
    "catalog_path",
    "catalog_name",
    "coverage_mask",
    "badpix_mask",
    "tile_info",
    "run_name",
    "overwrite",
    "deduplicate_by_filter",
    "max_measurement_images",
    "standardize_catalog",
    "config",
}


def _normalize_catalog_basename(name: str) -> str:
    """Normalise a catalog *filename* (no directories).

    Behaviour
    ---------
    * Path components in ``name`` are stripped (only the leaf is kept).
    * If ``name`` ends in ``.fits`` it is used verbatim. The caller's
      choice of suffix (``test_zp.fits``, ``my_run.fits``, ...) is
      respected and **not** rewritten to ``_phot.zp.fits``.
    * Otherwise ``name`` is treated as a stem and ``.fits`` is appended.

    Examples
    --------
    ``T01_20260512_DELVE`` -> ``T01_20260512_DELVE.fits``
    ``test_zp.fits``       -> ``test_zp.fits``
    ``test_zp``            -> ``test_zp.fits``
    """
    leaf = Path(name).name
    if leaf.lower().endswith(".fits"):
        return leaf
    return f"{leaf}.fits"


def _stem_from_catalog_basename(basename: str) -> str:
    """Derive ``run_name`` stem from a normalised catalog basename.

    Strips the ``.fits`` extension. Any earlier ``_phot.zp`` / ``.zp``
    suffixes are kept as part of the stem so intermediate files inherit
    the user's filename intent.
    """
    leaf = Path(basename).name
    if leaf.lower().endswith(".fits"):
        return leaf[:-5]
    return leaf


def _normalize_catalog_path(catalog_path: str | Path) -> Path:
    """Return the final calibrated-catalog path (full path).

    Accepts a path that already ends in ``.fits``, or a stem (``.fits``
    is then appended).
    """
    p = Path(catalog_path)
    if p.name.lower().endswith(".fits"):
        return p
    return p.with_name(f"{p.name}.fits")


def _resolve_output_paths(
    *,
    output_dir: str | Path | None,
    catalog_path: str | Path | None,
    catalog_name: str | None,
    run_name: str | None,
    detection_image: str,
    detection_label: str,
) -> tuple[Path, Path, Path, Path, Path, str]:
    """Return work_dir, raw_catalog, zp_catalog, log, manifest, run_name.

    The raw SE++ output is written as ``{run_name}_raw.fits`` next to the
    final calibrated catalog so the two are easy to tell apart.
    """
    if catalog_path is not None and catalog_name is not None:
        raise ValueError("Pass only one of catalog_path or catalog_name, not both.")

    if catalog_name is not None:
        if output_dir is None:
            raise ValueError("output_dir is required when catalog_name is set.")
        work_dir = Path(output_dir)
        basename = _normalize_catalog_basename(catalog_name)
        zp_catalog_path = work_dir / basename
        run_name = run_name or _stem_from_catalog_basename(basename)
    elif catalog_path is not None:
        zp_catalog_path = _normalize_catalog_path(catalog_path)
        zp_catalog_path.parent.mkdir(parents=True, exist_ok=True)
        basename = zp_catalog_path.name
        run_name = run_name or _stem_from_catalog_basename(basename)
        work_dir = Path(output_dir) if output_dir is not None else zp_catalog_path.parent
    elif output_dir is not None:
        work_dir = Path(output_dir)
        det_base = Path(detection_image).stem
        run_name = run_name or f"{det_base}_{detection_label}"
        zp_catalog_path = work_dir / f"{run_name}.fits"
    else:
        raise ValueError(
            "At least one of output_dir, catalog_name, or catalog_path is required."
        )

    work_dir.mkdir(parents=True, exist_ok=True)
    raw_catalog_path = work_dir / f"{run_name}_raw.fits"
    log_file = work_dir / f"{run_name}.log"
    manifest_path = work_dir / f"{run_name}_manifest.json"
    return work_dir, raw_catalog_path, zp_catalog_path, log_file, manifest_path, run_name


def _apply_detection_preset(
    cfg: PhotometryConfig, detection_label: str | None
) -> PhotometryConfig:
    """Fill any preset-controlled field that is still ``None`` on ``cfg``.

    The preset chosen depends on ``detection_label`` (the configured
    label takes precedence over the explicit argument when both exist).
    Returns the (possibly updated) config and emits an info-level log
    line describing the resolved values.
    """
    label = cfg.detection_label or detection_label
    preset = resolve_preset(label)
    changes: dict[str, Any] = {}
    for field_name in PRESET_TUNING_FIELDS:
        if getattr(cfg, field_name) is None and field_name in preset:
            changes[field_name] = preset[field_name]
    if changes:
        log.info(
            "Detection preset '%s' applied: %s",
            (label or "default"),
            ", ".join(f"{k}={v}" for k, v in changes.items()),
        )
        cfg = cfg.replace(**changes)
    return cfg


def _flatten_meta(prefix: str, value: Any, out: dict[str, Any]) -> None:
    """Flatten a (nested) mapping into ``KEY = scalar`` pairs.

    Used internally by :func:`_annotate_catalog_meta` to expand parts of
    the manifest into FITS-safe header cards.
    """
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten_meta(f"{prefix}_{k}".upper(), v, out)
        return
    if isinstance(value, (list, tuple)):
        out[prefix.upper()] = ",".join(str(v) for v in value)
        return
    if value is None:
        return
    out[prefix.upper()] = value


def _annotate_catalog_meta(
    meta: dict[str, Any],
    *,
    detection_image: str,
    coverage_mask: str | None,
    badpix_mask: str | None,
    reference_catalog: str,
    science_images: Sequence[str],
    detection_label: str,
    mask_ratio: float | None,
    cfg: PhotometryConfig,
    run_name: str,
) -> None:
    """Inject high-value run-time metadata into ``meta`` (in place).

    These cards land on the primary header of the final FITS catalog.
    They are kept short (<= 8 chars, no HIERARCH) so that legacy readers
    keep working. Values come from the same data we already write to the
    manifest JSON.
    """
    # Lazy version lookup avoids a circular import.
    try:
        from . import __version__ as version
    except Exception:
        version = "unknown"

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    meta["PHOTVER"] = (str(version), "phot7ds package version")
    meta["PHOTRUN"] = (str(run_name), "phot7ds run_name (stem of this catalog)")
    meta["PHOTDATE"] = (now, "Catalog write timestamp [UTC]")
    try:
        meta["PHOTHOST"] = (socket.gethostname(), "Hostname that wrote this catalog")
    except Exception:
        pass
    try:
        meta["PHOTUSR"] = (getpass.getuser(), "Username that wrote this catalog")
    except Exception:
        pass
    meta["PHOTPY"] = (platform.python_version(), "Python interpreter version")

    meta["DETLABEL"] = (str(detection_label), "Detection-image type label")
    meta["DETIMG"] = (
        str(Path(detection_image).name), "Detection image basename"
    )
    if coverage_mask:
        meta["COVMASK"] = (
            str(Path(coverage_mask).name), "Coverage mask basename"
        )
    if badpix_mask:
        meta["BADPMASK"] = (
            str(Path(badpix_mask).name), "Bad-pixel mask basename"
        )
    meta["REFCAT"] = (
        str(Path(reference_catalog).name), "Reference catalog basename"
    )
    meta["NSCIIMG"] = (
        int(len(science_images)), "Number of measurement (science) images"
    )
    for j, img in enumerate(science_images):
        meta[f"SCIMG{j:03d}"] = (
            str(Path(img).name), f"Science image #{j:03d} basename"
        )
    if mask_ratio is not None:
        meta["MSKRATIO"] = (
            round(float(mask_ratio), 3),
            "Ratio of pixels masked in coverage mask",
        )

    # Selected tuning fields (short keys, FITS-safe).
    meta["DETTHR"] = (
        float(cfg.detection_threshold) if cfg.detection_threshold is not None else None,
        "SE++ detection threshold (sigma)",
    )
    meta["DETMINAR"] = (
        int(cfg.detection_minimum_area) if cfg.detection_minimum_area is not None else None,
        "SE++ detection minimum area (pix)",
    )
    meta["KRNMINR"] = (
        float(cfg.auto_kron_min_radius) if cfg.auto_kron_min_radius is not None else None,
        "SE++ auto-kron min radius (pix)",
    )
    meta["PARTMINC"] = (
        float(cfg.partition_minimum_contrast)
        if cfg.partition_minimum_contrast is not None else None,
        "SE++ partition min contrast",
    )
    meta["FIXAPER"] = (
        ",".join(f"{a:g}" for a in cfg.fixed_apertures_arcsec),
        "Fixed apertures [arcsec]",
    )
    meta["PIXSCALE"] = (
        float(cfg.pixscale_arcsec), "Pixel scale [arcsec/pix]"
    )
    # Drop any None-valued cards we just inserted.
    for k in [
        "DETTHR", "DETMINAR", "KRNMINR", "PARTMINC",
    ]:
        v = meta.get(k)
        if isinstance(v, tuple) and v[0] is None:
            del meta[k]


def run_photometry(
    *,
    # --- Required per-run inputs ---
    science_images: Sequence[str],
    detection_image: str,
    reference_catalog: str,
    output_dir: str | Path | None = None,
    sepp_config_file: str | None = None,
    # --- Optional per-run inputs ---
    catalog_path: str | Path | None = None,
    catalog_name: str | None = None,
    coverage_mask: str | None = None,
    badpix_mask: str | None = None,
    tile_info: Any = None,
    run_name: str | None = None,
    overwrite: bool = False,
    deduplicate_by_filter: bool = True,
    max_measurement_images: int | None = None,
    standardize_catalog: bool = False,
    # --- Optional config bundle ---
    config: PhotometryConfig | None = None,
    # --- Schema / labelling (override config) ---
    bands: Sequence[str] | None = None,
    apertures: Sequence[str] | None = None,
    detection_label: str | None = None,
    coverage_mask_max_fraction: float | None = None,
    # --- SourceExtractor++ tuning (override config) ---
    detection_threshold: float | None = None,
    detection_minimum_area: int | None = None,
    auto_kron_min_radius: float | None = None,
    auto_kron_factor: float | None = None,
    background_cell_size: int | None = None,
    smoothing_box_size: int | None = None,
    partition_threshold_count: int | None = None,
    partition_minimum_area: int | None = None,
    partition_minimum_contrast: float | None = None,
    cleaning_minimum_area: int | None = None,
    flux_fractions: Sequence[float] | None = None,
    fixed_apertures_arcsec: Sequence[float] | None = None,
    pixscale_arcsec: float | None = None,
    thread_count: int | None = None,
    # --- Zero-point calibration (override config) ---
    match_radius_arcsec: float | None = None,
    mag_range: tuple[float, float] | None = None,
    spatial_poly_degree: int | None = None,
    polygon_margin: float | None = None,
    # --- Diagnostics (override config) ---
    save_residual_plots: bool | None = None,
    plot_axiscolor: str | None = None,
    # --- Depth estimation (override config) ---
    estimate_depth: bool | None = None,
    depth_n_sigma: float | None = None,
    depth_apertures: Sequence[str] | None = None,
    depth_n_empty_apertures: int | None = None,
    depth_empty_aperture: bool | None = None,
    depth_seed: int | None = None,
) -> PhotometryResult:
    """Run the full photometry pipeline for one image set.

    Parameters
    ----------
    science_images
        List of 7DS science (measurement) image paths. The ``FILTER`` FITS
        header keyword identifies the band of each image. The list order is
        irrelevant; images are reorganised internally by filter.
    detection_image
        Detection FITS image (built externally; see
        :mod:`phot7ds.detection`). Must have ``GAIN`` and ``SATURATE`` header
        keywords.
    reference_catalog
        Path to a Gaia XP synphot CSV with at least ``ra``, ``dec`` and
        ``mag_<band>`` columns for the bands present in ``science_images``.
    output_dir
        Working directory for all outputs. Created automatically if it does
        not exist (``mkdir -p``). Required when using ``catalog_name``.
    catalog_name
        Basename of the final calibrated catalog, written inside
        ``output_dir``. Examples: ``test_zp.fits``,
        ``T01234_20260512_DELVE.fits``, ``T01234_20260512_DELVE`` (the
        ``.fits`` extension is appended when missing). The leaf is kept
        verbatim -- there is no forced ``_phot.zp.fits`` suffix.
    catalog_path
        Alternative to ``catalog_name``: full path to the final catalog
        (``.fits`` appended if absent). Its parent directory is created
        if missing. Do not pass both ``catalog_name`` and ``catalog_path``.
    sepp_config_file
        Path to the SourceExtractor++ ``--config-file``. Required (either
        via this kwarg or via ``config.sepp_config_file``).
    coverage_mask
        Path to a precomputed coverage mask. If ``None``, one is built next
        to the output catalog by :func:`phot7ds.images.build_coverage_mask`.
    badpix_mask
        Optional bad-pixel mask FITS path passed to SE++ as
        ``--flag-image-badpix``.
    tile_info
        Optional single-row table (or dict-like) with ``ra1..ra4/dec1..dec4``.
        When provided, the Gaia XP reference is trimmed to the tile polygon
        before matching.
    run_name
        Basename stem for intermediate files (log, mask, raw catalog).
        Defaults to ``{detection_stem}_{detection_label}`` when neither
        ``catalog_name`` nor ``catalog_path`` is supplied, otherwise to
        the chosen filename without its ``.fits`` extension. The raw
        SE++ catalog is always saved as ``{run_name}_raw.fits``.
    overwrite
        If False and the final catalog already exists, the run is skipped
        and the existing files are summarised.
    deduplicate_by_filter
        If True (default), keep only one science image per filter (sorted by
        filename). Set to False to register every image with SE++ as-is.
    max_measurement_images
        Hard cap on the number of science images. When the post-dedup count
        exceeds this, :class:`ValueError` is raised. ``None`` disables the
        check.
    standardize_catalog
        If ``True``, reshape the table to the canonical column schema via
        :func:`~phot7ds.schema.standardize_catalog` before writing the FITS
        output. Default ``False`` (write SE++ + calibration columns as-is).
    config
        Optional :class:`PhotometryConfig` bundling the tuning knobs. Any
        explicit keyword argument always overrides the corresponding
        ``config`` field.
    bands, apertures, detection_label, coverage_mask_max_fraction
        Schema / labelling overrides.
    detection_threshold, detection_minimum_area, auto_kron_min_radius,
    auto_kron_factor, background_cell_size, smoothing_box_size,
    partition_threshold_count, partition_minimum_area,
    partition_minimum_contrast, cleaning_minimum_area, flux_fractions,
    fixed_apertures_arcsec, pixscale_arcsec, thread_count
        SourceExtractor++ tuning overrides.
    match_radius_arcsec, mag_range, spatial_poly_degree, polygon_margin
        Calibration overrides.
    save_residual_plots, plot_axiscolor
        Diagnostic overrides.
    estimate_depth, depth_n_sigma, depth_apertures,
    depth_n_empty_apertures, depth_empty_aperture, depth_seed
        Depth-estimation overrides. When ``estimate_depth`` is ``True``
        (default), both the error-curve fit and the empty-aperture method
        are run for the apertures listed in ``depth_apertures`` (default
        ``('aper05',)``); results are written to the log, the manifest, and
        the FITS catalog header (``UL{N}EM..`` / ``UL{N}RM..`` keys).
    detection_label
        Label of the detection image type (``'DELVE'`` or ``'7DT'``).
        Selects the SE++ tuning preset (see :mod:`phot7ds.presets`). Any
        explicit override of a tuning field wins over the preset.

    Returns
    -------
    PhotometryResult
        Paths and counts describing the outputs.
    """
    cfg = _merge_config(config=config, overrides=locals())
    cfg = _apply_detection_preset(cfg, detection_label)

    if not cfg.sepp_config_file:
        raise ValueError(
            "sepp_config_file is required (pass either directly or via config)."
        )

    work_dir, raw_catalog_path, zp_catalog_path, log_file, manifest_path, run_name = (
        _resolve_output_paths(
            output_dir=output_dir,
            catalog_path=catalog_path,
            catalog_name=catalog_name,
            run_name=run_name,
            detection_image=detection_image,
            detection_label=cfg.detection_label,
        )
    )

    configure_logging(log_file=log_file)

    if not overwrite and zp_catalog_path.exists():
        log.info("Final catalog already exists, skipping: %s", zp_catalog_path)
        return _summarise_existing(zp_catalog_path, manifest_path, log_file)

    log.info("=== phot7ds run: %s ===", run_name)
    log.info("Detection image: %s", detection_image)
    log.info("Science images : %d", len(science_images))

    sciimgs = _select_measurement_images(
        science_images, deduplicate=deduplicate_by_filter
    )
    if max_measurement_images is not None and len(sciimgs) > max_measurement_images:
        raise ValueError(
            f"Too many measurement images ({len(sciimgs)} > "
            f"{max_measurement_images}). Raise max_measurement_images or "
            "deduplicate the input list."
        )

    mask_ratio: float | None = None
    if coverage_mask is None:
        coverage_mask = str(work_dir / f"{run_name}_mask.fits")
        coverage_mask, mask_ratio = build_coverage_mask(
            detection_image=detection_image,
            science_images=sciimgs,
            output_path=coverage_mask,
            overwrite=overwrite,
            max_masked_fraction=cfg.coverage_mask_max_fraction,
        )
    else:
        try:
            with fits.open(coverage_mask) as hdul:
                v = hdul[0].header.get("MSKRATIO")
                mask_ratio = float(v) if v is not None else None
        except Exception:
            mask_ratio = None

    band_names, saturation_values = extract_band_names_and_saturation(sciimgs)
    log.info("Per-image bands: %s", band_names)

    det_hdr = fits.getheader(detection_image)
    det_gain = float(det_hdr.get("GAIN", 1.0))
    det_saturate = float(det_hdr.get("SATURATE", 60000.0))

    aper_radius_pix = [
        round(aper / cfg.pixscale_arcsec, 3)
        for aper in cfg.fixed_apertures_arcsec
    ]

    python_config_file = str(work_dir / f"{run_name}_sepp.py")
    generate_sepp_python_config(
        config_file=python_config_file,
        sciimgs=sciimgs,
        band_names=band_names,
        saturation_values=saturation_values,
        gain=det_gain,
        aperture_photometry=True,
        aper_radius_pix=aper_radius_pix,
        fixed_apertures=list(cfg.fixed_apertures_arcsec),
    )

    cmd = build_sepp_command(
        python_config_file=python_config_file,
        sepp_config_file=cfg.sepp_config_file,
        detection_image=detection_image,
        detection_gain=det_gain,
        detection_saturate=det_saturate,
        catalog_path=str(raw_catalog_path),
        coverage_mask=coverage_mask,
        badpix_mask=badpix_mask,
        detection_threshold=cfg.detection_threshold,
        detection_minimum_area=cfg.detection_minimum_area,
        auto_kron_min_radius=cfg.auto_kron_min_radius,
        auto_kron_factor=cfg.auto_kron_factor,
        background_cell_size=cfg.background_cell_size,
        smoothing_box_size=cfg.smoothing_box_size,
        partition_threshold_count=cfg.partition_threshold_count,
        partition_minimum_area=cfg.partition_minimum_area,
        partition_minimum_contrast=cfg.partition_minimum_contrast,
        flux_fractions=cfg.flux_fractions,
        clean_param=cfg.cleaning_minimum_area,
        thread_count=cfg.thread_count,
        log_file=str(log_file),
    )
    run_sepp(cmd, check=True)

    cat = split_array_columns_to_per_filter(
        str(raw_catalog_path),
        band_names=band_names,
        flux_fractions=cfg.flux_fractions,
        overwrite=True,
        fixed_apertures=list(cfg.fixed_apertures_arcsec),
    )
    if cat is None:
        raise RuntimeError("SE++ catalog could not be loaded for post-processing")

    ref_cat = load_gaiaxp_reference(
        reference_catalog,
        bands=band_names,
        tile_info=tile_info,
        margin=cfg.polygon_margin,
    )
    if len(ref_cat) == 0:
        log.warning("No Gaia XP reference sources within tile polygon")

    plot_dir = str(work_dir / "figures") if cfg.save_residual_plots else None
    calibrate_zeropoints(
        cat,
        ref_cat=ref_cat,
        band_names=band_names,
        apertures=list(cfg.apertures),
        match_radius_arcsec=cfg.match_radius_arcsec,
        mag_range=cfg.mag_range,
        spatial_poly_degree=cfg.spatial_poly_degree,
        plot_residuals=cfg.save_residual_plots,
        plot_dir=plot_dir,
        plot_title_extra=run_name,
    )

    # Persist constant ZP + scatter as FITS-safe header keys
    # (e.g. ZP05MG, ZE05MG, ZP10M575).
    zeropoints_to_meta(
        cat.meta,
        cat.meta.get("zeropoints"),
        cat.meta.get("zeropoint_scatter"),
    )

    depth_results: dict = {}
    if cfg.estimate_depth:
        depth_apertures = [a for a in cfg.depth_apertures if a in cfg.apertures]
        if not depth_apertures:
            depth_apertures = list(cfg.apertures)
        zeropoints = cat.meta.get("zeropoints")
        band_to_image = dict(zip(band_names, sciimgs))
        depth_results = estimate_depths(
            cat,
            bands=band_names,
            apertures=depth_apertures,
            n_sigma=cfg.depth_n_sigma,
            pixscale_arcsec=cfg.pixscale_arcsec,
            science_images=band_to_image,
            coverage_mask=coverage_mask,
            zeropoints=zeropoints,
            n_empty_apertures=cfg.depth_n_empty_apertures,
            seed=cfg.depth_seed,
            do_error_curve=True,
            do_empty_apertures=cfg.depth_empty_aperture,
        )
        if depth_results:
            log.info(
                "%d-sigma depth summary:\n%s",
                int(round(cfg.depth_n_sigma)),
                format_depth_table(depth_results, n_sigma=cfg.depth_n_sigma),
            )
            depth_results_to_meta(
                cat.meta, depth_results, n_sigma=cfg.depth_n_sigma,
            )

    strip_nonfits_units(cat)
    # Drop dict-valued meta entries (zeropoints map etc.) that cannot survive
    # the FITS header round-trip; keep them on the in-memory table only.
    for _k in list(cat.meta):
        if isinstance(cat.meta[_k], dict):
            del cat.meta[_k]
    _annotate_catalog_meta(
        cat.meta,
        detection_image=detection_image,
        coverage_mask=coverage_mask,
        badpix_mask=badpix_mask,
        reference_catalog=reference_catalog,
        science_images=sciimgs,
        detection_label=cfg.detection_label,
        mask_ratio=mask_ratio,
        cfg=cfg,
        run_name=run_name,
    )
    if standardize_catalog:
        schema = build_canonical_schema(
            bands=cfg.bands,
            apertures=cfg.apertures,
            flux_fractions=cfg.flux_fractions,
        )
        cat = apply_standard_catalog(cat, schema)
        log.info(
            "Unified schema: %d cols (placeholders=%d, dropped_dups=%d, extras=%d)",
            len(cat.colnames),
            cat.meta.get("NPLACE", 0),
            cat.meta.get("NDUPS", 0),
            cat.meta.get("NEXTRA", 0),
        )
    cat.write(str(zp_catalog_path), format="fits", overwrite=True)

    manifest = {
        "run_name": run_name,
        "detection_image": str(detection_image),
        "detection_label": cfg.detection_label,
        "coverage_mask": str(coverage_mask),
        "mask_ratio": mask_ratio,
        "badpix_mask": str(badpix_mask) if badpix_mask else None,
        "reference_catalog": str(reference_catalog),
        "science_images": list(sciimgs),
        "config": cfg.to_dict(),
    }
    if depth_results:
        manifest["depths"] = {
            f"{aper}__{band}": entry
            for (aper, band), entry in depth_results.items()
        }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    log.info("Wrote calibrated catalog: %s", zp_catalog_path)

    return PhotometryResult(
        catalog_path=str(zp_catalog_path),
        manifest_path=str(manifest_path),
        log_file=str(log_file),
        n_sources=len(cat),
    )


def _merge_config(
    *,
    config: PhotometryConfig | None,
    overrides: dict[str, Any],
) -> PhotometryConfig:
    """Merge ``config`` with explicit kwargs from :func:`run_photometry`.

    Explicit kwargs (anything that is not ``None`` in ``overrides``) win
    over ``config``; fields absent from both fall back to dataclass
    defaults.
    """
    cfg = config or PhotometryConfig()
    cfg_fields = {f.name for f in dataclasses.fields(PhotometryConfig)}
    changes: dict[str, Any] = {}
    for name in cfg_fields:
        if name in _RUN_ONLY_KWARGS:
            continue
        val = overrides.get(name)
        if val is None:
            continue
        if isinstance(val, list):
            val = tuple(val)
        changes[name] = val
    sepp_config_file = overrides.get("sepp_config_file")
    if sepp_config_file is not None:
        changes["sepp_config_file"] = sepp_config_file
    return cfg.replace(**changes) if changes else cfg


def _select_measurement_images(
    science_images: Sequence[str], *, deduplicate: bool
) -> list[str]:
    """Optionally deduplicate the science image list by filter.

    7DS surveys can produce multiple coadds per filter (different epochs);
    by default this helper keeps one representative image per filter. If the
    filter cannot be inferred for any image, the caller's original list is
    returned unchanged.
    """
    if not deduplicate:
        return list(science_images)
    bands_dict, *_ = get_filter_definitions(unit="angstrom")
    dict_sciimgs = organize_images_by_filter(
        science_images,
        bands_dict,
        filter_source="filename",
        output_form="dict",
        keep_duplicates=False,
    )
    selected = [v for v in dict_sciimgs.values() if isinstance(v, str)]
    if not selected:
        return list(science_images)
    if len(selected) != len(science_images):
        log.info(
            "Deduplicated science images: %d -> %d",
            len(science_images),
            len(selected),
        )
    return selected


def _summarise_existing(
    catalog_path: Path, manifest_path: Path, log_file: Path
) -> PhotometryResult:
    with fits.open(catalog_path) as hdul:
        n = hdul[1].header.get("NAXIS2", 0)
    return PhotometryResult(
        catalog_path=str(catalog_path),
        manifest_path=str(manifest_path),
        log_file=str(log_file),
        n_sources=int(n),
    )


__all__ = ["run_photometry", "PhotometryResult"]
