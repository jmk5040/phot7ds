"""
phot7ds - photometric catalog pipeline for the 7DS survey suite (RIS / WFS / IMS).

Given:

1. A list of 7DS science image paths (one or more per filter).
2. A detection image (a co-added DELVE mosaic, a 7DT detection coadd, ...).
3. A reference catalog for zero-point calibration (Gaia XP synphot CSV).
4. A SourceExtractor++ ``--config-file``.

:func:`run_photometry` produces a single zero-point-calibrated FITS catalog.
Pass ``standardize_catalog=True`` for the fixed canonical column layout.

Quickstart::

    from phot7ds import run_photometry

    result = run_photometry(
        science_images=[
            "/path/T01_g.fits",
            "/path/T01_r.fits",
            "/path/T01_m500.fits",
        ],
        detection_image="/path/detection.fits",
        reference_catalog="/path/gaiaxp.csv",
        output_dir="/path/out/",
        sepp_config_file="/path/7ds_sepp.config",
        # optional knobs:
        detection_threshold=10.0,
        fixed_apertures_arcsec=(5.0, 10.0),
        save_residual_plots=True,
    )
    print(result.catalog_path)

For repeatable runs you can pre-bundle the tuning knobs in a
:class:`PhotometryConfig` and pass it as ``config=cfg``::

    from phot7ds import PhotometryConfig, run_photometry

    cfg = PhotometryConfig(
        sepp_config_file="/path/7ds_sepp.config",
        detection_threshold=10.0,
    )
    result = run_photometry(
        science_images=[...],
        detection_image=...,
        reference_catalog=...,
        output_dir=...,
        config=cfg,
    )
"""
from __future__ import annotations

from ._logging import configure_logging
from .batch import BatchResult, batch_run
from .calibration import (
    apply_spatial_zeropoint,
    calibrate_zeropoints,
    load_gaiaxp_reference,
    match_nearest,
)
from .config import PhotometryConfig
from .config_io import (
    ensure_sepp_config,
    ensure_swarp_config,
    require_gaiaxp_reference,
    require_tile_table,
)
from .depth import (
    classical_limiting_mag,
    depth_from_empty_apertures,
    depth_from_error_curve,
    depth_results_to_meta,
    empty_aperture_sky_sigma,
    estimate_depths,
    format_depth_table,
    magerr_threshold_for_n_sigma,
    zeropoints_to_meta,
)
from .filters import DEFAULT_BANDS, get_filter_definitions
from .images import (
    build_coverage_mask,
    extract_band_names_and_saturation,
    organize_images_by_filter,
)
from .pipeline import PhotometryResult, run_photometry
from .presets import (
    DEFAULT_TUNING,
    DETECTION_PRESETS,
    PRESET_TUNING_FIELDS,
    resolve_preset,
)
from .schema import (
    build_canonical_schema,
    load_unified_catalog,
    standardize_catalog,
    strip_nonfits_units,
)
from .sepp import (
    DEFAULT_OUTPUT_PROPERTIES,
    build_sepp_command,
    generate_sepp_python_config,
    run_sepp,
    split_array_columns_to_per_filter,
)
from .tile_geometry import trim_to_tile_polygon

__version__ = "0.3.0"

__all__ = [
    "__version__",
    # primary API
    "run_photometry",
    "PhotometryResult",
    "batch_run",
    "BatchResult",
    "PhotometryConfig",
    # config / preset bootstrap helpers
    "ensure_sepp_config",
    "ensure_swarp_config",
    "require_tile_table",
    "require_gaiaxp_reference",
    "DETECTION_PRESETS",
    "DEFAULT_TUNING",
    "PRESET_TUNING_FIELDS",
    "resolve_preset",
    # filters / images
    "DEFAULT_BANDS",
    "get_filter_definitions",
    "organize_images_by_filter",
    "extract_band_names_and_saturation",
    "build_coverage_mask",
    # SE++
    "DEFAULT_OUTPUT_PROPERTIES",
    "generate_sepp_python_config",
    "build_sepp_command",
    "run_sepp",
    "split_array_columns_to_per_filter",
    # calibration
    "load_gaiaxp_reference",
    "apply_spatial_zeropoint",
    "calibrate_zeropoints",
    "match_nearest",
    # depth
    "classical_limiting_mag",
    "depth_from_empty_apertures",
    "depth_from_error_curve",
    "depth_results_to_meta",
    "empty_aperture_sky_sigma",
    "estimate_depths",
    "format_depth_table",
    "magerr_threshold_for_n_sigma",
    "zeropoints_to_meta",
    # schema / IO
    "build_canonical_schema",
    "standardize_catalog",
    "strip_nonfits_units",
    "load_unified_catalog",
    # geometry / logging
    "trim_to_tile_polygon",
    "configure_logging",
]
