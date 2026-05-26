"""
Optional configuration bundle for :func:`phot7ds.run_photometry`.

Most callers will simply pass keyword arguments directly to
:func:`run_photometry`. This module provides a frozen dataclass that bundles
the common knobs so they can be reused across runs::

    from phot7ds import PhotometryConfig, run_photometry

    cfg = PhotometryConfig(
        sepp_config_file="/path/7ds_sepp.config",
        detection_threshold=10.0,
        fixed_apertures_arcsec=(5.0, 10.0),
    )

    result = run_photometry(
        science_images=[...],
        detection_image="...",
        reference_catalog="...",
        output_dir="...",
        config=cfg,
    )

Any kwarg passed to :func:`run_photometry` always overrides the equivalent
field on ``config``.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

from .filters import DEFAULT_BANDS


@dataclass(frozen=True)
class PhotometryConfig:
    """Reusable settings for :func:`phot7ds.run_photometry`.

    All fields are optional except ``sepp_config_file`` (which must always
    point at a valid SourceExtractor++ ``--config-file``).
    """

    # --- Required ---
    sepp_config_file: str = ""

    # --- Schema ---
    bands: tuple[str, ...] = tuple(DEFAULT_BANDS)
    apertures: tuple[str, ...] = ("aper05", "aper10", "auto")
    detection_label: str = "DELVE"
    coverage_mask_max_fraction: float = 0.5

    # --- SourceExtractor++ tuning ---
    # Fields defaulting to ``None`` are filled from the detection preset
    # selected by ``detection_label`` (see :mod:`phot7ds.presets`). Pass an
    # explicit value to override the preset.
    detection_threshold: float | None = None
    detection_minimum_area: int | None = None
    auto_kron_min_radius: float | None = None
    partition_threshold_count: int | None = None
    partition_minimum_contrast: float | None = None
    # Invariant tuning (not preset-controlled):
    auto_kron_factor: float = 2.5
    background_cell_size: int = 256
    smoothing_box_size: int = 3
    partition_minimum_area: int = 9
    cleaning_minimum_area: int = 8
    flux_fractions: tuple[float, ...] = (0.5, 0.9)
    fixed_apertures_arcsec: tuple[float, ...] = (5.0, 10.0)
    pixscale_arcsec: float = 0.505
    thread_count: int = 4

    # --- Zero-point calibration ---
    match_radius_arcsec: float = 1.0
    mag_range: tuple[float, float] = (12.0, 16.0)
    spatial_poly_degree: int = 2
    polygon_margin: float = 0.06

    # --- Diagnostics ---
    save_residual_plots: bool = False
    plot_axiscolor: str = "elongation"

    # --- Depth estimation ---
    estimate_depth: bool = True
    depth_n_sigma: float = 5.0
    depth_apertures: tuple[str, ...] = ("aper05",)
    depth_n_empty_apertures: int = 2000
    depth_empty_aperture: bool = True
    depth_seed: int = 42

    def replace(self, **changes: Any) -> "PhotometryConfig":
        """Return a copy with ``changes`` overridden (frozen dataclass safe)."""
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-/dict-serialisable representation."""
        return dataclasses.asdict(self)


__all__ = ["PhotometryConfig"]
