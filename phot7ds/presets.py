"""
Default SourceExtractor++ tuning per detection-image type.

When :func:`phot7ds.run_photometry` runs, the ``detection_label`` field
of :class:`~phot7ds.PhotometryConfig` selects a preset of SE++ knobs.
Any field a user explicitly set on the config (or as a keyword on
``run_photometry``) always wins; the preset only fills in values that
were left at ``None`` on the merged config.

Presets currently shipped:

============== =================================================
detection_label  Notes
============== =================================================
``'DELVE'``      Deep DELVE-DR3 detection image (default).
``'7DT'``        Native 7DT detection coadd.
============== =================================================

Add a new entry by extending :data:`DETECTION_PRESETS`.
"""
from __future__ import annotations

from typing import Any

# Fields whose value depends on the detection-image type. Keep this list in
# sync with :class:`phot7ds.config.PhotometryConfig`.
PRESET_TUNING_FIELDS: tuple[str, ...] = (
    "detection_threshold",
    "detection_minimum_area",
    "auto_kron_min_radius",
    "partition_threshold_count",
    "partition_minimum_contrast",
)

# Per-preset overrides. Anything missing falls back to ``DEFAULT_TUNING``.
DETECTION_PRESETS: dict[str, dict[str, Any]] = {
    "DELVE": {
        "detection_threshold": 10.0,
        "detection_minimum_area": 9,
        "auto_kron_min_radius": 8.0,
        "partition_threshold_count": 32,
        "partition_minimum_contrast": 5.0e-4,
    },
    "7DT": {
        "detection_threshold": 1.5,
        "detection_minimum_area": 9,
        "auto_kron_min_radius": 3.5,
        "partition_threshold_count": 32,
        "partition_minimum_contrast": 1.0e-5,
    },
}

# Global fallback for any preset that doesn't specify a value.
DEFAULT_TUNING: dict[str, Any] = DETECTION_PRESETS["7DT"]


def resolve_preset(detection_label: str | None) -> dict[str, Any]:
    """Return the preset dict for ``detection_label`` (case-insensitive).

    Unknown labels fall back to :data:`DEFAULT_TUNING`. Missing keys in a
    matched preset are also filled from :data:`DEFAULT_TUNING`.
    """
    if detection_label is None:
        preset = DEFAULT_TUNING
    else:
        key = str(detection_label).strip().upper()
        preset = DETECTION_PRESETS.get(key, DEFAULT_TUNING)
    return {**DEFAULT_TUNING, **preset}


__all__ = [
    "PRESET_TUNING_FIELDS",
    "DETECTION_PRESETS",
    "DEFAULT_TUNING",
    "resolve_preset",
]
