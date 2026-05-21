"""
7DT filter definitions: broad ``g/r/i`` and medium-band ``m400 ... m875``.

The wavelengths and bandwidths come from the 7DT survey design. The colour
maps are convenience defaults for plotting; they are not used by the
photometry pipeline itself.
"""
from __future__ import annotations

from typing import Literal


def get_filter_definitions(unit: Literal["angstrom", "nm"] = "angstrom"):
    """Return canonical 7DT filter definitions.

    Parameters
    ----------
    unit
        Wavelength unit: ``'angstrom'`` (default) or ``'nm'``.

    Returns
    -------
    bands_dict : dict[str, float]
        Filter name -> central wavelength.
    bands_width : dict[str, float]
        Filter name -> half-bandwidth.
    bands_color : dict[str, str | tuple]
        Filter name -> matplotlib colour (broad bands: named; medium bands:
        coolwarm colormap value).
    lambda_to_color : dict[float, str | tuple]
        Wavelength -> colour, useful for line plots keyed by lambda.
    lambda_to_band : dict[float, str]
        Wavelength -> filter name.
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm

    if unit == "angstrom":
        broad_bands = {"g": 4770, "r": 6231, "i": 7625}
        bb_width = {"g": 1263 / 2, "r": 1149 / 2, "i": 1239 / 2}
        medium_bands = {f"m{w}": w * 10 for w in range(400, 900, 25)}
        mb_width = {f"m{w}": 125 for w in range(400, 900, 25)}
    elif unit == "nm":
        broad_bands = {"g": 477.0, "r": 623.1, "i": 762.5}
        bb_width = {"g": 126.3 / 2, "r": 114.9 / 2, "i": 123.9 / 2}
        medium_bands = {f"m{w}": w for w in range(400, 900, 25)}
        mb_width = {f"m{w}": 12.5 for w in range(400, 900, 25)}
    else:
        raise ValueError(f"unit must be 'angstrom' or 'nm', got {unit!r}")

    bb_color = {"g": "lightgreen", "r": "lightcoral", "i": "coral"}
    medium_wavelengths = list(range(400, 900, 25))
    norm = plt.Normalize(min(medium_wavelengths), max(medium_wavelengths))
    cmap = cm.coolwarm
    mb_color = {f"m{w}": cmap(norm(w)) for w in medium_wavelengths}

    bands_dict = {**broad_bands, **medium_bands}
    bands_width = {**bb_width, **mb_width}
    bands_color = {**bb_color, **mb_color}
    lambda_to_color = {bands_dict[b]: bands_color[b] for b in bands_dict}
    lambda_to_band = {v: k for k, v in bands_dict.items()}

    return bands_dict, bands_width, bands_color, lambda_to_color, lambda_to_band


# Canonical filter set the unified catalog schema is built around.
# Every saved catalog will contain these bands in this order; any missing
# band is filled with a placeholder column.
DEFAULT_BANDS: list[str] = [
    "g", "r", "i",
    "m400", "m425", "m450", "m475", "m500", "m525", "m550", "m575",
    "m600", "m625", "m650", "m675", "m700", "m725", "m750", "m775",
    "m800", "m825", "m850", "m875",
]


__all__ = ["get_filter_definitions", "DEFAULT_BANDS"]
