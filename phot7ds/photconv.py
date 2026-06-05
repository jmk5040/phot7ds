"""
Magnitude / flux conversions and 7DS filter colorization.

Ported from the 7DT ``Utils_7DT`` helpers so phot7ds is standalone. The
flux conventions match the EAzY / FAST++ input-catalog format (AB zeropoint
25.0 by default).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

# AB zeropoints by output unit.
_ZEROPOINTS = {"uJy": 23.93, "erg": -48.57, "FAST": 25.00}


def rsig(x: float, sig: int = 2) -> float:
    """Round ``x`` to ``sig`` significant figures (mantissa decimals)."""
    dex = int(np.floor(np.log10(x)))
    fac = round(x * 10 ** (-dex), sig)
    return np.double(f"{fac}E{dex}")


def AB2Jy(mag: float, unit: str = "uJy") -> float:
    """Convert an AB magnitude to flux in the given zeropoint system."""
    return 10 ** ((_ZEROPOINTS[unit] - mag) / 2.5)


def eAB2Jy(ABmag: float, ABerr: float, unit: str = "uJy") -> float:
    """Convert an AB magnitude error to a flux error."""
    return 2.5 / np.log(10) * ABerr * AB2Jy(ABmag, unit)


def mag_to_flux(maglist: Sequence[float], unit: str = "FAST") -> list[float]:
    """Convert AB magnitudes to fluxes for an EAzY/FAST++ input catalog.

    Masked or out-of-range (``mag <= 5`` or ``mag >= 30``) entries map to
    the sentinel ``-99`` used by FAST++ for "not observed".
    """
    fluxlist: list[float] = []
    for mag in maglist:
        if np.ma.is_masked(mag):
            fluxlist.append(-99)
        elif 5 < mag < 30:
            fluxlist.append(rsig(AB2Jy(mag, unit=unit), 5))
        else:
            fluxlist.append(-99)
    return fluxlist


def mag_to_flux_err(
    maglist: Sequence[float], errlist: Sequence[float], unit: str = "FAST"
) -> list[float]:
    """Convert AB magnitude errors to flux errors (sentinel ``-99``).

    Falls back to a 0.1 mag error when the supplied error is invalid.
    """
    fluxlist: list[float] = []
    for mag, err in zip(maglist, errlist):
        if np.ma.is_masked(mag):
            fluxlist.append(-99)
        elif 5 < mag < 30:
            try:
                fluxlist.append(rsig(eAB2Jy(mag, err, unit=unit), 5))
            except Exception:
                fluxlist.append(rsig(eAB2Jy(mag, 0.1, unit=unit), 5))
        else:
            fluxlist.append(-99)
    return fluxlist


def filter_colorization(unit: str = "angstrom"):
    """Return 7DS band definitions and a wavelength colour map.

    Returns ``(bands_dict, bands_width, bands_color, lambda_to_color,
    lambda_to_band)`` where ``bands_dict`` maps band name -> central
    wavelength (the key order also defines the canonical band order).
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm

    if unit == "angstrom":
        broad_bands = {"g": 4770, "r": 6231, "i": 7625}
        bb_width = {"g": 1263 / 2, "r": 1149 / 2, "i": 1239 / 2}
        bb_color = {"g": "lightgreen", "r": "lightcoral", "i": "coral"}
        medium_bands = {f"m{w}": w * 10 for w in range(400, 900, 25)}
        mb_width = {f"m{w}": 125 for w in range(400, 900, 25)}
    elif unit == "nm":
        broad_bands = {"g": 477.0, "r": 623.1, "i": 762.5}
        bb_width = {"g": 126.3 / 2, "r": 114.9 / 2, "i": 123.9 / 2}
        bb_color = {"g": "lightgreen", "r": "lightcoral", "i": "coral"}
        medium_bands = {f"m{w}": w for w in range(400, 900, 25)}
        mb_width = {f"m{w}": 12.5 for w in range(400, 900, 25)}
    else:
        raise ValueError("unit must be 'angstrom' or 'nm'")

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


__all__ = [
    "rsig",
    "AB2Jy",
    "eAB2Jy",
    "mag_to_flux",
    "mag_to_flux_err",
    "filter_colorization",
]
