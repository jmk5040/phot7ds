"""
N-sigma limiting magnitude (a.k.a. photometric depth) estimation.

Two complementary methods are exposed:

1. :func:`depth_from_error_curve` -- exponential fit to ``MAGERR`` vs ``MAG``.
   Pure-catalog approach: fast, no I/O, but depends on calibrated magnitude
   errors being well-behaved at the faint end. The implementation is a
   stabilised version of the classic "find where magerr ~ 0.2" trick.

2. :func:`depth_from_empty_apertures` -- empty-aperture sky sigma plus the
   classical SExtractor formula ``m_lim = ZP - 2.5 log10(N * sigma_aper)``.
   Random circular apertures are placed on a science image at positions
   that avoid catalog sources; the sigma-clipped standard deviation of
   their summed fluxes is taken as ``sigma_aper``. This naturally captures
   correlated background noise (which a per-pixel sigma would miss),
   without paying the cost of writing a full background-RMS check image.

The top-level :func:`estimate_depths` runs both methods (when inputs allow)
for every ``(band, aperture)`` and returns a tidy dict suitable for logging
and FITS-header embedding.
"""
from __future__ import annotations

import logging
import os
import warnings
from typing import Mapping, Sequence

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.wcs import WCS
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree

log = logging.getLogger(__name__)


def _sky_to_pixel(
    header: fits.Header, ra: np.ndarray, dec: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Convert source sky coords to 0-based pixel coords on ``header``'s WCS.

    Returns ``(x, y)`` arrays (0-based, matching numpy indexing) or
    ``(None, None)`` if the header carries no usable celestial WCS. This is
    what makes the empty-aperture sky sigma valid when the science image is
    *not* on the detection grid (e.g. single-frame inputs): each image's own
    WCS maps the shared source catalog onto its native pixel frame.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wcs = WCS(header)
            if not wcs.has_celestial:
                return None, None
            x, y = wcs.all_world2pix(
                np.asarray(ra, dtype=float), np.asarray(dec, dtype=float), 0
            )
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("WCS sky->pixel conversion failed: %s", exc)
        return None, None


# 1 magnitude = 1.0857 mag at SNR=1 (Pogson). Used to convert N-sigma to
# the corresponding magnitude-error threshold.
_POGSON = 2.5 / np.log(10.0)  # = 1.085736...


def magerr_threshold_for_n_sigma(n_sigma: float) -> float:
    """Return the magnitude-error value corresponding to S/N = ``n_sigma``.

    Uses the small-error approximation ``magerr ~ 1.0857 / SNR``. For
    ``n_sigma=5`` this returns ~0.217 mag (often rounded to 0.2 mag).
    """
    return float(_POGSON / float(n_sigma))


def classical_limiting_mag(
    n_sigma: float, zeropoint: float, sky_sigma_aperture: float
) -> float:
    """Classical N-sigma limit ``ZP - 2.5 log10(N * sigma_aper)``.

    Parameters
    ----------
    n_sigma
        Detection significance threshold (5 by convention).
    zeropoint
        Photometric zero-point in magnitudes such that
        ``mag = ZP - 2.5 log10(flux_ADU)``.
    sky_sigma_aperture
        Standard deviation of empty-aperture fluxes, in the same ADU scale
        as ``zeropoint``.
    """
    if sky_sigma_aperture <= 0 or not np.isfinite(sky_sigma_aperture):
        return float("nan")
    return float(zeropoint - 2.5 * np.log10(n_sigma * sky_sigma_aperture))


# ----------------------------------------------------------------------------
# Method 1: error-curve fitting
# ----------------------------------------------------------------------------


def _exp_curve(x: np.ndarray, a: float, b: float) -> np.ndarray:
    # ``curve_fit`` explores wide parameter ranges; numpy raises RuntimeWarning
    # on overflow which is noise here. The fitter handles ``inf`` correctly.
    with np.errstate(over="ignore", invalid="ignore"):
        return a * np.exp(b * x)


def depth_from_error_curve(
    cat: Table,
    band: str,
    aperture: str,
    *,
    n_sigma: float = 5.0,
    elongation_max: float = 1.5,
    snr_min: float = 5.0,
    mincut_start: float = 0.03,
    mincut_max: float = 0.20,
    mincut_step: float = 0.005,
    err_upper: float = 0.5,
    r_squared_min: float = 0.8,
    depth_max: float = 30.0,
    min_points: int = 30,
) -> dict[str, float | int | str]:
    """Estimate the N-sigma depth from a fit to ``MAGERR`` vs ``MAG``.

    The function selects point-like, well-detected sources, then fits an
    exponential ``magerr = a * exp(b * mag)`` to their magnitude/error pairs.
    The depth is the magnitude at which the curve crosses
    ``magerr_threshold = 1.0857 / n_sigma`` (i.e. the magnitude where the
    photometric SNR is expected to fall to ``n_sigma``).

    A simple recursive loop widens the lower magnitude-error cut until the
    fit's R^2 exceeds ``r_squared_min``, in case the bright end is dominated
    by the ZP scatter floor.

    Parameters
    ----------
    cat
        Output catalog of :func:`phot7ds.run_photometry` (post-calibration).
    band
        Band suffix in the catalog column names, e.g. ``'g'``, ``'m475'``.
    aperture
        Aperture prefix, e.g. ``'aper05'``, ``'aper10'``, ``'auto'``. The
        function reads ``f"{aperture}_mag_{band}"`` and
        ``f"{aperture}_mag_err_{band}"``.
    n_sigma
        Detection threshold in sigma (5 by convention).
    elongation_max
        Reject sources rounder than this in the ``elongation`` column.
    snr_min
        Reject sources whose ``snrratio`` (SE++ SNRRatio) is below this
        bound to keep the fit anchored on robust detections.
    mincut_start, mincut_max, mincut_step
        Lower bound on ``magerr`` used in the fit, widened iteratively if
        the fit fails the R^2 / depth-range sanity checks.
    err_upper
        Upper bound on ``magerr`` used in the fit.
    r_squared_min, depth_max
        Sanity checks on the converged fit.
    min_points
        Minimum number of sources required for a fit attempt.

    Returns
    -------
    dict
        ``{depth, magerr_threshold, n_points, mincut, r_squared, status}``.
        ``status`` is one of ``'ok'``, ``'no_columns'``, ``'no_points'``,
        ``'fit_failed'``.
    """
    mag_col = f"{aperture}_mag_{band}"
    err_col = f"{aperture}_mag_err_{band}"
    flag_col = f"{aperture}_flags_{band}"
    out = {
        "depth": float("nan"),
        "magerr_threshold": magerr_threshold_for_n_sigma(n_sigma),
        "n_points": 0,
        "mincut": float("nan"),
        "r_squared": float("nan"),
        "status": "ok",
    }
    if mag_col not in cat.colnames or err_col not in cat.colnames:
        out["status"] = "no_columns"
        return out

    mag = np.asarray(cat[mag_col], dtype=float)
    err = np.asarray(cat[err_col], dtype=float)
    base = np.isfinite(mag) & np.isfinite(err) & (err > 0)

    if flag_col in cat.colnames:
        base &= np.asarray(cat[flag_col], dtype=int) == 0
    if "source_flags" in cat.colnames:
        base &= np.asarray(cat["source_flags"], dtype=int) < 4
    if "elongation" in cat.colnames:
        elong = np.asarray(cat["elongation"], dtype=float)
        base &= np.isfinite(elong) & (elong < elongation_max)
    if "snrratio" in cat.colnames:
        snr = np.asarray(cat["snrratio"], dtype=float)
        base &= np.isfinite(snr) & (snr > snr_min)

    threshold = out["magerr_threshold"]
    mincut = float(mincut_start)
    while True:
        sel = base & (err >= mincut) & (err <= err_upper)
        n = int(sel.sum())
        out["n_points"] = n
        out["mincut"] = mincut
        if n < min_points:
            out["status"] = "no_points"
            return out
        try:
            weights = 1.0 / err[sel]
            params, _ = curve_fit(
                _exp_curve,
                mag[sel],
                err[sel],
                sigma=weights,
                p0=(1e-10, 0.7),
                maxfev=20000,
            )
            a, b = params
            if not np.isfinite(a) or not np.isfinite(b) or a <= 0 or b <= 0:
                raise RuntimeError("non-physical fit parameters")
            depth = float(np.log(threshold / a) / b)
            y_pred = _exp_curve(mag[sel], a, b)
            residuals = err[sel] - y_pred
            ss_total = float(np.sum((err[sel] - np.mean(err[sel])) ** 2))
            ss_residual = float(np.sum(residuals ** 2))
            r_squared = 1.0 - ss_residual / ss_total if ss_total > 0 else 0.0
            out["r_squared"] = r_squared
            if (
                np.isfinite(depth)
                and depth < depth_max
                and r_squared > r_squared_min
            ):
                out["depth"] = depth
                return out
        except Exception as exc:
            log.debug(
                "depth fit attempt failed for %s %s (mincut=%.3f): %s",
                aperture, band, mincut, exc,
            )

        mincut += mincut_step
        if mincut > mincut_max:
            out["status"] = "fit_failed"
            return out


# ----------------------------------------------------------------------------
# Method 2: empty-aperture sky sigma
# ----------------------------------------------------------------------------


def _circular_footprint(radius_pix: float) -> tuple[np.ndarray, int]:
    """Return ``(mask, half_size)`` for an aperture of the given radius."""
    r = float(radius_pix)
    rr = int(np.ceil(r))
    yy, xx = np.mgrid[-rr:rr + 1, -rr:rr + 1]
    mask = (xx * xx + yy * yy) <= r * r
    return mask, rr


def empty_aperture_sky_sigma(
    image_path: str,
    aperture_radius_pix: float,
    *,
    source_ra: np.ndarray | None = None,
    source_dec: np.ndarray | None = None,
    source_x: np.ndarray | None = None,
    source_y: np.ndarray | None = None,
    exclusion_radius_pix: float | None = None,
    coverage_mask: str | np.ndarray | None = None,
    n_apertures: int = 2000,
    seed: int = 42,
    sigma_clip: float = 3.0,
    max_attempts: int = 20,
) -> dict[str, float | int]:
    """Estimate the sky sigma in a circular aperture from empty regions.

    Many circular apertures are placed at random positions on the image,
    avoiding both catalog sources (via a KDTree) and any pixels flagged in
    the coverage mask. The sigma-clipped standard deviation of the summed
    aperture fluxes is the per-aperture sky noise.

    Parameters
    ----------
    image_path
        Path to the FITS science image. Read into memory once (mmap).
    aperture_radius_pix
        Aperture radius in pixels.
    source_ra, source_dec
        Sky coordinates (deg) of catalog sources to avoid. **Preferred**:
        they are converted to pixel positions using *this image's* WCS, so
        source exclusion is correct even when the science image is not on
        the detection grid (e.g. single-frame inputs). Falls back to
        ``source_x``/``source_y`` if the image has no usable WCS.
    source_x, source_y
        Pixel coordinates of catalog sources to avoid, used only when
        ``source_ra``/``source_dec`` are not given or WCS is unavailable.
        These are valid only when the source positions were measured on the
        same pixel grid as ``image_path``. If ``None`` and no sky coords are
        given, no source rejection is performed (sigma clipping still
        suppresses stars that fall inside an aperture, but exclusion is
        cleaner).
    exclusion_radius_pix
        Minimum distance from any source. Defaults to
        ``2.5 * aperture_radius_pix`` which avoids the wings of the PSF
        for typical seeing without rejecting most of the image.
    coverage_mask
        Either a FITS path or an in-memory bool/int array (1 = bad).
        Aperture centers that land on a masked pixel are rejected.
    n_apertures
        Target number of accepted apertures.
    seed
        Seed for the RNG so the function is reproducible.
    sigma_clip
        Sigma value passed to :func:`astropy.stats.sigma_clipped_stats`
        for the final summary statistic.
    max_attempts
        Maximum number of candidate-generation rounds before giving up.

    Returns
    -------
    dict
        ``{sky_sigma, n_apertures, n_attempts, mean, median, status}``.
        ``status`` is ``'ok'`` if at least ``n_apertures // 4`` apertures
        were accepted, otherwise ``'insufficient'``.
    """
    if not os.path.exists(image_path):
        log.warning("Image not found for empty-aperture depth: %s", image_path)
        return {
            "sky_sigma": float("nan"), "n_apertures": 0, "n_attempts": 0,
            "mean": float("nan"), "median": float("nan"), "status": "missing_image",
        }

    with fits.open(image_path, memmap=True) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)
        header = hdul[0].header

    ny, nx = data.shape

    if exclusion_radius_pix is None:
        exclusion_radius_pix = 2.5 * float(aperture_radius_pix)

    # Resolve source positions on THIS image's pixel grid. Prefer a WCS
    # transform of the shared sky coords; fall back to raw pixel coords only
    # when no WCS is available (same-grid assumption).
    sx = sy = None
    if source_ra is not None and source_dec is not None:
        sx, sy = _sky_to_pixel(header, source_ra, source_dec)
    if sx is None and source_x is not None and source_y is not None:
        sx = np.asarray(source_x, dtype=float)
        sy = np.asarray(source_y, dtype=float)

    tree = None
    if sx is not None and sy is not None:
        # Keep only sources that land inside the image footprint (with a
        # small margin) so the KDTree query stays meaningful.
        finite = np.isfinite(sx) & np.isfinite(sy)
        inframe = finite & (sx >= 0) & (sx < nx) & (sy >= 0) & (sy < ny)
        use = inframe if inframe.any() else finite
        if use.any():
            tree = cKDTree(np.column_stack([sx[use], sy[use]]))

    covmask: np.ndarray | None = None
    if coverage_mask is not None:
        if isinstance(coverage_mask, str):
            if os.path.exists(coverage_mask):
                with fits.open(coverage_mask, memmap=True) as h:
                    covmask = np.asarray(h[0].data).astype(bool)
            else:
                log.warning("Coverage mask not found, ignoring: %s", coverage_mask)
        else:
            covmask = np.asarray(coverage_mask).astype(bool)
        if covmask is not None and covmask.shape != data.shape:
            log.warning(
                "Coverage mask shape %s != image shape %s; ignoring mask",
                covmask.shape, data.shape,
            )
            covmask = None

    aper_mask, rr = _circular_footprint(aperture_radius_pix)
    margin = rr + 1

    rng = np.random.default_rng(seed)
    accepted_sums: list[float] = []
    n_attempts = 0
    candidate_batch = max(n_apertures * 5, 5000)

    while len(accepted_sums) < n_apertures and n_attempts < max_attempts:
        n_attempts += 1
        cx = rng.integers(margin, nx - margin, size=candidate_batch)
        cy = rng.integers(margin, ny - margin, size=candidate_batch)

        if covmask is not None:
            ok = ~covmask[cy, cx]
            cx, cy = cx[ok], cy[ok]

        if tree is not None and len(cx) > 0:
            dists, _ = tree.query(np.column_stack([cx, cy]), k=1)
            ok = dists > exclusion_radius_pix
            cx, cy = cx[ok], cy[ok]

        # Pre-extract stamps in a single pass.
        n_take = min(len(cx), n_apertures - len(accepted_sums))
        if n_take == 0:
            continue
        cx = cx[:n_take]
        cy = cy[:n_take]
        for x0, y0 in zip(cx, cy):
            stamp = data[y0 - rr:y0 + rr + 1, x0 - rr:x0 + rr + 1]
            if stamp.shape != aper_mask.shape:
                continue
            accepted_sums.append(float(stamp[aper_mask].sum()))

    if len(accepted_sums) < max(n_apertures // 4, 50):
        log.warning(
            "Empty-aperture sigma: only %d apertures accepted (target %d)",
            len(accepted_sums), n_apertures,
        )
        status = "insufficient"
    else:
        status = "ok"

    if not accepted_sums:
        return {
            "sky_sigma": float("nan"), "n_apertures": 0, "n_attempts": n_attempts,
            "mean": float("nan"), "median": float("nan"), "status": status,
        }

    sums = np.asarray(accepted_sums, dtype=np.float64)
    mean, median, sigma = sigma_clipped_stats(sums, sigma=sigma_clip, maxiters=5)
    return {
        "sky_sigma": float(sigma),
        "n_apertures": len(accepted_sums),
        "n_attempts": n_attempts,
        "mean": float(mean),
        "median": float(median),
        "status": status,
    }


def depth_from_empty_apertures(
    image_path: str,
    aperture_radius_pix: float,
    zeropoint: float,
    *,
    n_sigma: float = 5.0,
    source_ra: np.ndarray | None = None,
    source_dec: np.ndarray | None = None,
    source_x: np.ndarray | None = None,
    source_y: np.ndarray | None = None,
    exclusion_radius_pix: float | None = None,
    coverage_mask: str | np.ndarray | None = None,
    n_apertures: int = 2000,
    seed: int = 42,
) -> dict[str, float | int | str]:
    """Combine :func:`empty_aperture_sky_sigma` with the classical formula.

    Returns the empty-aperture sigma fields plus ``depth``,
    ``zeropoint``, ``aperture_radius_pix`` and ``n_sigma`` for traceability.
    Source positions are taken from ``source_ra``/``source_dec`` (converted
    via the image WCS) when available, otherwise from ``source_x``/
    ``source_y``.
    """
    stats = empty_aperture_sky_sigma(
        image_path,
        aperture_radius_pix,
        source_ra=source_ra,
        source_dec=source_dec,
        source_x=source_x,
        source_y=source_y,
        exclusion_radius_pix=exclusion_radius_pix,
        coverage_mask=coverage_mask,
        n_apertures=n_apertures,
        seed=seed,
    )
    out = dict(stats)
    out["zeropoint"] = float(zeropoint)
    out["aperture_radius_pix"] = float(aperture_radius_pix)
    out["n_sigma"] = float(n_sigma)
    out["depth"] = (
        classical_limiting_mag(n_sigma, zeropoint, stats["sky_sigma"])
        if stats["status"] != "missing_image"
        else float("nan")
    )
    return out


# ----------------------------------------------------------------------------
# Top-level orchestrator
# ----------------------------------------------------------------------------


def _aperture_radius_pix(aperture: str, pixscale_arcsec: float) -> float | None:
    """Map ``'aper05'`` to its radius in pixels, return ``None`` otherwise.

    ``aper{N}`` is treated as a diameter in arcseconds. The trailing ``'c'``
    used by spatially-corrected mag columns (``'aper05c'``) is stripped.
    """
    base = aperture[:-1] if aperture.endswith("c") else aperture
    if not base.startswith("aper"):
        return None
    try:
        diam_arcsec = float(base[4:])
    except ValueError:
        return None
    return (diam_arcsec / float(pixscale_arcsec)) / 2.0


def estimate_depths(
    cat: Table,
    *,
    bands: Sequence[str],
    apertures: Sequence[str],
    n_sigma: float = 5.0,
    pixscale_arcsec: float = 0.505,
    science_images: Mapping[str, str] | None = None,
    coverage_mask: str | None = None,
    zeropoints: Mapping[tuple[str, str], float] | None = None,
    n_empty_apertures: int = 2000,
    seed: int = 42,
    do_error_curve: bool = True,
    do_empty_apertures: bool = True,
) -> dict[tuple[str, str], dict[str, float | int | str]]:
    """Run both depth methods for every ``(band, aperture)`` combination.

    Parameters
    ----------
    cat
        Calibrated catalog from :func:`phot7ds.run_photometry`.
    bands
        Band names (as in column suffixes, e.g. ``'g'``, ``'m475'``).
    apertures
        Aperture prefixes (e.g. ``'aper05'``, ``'aper10'``). ``'auto'`` is
        only included in the error-curve method (it has no fixed radius).
    n_sigma
        Detection threshold (5 by convention).
    pixscale_arcsec
        Pixel scale, used to convert ``aper{N}`` (arcsec diameter) to pixel
        radius for the empty-aperture method.
    science_images
        Mapping ``{band: image_path}``. If provided together with
        ``zeropoints``, the empty-aperture method runs for each
        ``(band, aperture)``. Source exclusion uses each image's own WCS to
        project the catalog sky coordinates onto its native pixel grid, so
        the method is valid even when science images are not on the
        detection grid (e.g. single-frame inputs).
    coverage_mask
        Optional coverage mask path/array shared by all bands.
    zeropoints
        Mapping ``{(aperture, band): zeropoint_mag}`` produced by
        :func:`phot7ds.calibration.calibrate_zeropoints`. Required for the
        empty-aperture method.
    n_empty_apertures
        Number of empty apertures per band.
    seed
        RNG seed for the empty-aperture method.
    do_error_curve, do_empty_apertures
        Toggle each method independently.

    Returns
    -------
    dict
        ``{(aperture, band): {curve: {...}, empty: {...}}}``. ``curve``
        and ``empty`` carry the per-method result dicts; either may be
        missing when the corresponding inputs are unavailable.
    """
    results: dict[tuple[str, str], dict[str, float | int | str]] = {}

    # Sky coords are the robust source-exclusion reference: each science
    # image converts them to its own pixel grid via WCS. Pixel centroids are
    # kept only as a same-grid fallback when an image lacks a usable WCS.
    source_ra = (
        np.asarray(cat["world_centroid_alpha"], dtype=float)
        if "world_centroid_alpha" in cat.colnames else None
    )
    source_dec = (
        np.asarray(cat["world_centroid_delta"], dtype=float)
        if "world_centroid_delta" in cat.colnames else None
    )
    source_x = (
        np.asarray(cat["pixel_centroid_x"], dtype=float)
        if "pixel_centroid_x" in cat.colnames else None
    )
    source_y = (
        np.asarray(cat["pixel_centroid_y"], dtype=float)
        if "pixel_centroid_y" in cat.colnames else None
    )

    for aperture in apertures:
        for band in bands:
            key = (aperture, band)
            entry: dict[str, dict] = {}

            if do_error_curve:
                entry["curve"] = depth_from_error_curve(
                    cat, band=band, aperture=aperture, n_sigma=n_sigma,
                )

            if do_empty_apertures:
                radius_pix = _aperture_radius_pix(aperture, pixscale_arcsec)
                img_path = science_images.get(band) if science_images else None
                zp = zeropoints.get(key) if zeropoints else None
                if radius_pix is not None and img_path is not None and zp is not None:
                    entry["empty"] = depth_from_empty_apertures(
                        image_path=img_path,
                        aperture_radius_pix=radius_pix,
                        zeropoint=float(zp),
                        n_sigma=n_sigma,
                        source_ra=source_ra,
                        source_dec=source_dec,
                        source_x=source_x,
                        source_y=source_y,
                        coverage_mask=coverage_mask,
                        n_apertures=n_empty_apertures,
                        seed=seed,
                    )

            if entry:
                results[key] = entry

    return results


def format_depth_table(
    results: Mapping[tuple[str, str], Mapping[str, Mapping[str, float | int | str]]],
    n_sigma: float = 5.0,
) -> str:
    """Pretty-print a depth table for logging.

    The returned string is a single multi-line block with one row per
    ``(aperture, band)`` and the two methods side by side.
    """
    if not results:
        return "(no depth estimates)"

    lines = []
    header = (
        f"{n_sigma:.0f}-sigma depth (mag)   "
        "| curve_fit  n     R^2    | empty_aper  n     sigma_ADU"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for (aperture, band), entry in sorted(results.items()):
        curve = entry.get("curve", {}) or {}
        empty = entry.get("empty", {}) or {}
        curve_str = (
            f"{curve.get('depth', float('nan')):7.3f}"
            f"  {curve.get('n_points', 0):5d}"
            f"  {curve.get('r_squared', float('nan')):5.2f}"
        )
        empty_str = (
            f"{empty.get('depth', float('nan')):7.3f}"
            f"  {empty.get('n_apertures', 0):5d}"
            f"  {empty.get('sky_sigma', float('nan')):.4g}"
            if empty else "   ---       ---     ---"
        )
        lines.append(
            f"  {aperture:>7s} {band:<6s}        | {curve_str} | {empty_str}"
        )
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# FITS header key conventions
# ----------------------------------------------------------------------------
#
# The depth / zero-point header keys follow these patterns (all 8 chars max,
# which is the FITS standard limit; no HIERARCH cards are generated):
#
#   UL{N}EM{BAND}      N-sigma depth, error-curve fit, 5" aperture [mag]
#   UL{N}RM{BAND}      N-sigma depth, background-RMS sampling, 5" aperture [mag]
#   BRMSM{BAND}        Empty-aperture sky sigma (background RMS), 5" aperture [ADU]
#
#   ZP{APER}M{BAND}    Constant zero-point in magnitudes
#   ZE{APER}M{BAND}    1-sigma scatter of the zero-point fit (mag)
#
# Tokens:
#   {N}      digit, the SNR threshold (5 by convention)
#   {APER}   2 chars: '05' (5"), '10' (10"), 'AU' (auto), 'AC' (autoc), ...
#   {BAND}   1-3 chars: 'G','R','I','Z' for broadband; '400'..'875' for m-bands
#
# Depth quantities are emitted *only* for the canonical 5" aperture, since
# that is the survey-defined 7DS depth reference. ZP keys are emitted for
# every (aperture, band) returned by the calibration step.


_CANONICAL_DEPTH_APERTURE = "aper05"
_CANONICAL_DEPTH_APERTURE_TOKEN = "05"


def _band_keyword_token(band: str) -> str:
    """Return a FITS-keyword-friendly band suffix (1-3 chars).

    - Broadbands ``g``/``r``/``i``/``z`` -> ``G``/``R``/``I``/``Z`` (1 char).
    - Medium bands ``m400``..``m875`` -> just the digit string (3 chars).
    - Anything else -> uppercased and truncated to 3 chars.
    """
    if not band:
        return "X"
    b = band.strip()
    if len(b) == 1 and b.lower() in "griz":
        return b.upper()
    if b.lower().startswith("m"):
        digits = "".join(c for c in b[1:] if c.isdigit())
        if digits:
            return digits[:3]
    return b.upper()[:3]


def _aperture_keyword_token(aperture: str) -> str:
    """Return a 2-3 char FITS-keyword fragment for an aperture name.

    Examples
    --------
    ``aper05`` -> ``'05'``, ``aper10`` -> ``'10'``, ``auto`` -> ``'AU'``,
    ``aper05c`` -> ``'05C'``, ``autoc`` -> ``'AC'``.
    """
    if not aperture:
        return "XX"
    a = aperture.strip().lower()
    if a == "auto":
        return "AU"
    if a == "autoc":
        return "AC"
    if a.startswith("aper"):
        rest = a[4:]
        c_suffix = ""
        if rest.endswith("c"):
            c_suffix = "C"
            rest = rest[:-1]
        if rest.isdigit():
            return f"{int(rest):02d}{c_suffix}"
    return aperture.upper()[:2]


def _depth_keyword(n_sigma: int, method_letter: str, band: str) -> str:
    """Build a FITS keyword for a depth value (always 5" aperture).

    Returns ``UL{N}{E|R}M{BAND}`` -- always within the 8-character FITS
    standard for the canonical 7DS bands.
    """
    band_tok = _band_keyword_token(band)
    return f"UL{int(n_sigma)}{method_letter}M{band_tok}"


def _sky_sigma_keyword(band: str) -> str:
    """Build a FITS keyword for the background RMS (sky sigma) in ADU."""
    band_tok = _band_keyword_token(band)
    return f"BRMSM{band_tok}"


def _zp_keyword(prefix: str, aperture: str, band: str) -> str:
    """Build a FITS keyword for a zero-point quantity (``ZP`` or ``ZE``)."""
    aper_tok = _aperture_keyword_token(aperture)
    band_tok = _band_keyword_token(band)
    return f"{prefix}{aper_tok}M{band_tok}"


def depth_results_to_meta(
    meta: dict,
    results: Mapping[tuple[str, str], Mapping[str, Mapping[str, float | int | str]]],
    n_sigma: float = 5.0,
) -> None:
    """Embed depth values as FITS-safe header keys in ``meta`` (in place).

    Only the canonical 5" aperture is reported (``cfg.depth_apertures`` may
    request others, but those are kept in the run log and manifest only).
    For each ``(aper05, band)`` pair the following keys are inserted:

    - ``UL{N}EM{BAND}`` -- N-sigma depth from the magnitude-error curve fit
      [mag].
    - ``UL{N}RM{BAND}`` -- N-sigma depth from the empty-aperture / background
      RMS sampling [mag].
    - ``BRMSM{BAND}``   -- raw empty-aperture sky sigma used to compute the
      depth above [ADU per 5" aperture].

    ``N`` reflects ``n_sigma`` (rounded to the nearest integer). The card
    value is in the unit indicated by the comment; the comment carries the
    band metadata in human-readable form.
    """
    n = int(round(n_sigma))
    for (aperture, band), entry in results.items():
        if aperture != _CANONICAL_DEPTH_APERTURE:
            # Per-survey convention: only the 5" aperture appears in the
            # FITS header. Other apertures still live in the manifest /log.
            continue
        for method, marker, descr in (
            ("curve", "E", "curvefit"),
            ("empty", "R", "bkgRMS"),
        ):
            ent = entry.get(method)
            if not ent:
                continue
            depth = ent.get("depth", float("nan"))
            if not np.isfinite(depth):
                continue
            key = _depth_keyword(n, marker, band)
            meta[key] = (
                round(float(depth), 3),
                f"{n}sig {aperture} {band} {descr} depth [mag]",
            )
        # Sky sigma in ADU (raw empty-aperture noise; useful for sanity
        # checks and downstream limit-mag recomputation with an updated ZP).
        empty = entry.get("empty") or {}
        sigma = empty.get("sky_sigma", float("nan"))
        if np.isfinite(sigma):
            meta[_sky_sigma_keyword(band)] = (
                round(float(sigma), 3),
                f"sky sigma {aperture} {band} bkgRMS [ADU]",
            )


def zeropoints_to_meta(
    meta: dict,
    zeropoints: Mapping[tuple[str, str], float] | None,
    zeropoint_scatter: Mapping[tuple[str, str], float] | None = None,
) -> None:
    """Embed per-(aperture, band) zero-points as FITS-safe header keys.

    Keys are emitted as

    - ``ZP{APER}M{BAND}`` -- constant zero-point in magnitudes.
    - ``ZE{APER}M{BAND}`` -- 1-sigma scatter (calibration RMS), if provided.

    Aperture tokens follow :func:`_aperture_keyword_token` (``'05'``,
    ``'10'``, ``'AU'``, ...). Band tokens follow :func:`_band_keyword_token`
    (``'G'``/``'R'``/``'I'``/``'Z'`` for broadbands; the 3-digit central
    wavelength for medium-bands).
    """
    if not zeropoints:
        return
    scatter = zeropoint_scatter or {}
    for (aperture, band), zp in zeropoints.items():
        if zp is None or not np.isfinite(zp):
            continue
        key_zp = _zp_keyword("ZP", aperture, band)
        meta[key_zp] = (
            round(float(zp), 3),
            f"ZP {aperture} {band} [mag]",
        )
        ze = scatter.get((aperture, band))
        if ze is None or not np.isfinite(ze):
            continue
        key_ze = _zp_keyword("ZE", aperture, band)
        meta[key_ze] = (
            round(float(ze), 3),
            f"ZP scatter {aperture} {band} [mag]",
        )


__all__ = [
    "magerr_threshold_for_n_sigma",
    "classical_limiting_mag",
    "depth_from_error_curve",
    "empty_aperture_sky_sigma",
    "depth_from_empty_apertures",
    "estimate_depths",
    "format_depth_table",
    "depth_results_to_meta",
    "zeropoints_to_meta",
]
