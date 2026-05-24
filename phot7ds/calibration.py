"""
Photometric zero-point calibration against a Gaia XP synphot reference.

The pipeline:

1. Loads a Gaia XP synphot reference catalog (CSV with ``ra``, ``dec``,
   ``mag_<band>`` columns), optionally restricted to a tile polygon.
2. Nearest-neighbour matches the SE++ catalog to the reference.
3. For each band+aperture, fits a per-image 2D polynomial zero-point
   (``apply_spatial_zeropoint``) and also computes a constant ZP, applying
   both to the full target table.
"""
from __future__ import annotations

import logging
import os
from typing import Sequence

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.modeling import fitting, models
from astropy.stats import sigma_clip, sigma_clipped_stats
from astropy.table import Table

from .tile_geometry import trim_to_tile_polygon

log = logging.getLogger(__name__)

REF_PREFIX = "gaiaxp_"


def load_gaiaxp_reference(
    csv_path: str,
    bands: Sequence[str],
    *,
    tile_info=None,
    margin: float = 0.06,
) -> Table:
    """Load a Gaia XP synphot reference catalog and (optionally) trim it.

    Parameters
    ----------
    csv_path
        Path to the Gaia XP synphot CSV.
    bands
        Filter bases to load (``mag_<band>`` columns). Bands without a
        column in the CSV are silently skipped.
    tile_info
        If provided, the catalog is trimmed to this tile's polygon via
        :func:`phot7ds.tile_geometry.trim_to_tile_polygon`.
    margin
        Polygon shrink factor; passed through to
        :func:`trim_to_tile_polygon` (only used when ``tile_info`` is given).

    Returns
    -------
    Table
        Reference catalog with ``ra``, ``dec`` and a subset of ``mag_<band>``
        columns.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Gaia XP reference catalog not found: {csv_path}")

    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    band_bases = sorted({b.split("-")[0] for b in bands})
    mag_cols = [f"mag_{b}" for b in band_bases if f"mag_{b}" in header]
    usecols = ["ra", "dec"] + mag_cols

    df = pd.read_csv(csv_path, usecols=usecols)
    cat = Table.from_pandas(df)

    if tile_info is not None:
        cat = trim_to_tile_polygon(tile_info, cat, margin=margin)
        log.info("Trimmed Gaia XP reference to tile polygon: n=%d", len(cat))
    return cat


def match_nearest(
    target_ra: np.ndarray,
    target_dec: np.ndarray,
    ref_ra: np.ndarray,
    ref_dec: np.ndarray,
    *,
    match_radius_arcsec: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest-neighbour match target to reference on the sky.

    Returns
    -------
    nn_idx : ndarray[int]
        For each target row, the index of the closest reference row.
    nn_sep_arcsec : ndarray[float]
        Separation to that reference row, in arcsec.
    matched_mask : ndarray[bool]
        ``True`` where the separation is within ``match_radius_arcsec``.
    """
    incoord = SkyCoord(np.asarray(target_ra) * u.deg, np.asarray(target_dec) * u.deg)
    refcoord = SkyCoord(np.asarray(ref_ra) * u.deg, np.asarray(ref_dec) * u.deg)
    nn_idx, nn_sep, _ = incoord.match_to_catalog_sky(refcoord)
    matched_mask = nn_sep < (match_radius_arcsec * u.arcsec)
    return np.asarray(nn_idx), nn_sep.to_value(u.arcsec), np.asarray(matched_mask)


def apply_spatial_zeropoint(
    target_tbl: Table,
    ref_tbl: Table,
    *,
    x_col: str = "pixel_centroid_x",
    y_col: str = "pixel_centroid_y",
    mag_inst_col: str = "auto_mag",
    mag_ref_col: str = "mag_r",
    poly_degree: int = 2,
    new_col: str | None = "auto_mag_corrected",
) -> tuple[Table, models.Polynomial2D, float]:
    """Fit a 2-D polynomial zero-point surface to reference stars and apply it.

    Parameters
    ----------
    target_tbl
        Table containing all detections to be corrected. Modified in place if
        ``new_col`` is not ``None``.
    ref_tbl
        Calibration subset (high-SNR, flag-clean stars) with both the
        instrumental magnitude and the reference magnitude columns.
    x_col, y_col
        Pixel coordinate column names.
    mag_inst_col
        Instrumental magnitude column in ``ref_tbl`` *and* ``target_tbl``.
    mag_ref_col
        Reference magnitude column in ``ref_tbl``.
    poly_degree
        Degree of the 2D polynomial. ``2`` is usually enough.
    new_col
        If given, a new column with the spatially-corrected magnitude is
        added to ``target_tbl``. If ``None``, only the fit + RMSE are
        returned.

    Returns
    -------
    target_tbl
        Same object passed in, with optional new column.
    zp_model
        Fitted :class:`~astropy.modeling.models.Polynomial2D`.
    zp_rmse
        Standard deviation of the (sigma-clipped) residuals.
    """
    def _as_array(col) -> np.ndarray:
        arr = np.asanyarray(col)
        if np.ma.isMaskedArray(arr):
            arr = arr.filled(np.nan)
        return np.asarray(arr, dtype=float)

    raw_zp = _as_array(ref_tbl[mag_ref_col]) - _as_array(ref_tbl[mag_inst_col])
    x = _as_array(ref_tbl[x_col])
    y = _as_array(ref_tbl[y_col])

    finite = np.isfinite(raw_zp) & np.isfinite(x) & np.isfinite(y)
    clipped = sigma_clip(raw_zp[finite], sigma=3, maxiters=3)
    mask = np.zeros_like(raw_zp, dtype=bool)
    mask[finite] = ~clipped.mask
    x_clean, y_clean, zp_clean = x[mask], y[mask], raw_zp[mask]

    log.info(
        "Spatial ZP: fitting with %d/%d stars for %s",
        len(zp_clean), len(raw_zp), mag_ref_col,
    )

    zp_init = models.Polynomial2D(degree=poly_degree)
    fitter = fitting.LinearLSQFitter()
    zp_model = fitter(zp_init, x_clean, y_clean, zp_clean)

    residuals = zp_clean - zp_model(x_clean, y_clean)
    residuals = residuals[~sigma_clip(residuals, sigma=2.5, maxiters=3).mask]
    zp_rmse = float(np.std(residuals))
    log.info("Spatial ZP RMSE = %.4f mag", zp_rmse)

    if new_col is not None:
        spatial_zp = zp_model(_as_array(target_tbl[x_col]), _as_array(target_tbl[y_col]))
        target_tbl[new_col] = target_tbl[mag_inst_col] + spatial_zp

    return target_tbl, zp_model, zp_rmse


def calibrate_zeropoints(
    cat: Table,
    *,
    ref_cat: Table,
    band_names: Sequence[str],
    apertures: Sequence[str],
    match_radius_arcsec: float = 1.0,
    mag_range: tuple[float, float] = (12.0, 16.0),
    spatial_poly_degree: int = 2,
    plot_residuals: bool = False,
    plot_dir: str | None = None,
    plot_title_extra: str = "",
) -> Table:
    """End-to-end per-band, per-aperture ZP calibration in-place on ``cat``.

    For each ``(band, aperture)``:

    1. Build a clean calibration subset: position-matched within
       ``match_radius_arcsec``, ``source_flags == 0``,
       per-band per-aperture ``flags == 0``, reference magnitude in
       ``mag_range``.
    2. Fit a 2-D spatial ZP and store the corrected column
       ``{aperture}c_mag_{band}`` on the full catalog.
    3. Compute a constant ZP via :func:`astropy.stats.sigma_clipped_stats`
       and apply it to ``{aperture}_mag_{band}``.
    4. Propagate the spatial-ZP RMSE / constant-ZP scatter into
       ``*_mag_err_*`` columns.

    Optionally produces a diagnostic figure per band+aperture via
    :func:`phot7ds.diagnostics.plot_phot_residual_map` when
    ``plot_residuals=True``.

    Returns ``cat`` for chaining.
    """
    if plot_residuals:
        from .diagnostics import plot_phot_residual_map
        if plot_dir is not None:
            os.makedirs(plot_dir, exist_ok=True)

    incoord_ra = np.asarray(cat["world_centroid_alpha"])
    incoord_dec = np.asarray(cat["world_centroid_delta"])
    refcoord_ra = np.asarray(ref_cat["ra"])
    refcoord_dec = np.asarray(ref_cat["dec"])
    nn_idx, _, matched_mask = match_nearest(
        incoord_ra, incoord_dec, refcoord_ra, refcoord_dec,
        match_radius_arcsec=match_radius_arcsec,
    )
    clean_mask = matched_mask & (np.asarray(cat["source_flags"]) == 0)

    for band in band_names:
        band_ref = band.split("-")[0]
        gaia_mag_col = f"mag_{band_ref}"
        if gaia_mag_col not in ref_cat.colnames:
            log.warning("Missing reference column %s, skipping band %s", gaia_mag_col, band)
            continue

        gaia_mag_all = np.full(len(cat), np.nan, dtype=float)
        if np.any(matched_mask):
            gaia_mag_all[matched_mask] = np.asarray(ref_cat[gaia_mag_col])[
                nn_idx[matched_mask]
            ]

        for aperture in apertures:
            flag_col = f"{aperture}_flags_{band}"
            if flag_col not in cat.colnames:
                log.warning("Missing %s; skipping calibration", flag_col)
                continue
            zmask = clean_mask.copy()
            zmask &= np.asarray(cat[flag_col]) == 0
            zmask &= np.isfinite(gaia_mag_all)
            zmask &= gaia_mag_all > mag_range[0]
            zmask &= gaia_mag_all < mag_range[1]
            if not np.any(zmask):
                log.info("No calibration sources for band=%s aper=%s", band, aperture)
                continue

            ztbl = cat[zmask].copy()
            ztbl[f"{REF_PREFIX}mag_{band_ref}"] = gaia_mag_all[zmask]

            ztbl, zp_model, zp_rmse = apply_spatial_zeropoint(
                target_tbl=ztbl,
                ref_tbl=ztbl,
                x_col="pixel_centroid_x",
                y_col="pixel_centroid_y",
                mag_inst_col=f"{aperture}_mag_{band}",
                mag_ref_col=f"{REF_PREFIX}mag_{band_ref}",
                poly_degree=spatial_poly_degree,
                new_col=f"{aperture}c_mag_{band}",
            )

            mag_diff = (
                ztbl[f"{REF_PREFIX}mag_{band_ref}"] - ztbl[f"{aperture}_mag_{band}"]
            )
            _, zp, zperr = sigma_clipped_stats(mag_diff, sigma=2.0, maxiters=5)
            ztbl[f"{aperture}_mag_{band}"] += zp
            ztbl[f"{aperture}_mag_err_{band}"] = np.hypot(
                ztbl[f"{aperture}_mag_err_{band}"], zperr
            )

            # Record the constant ZP and its scatter so downstream code
            # (e.g. depth estimation) can convert ADU back to magnitudes.
            cat.meta.setdefault("zeropoints", {})[(aperture, band)] = float(zp)
            cat.meta.setdefault("zeropoint_scatter", {})[(aperture, band)] = float(
                zperr
            )

            if plot_residuals:
                base = f"{plot_title_extra}_{band}".strip("_")
                for tag in (aperture, f"{aperture}c"):
                    savefig = (
                        os.path.join(plot_dir, f"{base}_{tag}_zp.png")
                        if plot_dir is not None else None
                    )
                    plot_phot_residual_map(
                        ztbl,
                        band=band,
                        gaia_mag_col=f"{REF_PREFIX}mag_{band_ref}",
                        axiscolor="elongation",
                        mag_type=tag,
                        mag_range=mag_range,
                        sigma=2.0,
                        dmag_lim=0.10,
                        additional_title=plot_title_extra,
                        savefigname=savefig,
                    )

            # Apply both corrections to the full catalog.
            spatial_zp_full = zp_model(
                np.asarray(cat["pixel_centroid_x"], dtype=float),
                np.asarray(cat["pixel_centroid_y"], dtype=float),
            )
            cat[f"{aperture}c_mag_{band}"] = (
                cat[f"{aperture}_mag_{band}"] + spatial_zp_full
            )
            cat[f"{aperture}c_mag_err_{band}"] = np.hypot(
                cat[f"{aperture}_mag_err_{band}"], zp_rmse
            )
            cat[f"{aperture}_mag_{band}"] += zp
            cat[f"{aperture}_mag_err_{band}"] = np.hypot(
                cat[f"{aperture}_mag_err_{band}"], zp_rmse
            )
    return cat


__all__ = [
    "REF_PREFIX",
    "load_gaiaxp_reference",
    "match_nearest",
    "apply_spatial_zeropoint",
    "calibrate_zeropoints",
]
