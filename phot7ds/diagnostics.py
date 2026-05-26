"""
Diagnostic plots for the photometric calibration step.

Only :func:`plot_phot_residual_map` is currently exposed; it visualises the
2D zero-point residuals on the focal plane plus a ZP-vs-magnitude scatter.
"""
from __future__ import annotations

import gc
import logging
from typing import Literal

import numpy as np

log = logging.getLogger(__name__)


def plot_phot_residual_map(
    mtbl,
    band: str,
    gaia_mag_col: str,
    *,
    mag_type: str = "auto",
    mag_range: tuple[float, float] = (12, 17),
    sigma: float = 3.0,
    dmag_lim: float = 0.15,
    axiscolor: Literal["elongation", "fwhm"] = "elongation",
    additional_title: str = "",
    savefigname: str | None = None,
    pixscale_arcsec: float = 0.505,
) -> None:
    """Plot the spatial residual map of (reference - instrumental) magnitudes.

    Parameters
    ----------
    mtbl
        Calibration table with at minimum ``pixel_centroid_x/y``,
        ``{mag_type}_mag_{band}``, ``{mag_type}_mag_err_{band}``,
        ``elongation`` and ``gaia_mag_col``.
    band
        Band suffix used in the magnitude column names.
    gaia_mag_col
        Reference magnitude column (e.g. ``gaiaxp_mag_r``).
    mag_type
        Magnitude column family: ``'auto'``, ``'aper05'``, ``'autoc'``, ...
    mag_range
        Reference-magnitude window used to select calibration stars.
    sigma
        Sigma clipping threshold applied to the residuals before plotting.
    dmag_lim
        Colour-axis half-range for the residual scatter.
    axiscolor
        Second colorbar source: ``'elongation'`` or ``'fwhm'`` (derived from
        ``flux_rad_50_<band>`` times ``pixscale_arcsec``).
    additional_title
        Extra text appended to the title.
    savefigname
        If given, save the figure here; otherwise call ``plt.show()``.
    pixscale_arcsec
        Pixel scale, used only when ``axiscolor='fwhm'`` for axis units.
    """
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    from astropy.stats import sigma_clip
    from matplotlib.gridspec import GridSpec
    from scipy.stats import binned_statistic

    mag_col = f"{mag_type}_mag_{band}"
    if mag_type.endswith("c"):
        magerr_col = f"{mag_type[:-1]}_mag_err_{band}"
    else:
        magerr_col = f"{mag_type}_mag_err_{band}"

    x = np.asarray(mtbl["pixel_centroid_x"], dtype=np.float32)
    y = np.asarray(mtbl["pixel_centroid_y"], dtype=np.float32)
    gmag = np.asarray(mtbl[gaia_mag_col], dtype=np.float32)
    dmag = np.asarray(mtbl[mag_col] - mtbl[gaia_mag_col], dtype=np.float32)

    for cand in (
        f"{mag_type}_magerr_{band}",
        f"{mag_type}_mag_err_{band}",
        f"{mag_type}_err_{band}",
    ):
        if cand in mtbl.colnames:
            magerr_col = cand
            break
    gaiaerr_col = None
    for cand in (f"{gaia_mag_col}_err", f"{gaia_mag_col}_error"):
        if cand in mtbl.colnames:
            gaiaerr_col = cand
            break

    if magerr_col in mtbl.colnames:
        dmag_err = np.asarray(mtbl[magerr_col], dtype=np.float32)
        if gaiaerr_col is not None:
            dmag_err = np.sqrt(
                dmag_err**2 + np.asarray(mtbl[gaiaerr_col], dtype=np.float32) ** 2
            ).astype(np.float32)
    else:
        dmag_err = None

    if axiscolor == "fwhm":
        try:
            axprm = np.asarray(
                mtbl[f"flux_rad_50_{band}"] * pixscale_arcsec, dtype=np.float32
            )
        except KeyError:
            log.warning("flux_rad_50_%s missing; falling back to elongation", band)
            axprm = np.asarray(mtbl["elongation"], dtype=np.float32)
            axiscolor = "elongation"
    elif axiscolor == "elongation":
        axprm = np.asarray(mtbl["elongation"], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported axiscolor: {axiscolor!r}")

    msel = (
        (gmag > mag_range[0])
        & (gmag < mag_range[1])
        & np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(dmag)
        & np.isfinite(axprm)
    )
    if dmag_err is not None:
        msel &= np.isfinite(dmag_err)
    if not np.any(msel):
        return
    x, y, gmag, dmag, axprm = x[msel], y[msel], gmag[msel], dmag[msel], axprm[msel]
    if dmag_err is not None:
        dmag_err = dmag_err[msel]

    mask = ~sigma_clip(dmag, sigma=sigma).mask
    if not np.any(mask):
        return
    exc_x, exc_y, exc_dmag = x[~mask], y[~mask], dmag[~mask]
    x, y, gmag, dmag, axprm = x[mask], y[mask], gmag[mask], dmag[mask], axprm[mask]
    if dmag_err is not None:
        dmag_err = dmag_err[mask]
    if len(dmag) < 3:
        return

    mag_rmse = float(np.sqrt(np.mean(dmag**2)))
    zp_median = float(np.median(dmag))
    zp_sigma = float(1.4826 * np.median(np.abs(dmag - zp_median)))

    px = np.polyfit(x.astype(np.float64), dmag.astype(np.float64), 1)
    py = np.polyfit(y.astype(np.float64), dmag.astype(np.float64), 1)

    def _bin_stats(coord: np.ndarray, nbin: int):
        if coord.size == 0:
            empty = np.array([])
            return empty, empty, empty
        bins = np.linspace(coord.min(), coord.max(), nbin + 1)
        cen = 0.5 * (bins[:-1] + bins[1:])
        med, _, _ = binned_statistic(coord, dmag, statistic="median", bins=bins)
        mad, _, _ = binned_statistic(
            coord,
            dmag,
            statistic=lambda v: 1.4826 * np.median(np.abs(v - np.median(v))),
            bins=bins,
        )
        return cen, med, mad

    xcen, xmed, xmad = _bin_stats(x, nbin=5)
    ycen, ymed, ymad = _bin_stats(y, nbin=3)

    fig = None
    try:
        fig = plt.figure(figsize=(8, 7.5))
        gs = GridSpec(3, 3, width_ratios=[4, 1, 0.1], height_ratios=[3, 1, 1.2])
        ax_xy = fig.add_subplot(gs[0, 0])
        ax_dy = fig.add_subplot(gs[0, 1], sharey=ax_xy)
        ax_dx = fig.add_subplot(gs[1, 0], sharex=ax_xy)
        ax_zp = fig.add_subplot(gs[2, 0:2])
        cax_dmag = fig.add_subplot(gs[0:1, 2])
        cax_axprm = fig.add_subplot(gs[1:3, 2])

        sc_dmag = ax_xy.scatter(
            x, y, c=dmag, s=20, edgecolors="black", linewidths=0.1,
            cmap=cm.coolwarm, vmin=-dmag_lim, vmax=dmag_lim,
            label=f"Reference stars (n={len(dmag)})",
        )
        ax_xy.scatter(
            exc_x, exc_y, s=20, marker="x", color="k", alpha=0.5, linewidths=0.8,
            label=f"{sigma}σ clipped (n={len(exc_dmag)})",
        )
        ax_xy.legend(loc="upper right", fontsize="large")
        ax_xy.set_ylabel("Y [pixel]")
        ax_xy.invert_yaxis()
        title = (
            f"Photometry Residual Map {additional_title}\n"
            f"{mag_type.split('_')[0]} | "
            f"{mag_range[0]} < {band} < {mag_range[1]} | RMSE={mag_rmse:.3f}"
        )
        ax_xy.set_title(title)

        axprm_min = float(np.quantile(axprm, 0.05))
        axprm_max = float(np.quantile(axprm, 0.95))
        if axiscolor == "fwhm":
            axprm_min = min(2.0, axprm_min)
            axprm_max = max(4.0, axprm_max)

        sc_axprm = ax_dx.scatter(
            x, dmag, s=12, c=axprm, alpha=1, edgecolors="black", linewidths=0.1,
            cmap=cm.plasma, vmin=axprm_min, vmax=axprm_max,
        )
        ax_dx.errorbar(
            xcen, xmed, yerr=xmad, fmt="s",
            markerfacecolor="white", markeredgecolor="black",
            markeredgewidth=1.2, color="k", capsize=3,
        )
        xx = np.linspace(x.min(), x.max(), 200)
        ax_dx.plot(
            xx, px[0] * xx + px[1], "r", lw=2,
            label=rf"$\Delta\mathrm{{mag}} = {px[0]:.2e}x {px[1]:+.3f}$",
        )
        ax_dx.set_ylim(-dmag_lim, dmag_lim)
        plt.setp(ax_dx.get_xticklabels(), visible=False)
        ax_dx.set_xlabel("X [pixel]", labelpad=5)
        ax_dx.set_ylabel(r"$\Delta\mathrm{mag}$")
        ax_dx.legend(loc="upper right", fontsize="medium")
        ax_dx.grid(True, alpha=0.5, ls="--")

        ax_dy.scatter(
            dmag, y, s=12, c=axprm, alpha=1, edgecolors="black", linewidths=0.1,
            cmap=cm.plasma, vmin=axprm_min, vmax=axprm_max,
        )
        ax_dy.errorbar(
            ymed, ycen, xerr=ymad, fmt="s",
            markerfacecolor="white", markeredgecolor="black",
            markeredgewidth=1.2, color="k", capsize=3,
        )
        yy = np.linspace(y.min(), y.max(), 200)
        ax_dy.plot(py[0] * yy + py[1], yy, "r", lw=2)
        ax_dy.set_title(
            rf"$\Delta\mathrm{{mag}}=$" + "\n" + rf"${py[0]:.2e}y {py[1]:+.3f}$",
            fontsize="small",
        )
        plt.setp(ax_dy.get_yticklabels(), visible=False)
        ax_dy.set_xlim(-dmag_lim, dmag_lim)
        ax_dy.set_xlabel(r"$\Delta\mathrm{mag}$")
        ax_dy.invert_yaxis()
        ax_dy.grid(True, alpha=0.5, ls="--")

        if dmag_err is not None:
            ax_zp.errorbar(
                gmag, dmag, yerr=dmag_err, fmt="none", ecolor="k", alpha=0.3,
                elinewidth=0.3, capsize=2, zorder=0,
            )
        ax_zp.scatter(
            gmag, dmag, s=12, c=axprm, alpha=1, edgecolors="black", linewidths=0.3,
            cmap=cm.plasma, vmin=axprm_min, vmax=axprm_max,
        )
        ax_zp.axhline(
            zp_median, color="crimson", lw=1.8,
            label=f"Zero-point = {zp_median:+.3f} mag ",
        )
        ax_zp.axhspan(
            zp_median - zp_sigma, zp_median + zp_sigma,
            color="crimson", alpha=0.18, label=f"1σ = {zp_sigma:.3f} mag",
        )
        ax_zp.set_xlim(mag_range[0], mag_range[1])
        ax_zp.set_ylim(-0.15, 0.15)
        ax_zp.set_xlabel("Reference Magnitude", labelpad=4)
        ax_zp.set_ylabel("ZP")
        ax_zp.grid(True, alpha=0.5, ls="--")
        ax_zp.legend(loc="upper left", fontsize="small")

        cb1 = plt.colorbar(sc_dmag, cax=cax_dmag)
        cb1.set_label(r"$\Delta\mathrm{mag}$")
        cb2 = plt.colorbar(sc_axprm, cax=cax_axprm)
        cb2.set_label("FWHM [arcsec]" if axiscolor == "fwhm" else "Elongation")
        fig.subplots_adjust(
            left=0.07, right=0.92, bottom=0.08, top=0.92, wspace=0.08, hspace=0.18
        )
        if savefigname is not None:
            fig.savefig(savefigname, dpi=250)
        else:
            plt.show()
    finally:
        if fig is not None:
            fig.clf()
            plt.close(fig)
        gc.collect()


__all__ = ["plot_phot_residual_map"]
