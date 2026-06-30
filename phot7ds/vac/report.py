"""
Human-readable run log for a value-added catalog run.

:func:`write_run_log` collects the cross-matching, flux, photo-z and
SED-fitting metadata produced by the pipeline into a single plain-text log
written next to the value-added catalog.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from .config import VACConfig

log = logging.getLogger(__name__)


def _section(title: str) -> str:
    return f"\n{'-' * 70}\n{title}\n{'-' * 70}\n"


def _kv(key: str, value) -> str:
    return f"  {key:<22}: {value}\n"


def write_run_log(
    cfg: VACConfig,
    tile: str,
    log_path: str | Path,
    *,
    match_info: dict | None = None,
    flux_info: dict | None = None,
    eazy_info: dict | None = None,
    fastpp_info: dict | None = None,
    extra: dict | None = None,
) -> str:
    """Write a value-added run log and return its path."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("=" * 70 + "\n")
    lines.append(f"phot7ds.vac value-added catalog run log\n")
    lines.append("=" * 70 + "\n")
    lines.append(_kv("tile", tile))
    lines.append(_kv("detection_ref", cfg.detection_ref))
    lines.append(_kv("timestamp (UTC)", datetime.now(timezone.utc).isoformat(timespec="seconds")))
    lines.append(_kv("aperture", cfg.aperture))

    lines.append(_section("Cross-match"))
    lines.append(_kv("auto_download", cfg.auto_download))
    lines.append(_kv("match_radius_arcsec", cfg.match_radius_arcsec))
    lines.append(_kv("use_vhs / use_galex / use_wise",
                     f"{cfg.use_vhs} / {cfg.use_galex} / {cfg.use_wise}"))
    if match_info:
        for k, v in match_info.items():
            lines.append(_kv(k, v))

    if flux_info:
        lines.append(_section("Filters & flux catalog"))
        filters = flux_info.get("filters", [])
        lines.append(_kv("n_filters", len(filters)))
        lines.append(_kv("n_input_sources", flux_info.get("n_input")))
        lines.append(_kv("n_pass_coverage_cut", flux_info.get("n_pass")))
        lines.append(_kv("min_filter_fraction", flux_info.get("min_filter_fraction")))
        lines.append(_kv("error_margin", flux_info.get("error_margin")))
        lam = flux_info.get("lambda_c", {})
        ext = flux_info.get("extinction", {})
        lines.append("\n  filters (name, lambda_c [AA], A_lambda [mag]):\n")
        for f in filters:
            lines.append(
                f"    {f:<14} lambda_c={lam.get(f, float('nan')):>9.1f}  "
                f"A={ext.get(f, float('nan')):>6.3f}\n"
            )

    if eazy_info:
        engine = eazy_info.get("engine", "eazy-py")
        lines.append(_section(f"EAzY ({engine}) photo-z"))
        lines.append(_kv("engine", engine))
        if eazy_info.get("binary"):
            lines.append(_kv("binary", eazy_info.get("binary")))
        lines.append(_kv("apply_prior", eazy_info.get("apply_prior")))
        lines.append(_kv("prior_band", eazy_info.get("prior_band")))
        lines.append(_kv("prior_filter", eazy_info.get("prior_filter")))
        lines.append(_kv("prior_filter_id", eazy_info.get("prior_filter_id")))
        lines.append(_kv("prior_file", eazy_info.get("prior_file")))
        lines.append(_kv("redshift_column", eazy_info.get("zphot_column")))
        lines.append(_kv("n_targets", eazy_info.get("n_targets")))
        if eazy_info.get("run_log"):
            lines.append(_kv("run_log", eazy_info.get("run_log")))
        if "grid_n_proc" in eazy_info:
            lines.append(_kv("template-grid n_proc", eazy_info.get("grid_n_proc")))
        if "fit_n_proc" in eazy_info:
            lines.append(_kv("fit n_proc", eazy_info.get("fit_n_proc")))
        params = eazy_info.get("params", {})
        if params:
            lines.append("\n  eazy parameters:\n")
            for k in sorted(params):
                lines.append(f"    {k:<20} = {params[k]}\n")

    if fastpp_info:
        lines.append(_section("FAST++ SED fitting"))
        lines.append(_kv("binary", fastpp_info.get("binary")))
        lines.append(_kv("name_zphot", fastpp_info.get("name_zphot")))
        lines.append(_kv("n_fits", fastpp_info.get("n_fits")))
        if fastpp_info.get("run_log"):
            lines.append(_kv("run_log", fastpp_info.get("run_log")))
        params = fastpp_info.get("params", {})
        if params:
            lines.append("\n  fast++ parameters:\n")
            for k in sorted(params):
                lines.append(f"    {k:<20} = {params[k]}\n")

    if extra:
        lines.append(_section("Outputs"))
        for k, v in extra.items():
            lines.append(_kv(k, v))

    with open(log_path, "w") as fh:
        fh.writelines(lines)
    log.info("Wrote VAC run log: %s", log_path)
    return str(log_path)


__all__ = ["write_run_log"]
