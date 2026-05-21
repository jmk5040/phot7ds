"""
Batch driver: run :func:`phot7ds.run_photometry` over many image sets.

This is a thin loop. Each job is a dict of kwargs accepted by
:func:`run_photometry`; ``batch_run`` merges in any shared kwargs you pass at
the call site (per-job kwargs always win) and runs them sequentially. Errors
in one job don't kill the rest by default.

Example::

    from phot7ds import batch_run, PhotometryConfig

    cfg = PhotometryConfig(
        sepp_config_file="/path/7ds_sepp.config",
        detection_threshold=10.0,
    )

    results = batch_run(
        [
            dict(
                science_images=["/path/T01_g.fits", "/path/T01_r.fits", ...],
                detection_image="/path/T01_det.fits",
                reference_catalog="/path/gaia_T01.csv",
                output_dir="/path/out/T01",
                run_name="T01_run01",
            ),
            dict(
                science_images=["/path/T02_g.fits", ...],
                detection_image="/path/T02_det.fits",
                reference_catalog="/path/gaia_T02.csv",
                output_dir="/path/out/T02",
                run_name="T02_run01",
            ),
        ],
        config=cfg,                # shared default for every job
        thread_count=8,            # shared default; overridable per-job
    )

    for r in results:
        print(r.status, r.job.get("run_name"), r.error)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from .pipeline import PhotometryResult, run_photometry

log = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """One outcome from :func:`batch_run`.

    Attributes
    ----------
    job
        The (merged) job dict actually passed to :func:`run_photometry`.
    status
        ``'ok'`` if the run produced a catalog, ``'failed'`` otherwise.
    result
        Populated when ``status == 'ok'``.
    error
        Stringified exception when ``status == 'failed'``.
    """

    job: dict[str, Any]
    status: Literal["ok", "failed"]
    result: PhotometryResult | None = None
    error: str | None = None


def batch_run(
    jobs: Sequence[Mapping[str, Any]],
    *,
    on_error: Literal["raise", "continue"] = "continue",
    **shared_kwargs: Any,
) -> list[BatchResult]:
    """Run :func:`run_photometry` over a list of jobs.

    Parameters
    ----------
    jobs
        Sequence of dicts; each must contain at least
        ``science_images``, ``detection_image``, ``reference_catalog`` and
        ``output_dir``. Any other :func:`run_photometry` kwarg may also be
        set per-job.
    on_error
        ``'continue'`` (default) records the failure in
        :class:`BatchResult` and proceeds to the next job. ``'raise'``
        propagates the exception immediately.
    **shared_kwargs
        Kwargs applied to every job (e.g. ``config=cfg``, ``thread_count=8``).
        Each job's own keys override these.

    Returns
    -------
    list[BatchResult]
        One entry per input job, in input order.
    """
    results: list[BatchResult] = []
    for i, raw_job in enumerate(jobs):
        merged: dict[str, Any] = {**shared_kwargs, **dict(raw_job)}
        label = merged.get("run_name") or merged.get("detection_image") or f"job_{i}"
        log.info("batch_run [%d/%d]: %s", i + 1, len(jobs), label)
        try:
            result = run_photometry(**merged)
            results.append(BatchResult(job=merged, status="ok", result=result))
        except Exception as exc:
            if on_error == "raise":
                raise
            log.exception("batch_run job failed: %s", label)
            results.append(
                BatchResult(job=merged, status="failed", result=None, error=str(exc))
            )
    return results


__all__ = ["batch_run", "BatchResult"]
