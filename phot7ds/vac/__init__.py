"""
phot7ds.vac - value-added catalog pipeline.

Turns a phot7ds photometric catalog into a value-added catalog:

1. Cross-match galaxies against REGALADE (+ optional VHS / GALEX).
2. Assemble an extinction-corrected flux input catalog.
3. Run eazy-py for photometric redshifts.
4. Run FAST++ for SED fitting (stellar mass, SFR, ...).
5. Merge the results back onto the original catalog.

This subpackage needs optional heavy dependencies (``eazy``, ``sfdmap``,
``extinction``) plus the compiled ``fast++`` binary. Install the extras
with::

    pip install "phot7ds[vac]"

The heavy imports are deferred to call time so ``import phot7ds`` keeps
working without them; a clear error is raised if they are missing when a
stage that needs them runs.
"""
from __future__ import annotations

from .config import VACConfig

# Lightweight, always-importable pieces.
from .crossmatch import build_galaxy_catalog
from .fluxes import build_flux_catalog

# ``pipeline`` imports photoz/sedfit lazily, so it is safe to expose here.
from .pipeline import VACResult, run_value_added

__all__ = [
    "VACConfig",
    "VACResult",
    "run_value_added",
    "build_galaxy_catalog",
    "build_flux_catalog",
]
