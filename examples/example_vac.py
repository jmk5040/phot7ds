"""
End-to-end example: build a value-added catalog with ``phot7ds.vac``.

Takes a phot7ds photometric catalog for one tile and produces a
value-added catalog by:

1. cross-matching galaxies against REGALADE (+ optional VHS / GALEX),
2. assembling an extinction-corrected flux input catalog,
3. running eazy-py for photometric redshifts,
4. running FAST++ for SED fitting,
5. merging everything into ``{tile}_{ref}_value_added.fits``.

This needs the optional ``vac`` extra and the FAST++ binary::

    pip install -e ".[vac]"
    python examples/example_vac.py

Edit the constants below for your system. Paths are resolved relative to
this file where possible; the EAzY/FAST++ ``LIB`` tree and external
reference catalogs must be supplied by the user.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Allow ``python examples/example_vac.py`` without a prior pip install.
_HERE = Path(__file__).resolve().parent
_PHOT7DS_ROOT = _HERE.parent
if str(_PHOT7DS_ROOT) not in sys.path:
    sys.path.insert(0, str(_PHOT7DS_ROOT))

from phot7ds.vac import VACConfig, run_value_added  # noqa: E402

# --- user settings ------------------------------------------------------
TILE = "T00236"
DETECTION_REF = "7DS"  # tag used in the catalog file names

# phot7ds photometric catalog for the tile (FITS).
CATALOG_PATH = f"/data/data1/7DS/RIS/catalog/7ds/{TILE}/{TILE}_{DETECTION_REF}_phot.fits"

# Tile-definition table (polygon corners + center).
TILE_TABLE = "/data/data1/7DS/RIS/config/7DT_tiles.fits"

# EAzY/FAST++ config tree and external reference catalogs.
LIB_DIR = "/data/data1/7DS/RIS/config/LIB"
CATALOG_DIR = "/data/data1/7DS/RIS/catalog"
OUTPUT_ROOT = "/data/data1/7DS/RIS/results/vac"
FASTPP_BIN = "/home/jmkastro/fastpp/bin/fast++"
# Auto-download missing external catalogs (REGALADE/VHS/GALEX) via VizieR.
AUTO_DOWNLOAD = True
# Toggle the heavy stages.
DO_PHOTOZ = True
DO_SEDFIT = True
# -----------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    config = VACConfig(
        lib_dir=LIB_DIR,
        catalog_dir=CATALOG_DIR,
        output_root=OUTPUT_ROOT,
        fastpp_bin=FASTPP_BIN,
        detection_ref=DETECTION_REF,
        aperture="aper05c",
        # The 7DS filters that enter the fit are auto-detected from the
        # catalog's <aperture>_mag_* columns (no use_medium/use_broad needed).
        use_vhs=True,
        use_galex=True,
        use_wise=True,
        # Fetch absent REGALADE/VHS/GALEX catalogs on demand.
        auto_download=AUTO_DOWNLOAD,
        # Default magnitude prior: 7DS m625 (prior_m6250_extend.dat). When
        # aper05c_mag_m625 is absent the run proceeds without a prior.
        prior_band="m625",
        n_proc=8,
    )

    result = run_value_added(
        catalog_path=CATALOG_PATH,
        tile=TILE,
        tile_table=TILE_TABLE,
        config=config,
        do_photoz=DO_PHOTOZ,
        do_sedfit=DO_SEDFIT,
    )

    print("\n=== value-added catalog summary ===")
    print(f"tile         : {result.tile}")
    print(f"matched gals : {result.n_matched}")
    print(f"flux sources : {result.n_flux}")
    print(f"photo-z done : {result.photoz_done}")
    print(f"SED fit done : {result.sedfit_done}")
    print(f"output       : {result.value_added_path}")
    print(f"run log      : {result.log_path}")


if __name__ == "__main__":
    main()
