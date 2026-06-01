"""
End-to-end example: DELVE detection (optional) + ``run_photometry``.

This script is intentionally self-contained:

* All paths are resolved **relative to this file** so the example works
  from any checkout location (no hard-coded absolute paths).
* SourceExtractor++ and SWarp config files are auto-generated on the
  first run via :func:`phot7ds.ensure_sepp_config` /
  :func:`phot7ds.ensure_swarp_config`. Existing files are left alone.
* The tile table and Gaia XP reference catalog *must* be supplied by
  the user. Missing files raise a clear :class:`FileNotFoundError` with
  the expected layout.

Edit the constants below for your system if your tiles, reference
catalogs or DELVE cache live elsewhere, then run::

    pip install -e .                  # or: export PYTHONPATH=$PWD:$PYTHONPATH
    python examples/example_run.py

See ``examples/config/README.md`` for the expected directory layout
and ``examples/config/column_convention.md`` for the output column
cheat-sheet.
"""

from __future__ import annotations

import glob
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from astropy.io import fits
from astropy.table import Table

# Allow ``python examples/example_run.py`` without a prior pip install.
_HERE = Path(__file__).resolve().parent
_PHOT7DS_ROOT = _HERE.parent
if str(_PHOT7DS_ROOT) not in sys.path:
    sys.path.insert(0, str(_PHOT7DS_ROOT))

from phot7ds import (  # noqa: E402  (imported after sys.path tweak above)
    PhotometryConfig,
    batch_run,
    ensure_sepp_config,
    ensure_swarp_config,
    require_gaiaxp_reference,
    require_tile_table,
    run_photometry,
)
from phot7ds.detection import build_delve_detection_image  # noqa: E402

# --- Repo-relative paths (override on demand) ------------------------------
CONFIG_DIR = _HERE / "config"
SEPP_CONFIG = CONFIG_DIR / "7ds_sepp.config"   # auto-created if missing
SWARP_CONFIG = CONFIG_DIR / "7ds.swarp"        # auto-created if missing
TILE_TABLE = CONFIG_DIR / "7DT_tiles.ascii"    # USER-supplied, must exist
REFERENCE_DIR = CONFIG_DIR / "gaiaxp"          # USER-supplied Gaia XP CSVs
DETECT_IMG_DIR = CONFIG_DIR / "DELVE"          # output dir for DELVE mosaics
OUTPUT_DIR = _HERE / "example_run"             # photometry outputs

# Set True to rebuild DELVE even when a cached coadd already exists.
FORCE_BUILD_DELVE = False

# Set True to rebuild the catalog even when a cached one already exists.
FORCE_BUILD_CATALOG = True

# Set the detection threshold for the DELVE detection image (otherwise using default configuration based on the detection label)
DETECTION_THRESHOLD = 10.0

# Worker counts: for photometry and DELVE SIA downloads
NCORES = 12

# Bootstrap the SE++ / SWarp configs on the first run.
SEPP_CONFIG = ensure_sepp_config(SEPP_CONFIG)
SWARP_CONFIG = ensure_swarp_config(SWARP_CONFIG)

# Required science images. The OBJECT / OBJCTRA / OBJCTDEC FITS headers
# identify the tile (and supply the SWarp -CENTER for DELVE).
science_images = [
    "/data/data1/processed_1x1_gain2750/T06910/7DT14/m625/calib_7DT14_T06910_20250620_024540_m625_300.com.fits",
    "/data/data1/processed_1x1_gain2750/T06910/7DT06/m575/calib_7DT06_T06910_20250620_024534_m575_300.com.fits",
    "/data/data1/processed_1x1_gain2750/T06910/7DT10/m825/calib_7DT10_T06910_20250620_024533_m825_300.com.fits",
]
for path in science_images:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Science image not found: {path}")
tile = fits.getheader(science_images[0])["OBJECT"]


def _load_tile_table(path: Path) -> Table:
    """Read the tile table tolerating non-ASCII content.

    The shipped ``7DT_tiles.ascii`` sometimes contains non-UTF-8 bytes,
    in which case astropy's ASCII reader raises
    :class:`UnicodeDecodeError`. If a FITS sibling exists we fall back
    to it; otherwise we try Latin-1 explicitly.
    """
    require_tile_table(path)
    try:
        return Table.read(path, format="ascii")
    except UnicodeDecodeError:
        sibling = path.with_suffix(".fits")
        if sibling.exists():
            return Table.read(sibling, format="fits")
        with open(path, encoding="latin-1") as fh:
            return Table.read(fh, format="ascii")


# --- DELVE detection image / mask -----------------------------------------
detection_image = f"{DETECT_IMG_DIR}/{tile}/{tile}_DELVE_DR3_IMAGE_det.fits"
detection_mask = f"{DETECT_IMG_DIR}/{tile}/{tile}_DELVE_DR3_MASK_det.fits"

if (
    os.path.exists(detection_image)
    and os.path.exists(detection_mask)
    and not FORCE_BUILD_DELVE
):
    print(f"Using existing DELVE detection image: {detection_image}")
    print(f"Using existing DELVE detection mask : {detection_mask}")
else:
    print(f"Building DELVE detection image: {detection_image}")
    print(f"Building DELVE detection mask : {detection_mask}")
    logging.basicConfig(level=logging.INFO)
    tile_tbl = _load_tile_table(TILE_TABLE)
    tile_info = tile_tbl[tile_tbl["tile"] == tile]
    if len(tile_info) == 0:
        raise ValueError(f"No row for tile {tile!r} in {TILE_TABLE}")
    detection_image, _det_weight = build_delve_detection_image(
        tile_info=tile_info,
        imgtype="image",
        output_path=f"{DETECT_IMG_DIR}/{tile}",
        swarp_cfg_path=str(SWARP_CONFIG),
        ncores=NCORES,
        max_retries=5,
    )
    detection_mask, _mask_weight = build_delve_detection_image(
        tile_info=tile_info,
        imgtype="mask",
        output_path=f"{DETECT_IMG_DIR}/{tile}",
        swarp_cfg_path=str(SWARP_CONFIG),
        ncores=NCORES,
        max_retries=5,
    )
    print(f"Built DELVE detection image: {detection_image}")
    print(f"Built DELVE detection mask : {detection_mask}")

detection_tag = "DELVE" if "DELVE" in os.path.basename(detection_image) else "7DT"

# --- Reference catalog ----------------------------------------------------
reference_catalog = require_gaiaxp_reference(REFERENCE_DIR, tile=tile)

# Output catalog name: leaf only; saved under OUTPUT_DIR (created if missing).
# Whatever you pass here is preserved verbatim (no `_phot.zp.fits` rename).
catalog_name = "test_zp.fits"

EXAMPLE = {
    "sepp_config_file": str(SEPP_CONFIG),
    "reference_catalog": str(reference_catalog),
    "detection_image": detection_image,
    "badpix_mask": detection_mask,
    "science_images": science_images,
    "catalog_name": catalog_name,
    "output_dir": str(OUTPUT_DIR),
    "thread_count": NCORES,
}


def run_single() -> None:
    """Pass everything inline as keyword arguments."""
    result = run_photometry(
        science_images=EXAMPLE["science_images"],
        detection_image=EXAMPLE["detection_image"],
        reference_catalog=EXAMPLE["reference_catalog"],
        output_dir=EXAMPLE["output_dir"],
        catalog_name=EXAMPLE["catalog_name"],
        badpix_mask=EXAMPLE["badpix_mask"],
        sepp_config_file=EXAMPLE["sepp_config_file"],
        detection_label=detection_tag,
        fixed_apertures_arcsec=(5.0, 10.0),
        save_residual_plots=True,
        thread_count=EXAMPLE["thread_count"],
        overwrite=FORCE_BUILD_CATALOG,
        standardize_catalog=False,
        detection_threshold=DETECTION_THRESHOLD,
    )
    print(f"catalog : {result.catalog_path}")
    print(f"manifest: {result.manifest_path}")
    print(f"log     : {result.log_file}")
    print(f"sources : {result.n_sources}")


def run_batch() -> None:
    """Reuse a :class:`PhotometryConfig` over a list of jobs."""
    cfg = PhotometryConfig(
        sepp_config_file=EXAMPLE["sepp_config_file"],
        detection_label=detection_tag,
        fixed_apertures_arcsec=(5.0, 10.0),
        save_residual_plots=True,
    )

    jobs = [
        dict(
            science_images=EXAMPLE["science_images"],
            detection_image=EXAMPLE["detection_image"],
            reference_catalog=EXAMPLE["reference_catalog"],
            output_dir=EXAMPLE["output_dir"],
            catalog_name=EXAMPLE["catalog_name"],
        ),
    ]

    results = batch_run(jobs, config=cfg, thread_count=8)
    for r in results:
        label = r.job.get("catalog_name") or r.job.get("detection_image")
        if r.status == "ok":
            print(f"OK   {label}  -> {r.result.catalog_path}")
        else:
            print(f"FAIL {label}: {r.error}")


if __name__ == "__main__":
    run_single()