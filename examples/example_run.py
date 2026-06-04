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

from glob import glob
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
from phot7ds.detection import (  # noqa: E402
    build_7ds_detection_image,
    build_delve_detection_image,
)

# --- Repo-relative paths (override on demand) ------------------------------
CONFIG_DIR = _HERE / "config"
SEPP_CONFIG = CONFIG_DIR / "7ds_sepp.config"   # auto-created if missing
SWARP_CONFIG = CONFIG_DIR / "7ds.swarp"        # auto-created if missing
TILE_TABLE = CONFIG_DIR / "7DT_tiles.ascii"    # USER-supplied, must exist
REFERENCE_DIR = CONFIG_DIR / "gaiaxp"          # USER-supplied Gaia XP CSVs
OUTPUT_DIR = _HERE / "example_run"             # photometry outputs

# Detection image source:
#   "DELVE" -> download a DELVE-DR3 mosaic + bad-pixel mask for the tile.
#   "7DS"   -> stack the tile's local single-band coadds into a white image.
# DETECTION_SOURCE = "DELVE"
DETECTION_SOURCE = "7DS"
if DETECTION_SOURCE == "DELVE":
    DETECT_IMG_DIR = "/data/data1/7DS/DELVE"          # output dir for detection mosaics
elif DETECTION_SOURCE == "7DS":
    DETECT_IMG_DIR = "/data/data2/RIS/data"          # output dir for detection mosaics
else:
    raise ValueError(f"Unknown DETECTION_SOURCE: {DETECTION_SOURCE!r}")
# For DETECTION_SOURCE == "7DS": directory holding this tile's per-band
# ``*_coadd.fits`` science images plus ``*_coadd_weight.fits`` weight maps.
tile = "T00238"
# Output catalog name: leaf only; saved under OUTPUT_DIR (created if missing).
# Whatever you pass here is preserved verbatim (no `_phot.zp.fits` rename).
catalog_name = f"{tile}_{DETECTION_SOURCE}_phot.fits"

SEVENDS_IMAGE_DIR = f"/data/data2/RIS/data/{tile}"
# Stack medium bands only (skip g/r/i) when building the 7DS white image.
SEVENDS_MEDIUM_ONLY = True
# Keep one representative (sharpest-seeing) image per band instead of
# stacking every visit. Useful for tiles with many repeat exposures.
SEVENDS_ONE_PER_BAND = True

# Set True to rebuild the detection image even when a cached one exists.
FORCE_BUILD_DELVE = False

# Set True to rebuild the catalog even when a cached one already exists.
FORCE_BUILD_CATALOG = True

# Set the detection threshold for the DELVE detection image (otherwise using default configuration based on the detection label)
DETECTION_THRESHOLD = None

# Worker counts: for photometry and DELVE SIA downloads
NCORES = 12

# Bootstrap the SE++ / SWarp configs on the first run.
SEPP_CONFIG = ensure_sepp_config(SEPP_CONFIG)
SWARP_CONFIG = ensure_swarp_config(SWARP_CONFIG)

# Required science images. The OBJECT / OBJCTRA / OBJCTDEC FITS headers
# identify the tile (and supply the SWarp -CENTER for DELVE).
science_images = sorted(
    f for f in glob(f"{SEVENDS_IMAGE_DIR}/*_coadd.fits")
    if not f.endswith("_weight.fits")
)
for path in science_images:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Science image not found: {path}")

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


# --- Detection image (DELVE download or 7DS white stack) ------------------
logging.basicConfig(level=logging.INFO)


def build_delve() -> tuple[str, str | None]:
    """Download a DELVE-DR3 detection image + bad-pixel mask for the tile."""
    det_img = f"{DETECT_IMG_DIR}/{tile}/{tile}_DELVE_DR3_IMAGE_det.fits"
    det_mask = f"{DETECT_IMG_DIR}/{tile}/{tile}_DELVE_DR3_MASK_det.fits"
    if os.path.exists(det_img) and os.path.exists(det_mask) and not FORCE_BUILD_DELVE:
        print(f"Using existing DELVE detection image: {det_img}")
        print(f"Using existing DELVE detection mask : {det_mask}")
        return det_img, det_mask
    print(f"Building DELVE detection image: {det_img}")
    tile_tbl = _load_tile_table(TILE_TABLE)
    tile_info = tile_tbl[tile_tbl["tile"] == tile]
    if len(tile_info) == 0:
        raise ValueError(f"No row for tile {tile!r} in {TILE_TABLE}")
    det_img, _ = build_delve_detection_image(
        tile_info=tile_info, imgtype="image",
        output_path=f"{DETECT_IMG_DIR}/{tile}",
        swarp_cfg_path=str(SWARP_CONFIG), ncores=NCORES, max_retries=5,
    )
    det_mask, _ = build_delve_detection_image(
        tile_info=tile_info, imgtype="mask",
        output_path=f"{DETECT_IMG_DIR}/{tile}",
        swarp_cfg_path=str(SWARP_CONFIG), ncores=NCORES, max_retries=5,
    )
    print(f"Built DELVE detection image: {det_img}")
    return det_img, det_mask


def build_7ds() -> tuple[str, str | None]:
    """Stack the tile's local single-band coadds into a white detection image."""
    image_dir = SEVENDS_IMAGE_DIR
    det_img = f"{DETECT_IMG_DIR}/{tile}/{tile}_7DS_EDR_IMAGE_det.fits"
    if os.path.exists(det_img) and not FORCE_BUILD_DELVE:
        print(f"Using existing 7DS detection image: {det_img}")
        return det_img, None
    print(f"Building 7DS white detection image from: {image_dir}")
    det_img, _ = build_7ds_detection_image(
        image_dir=image_dir,
        output_path=f"{DETECT_IMG_DIR}/{tile}",
        swarp_cfg_path=str(SWARP_CONFIG),
        tile=tile,
        medium_only=SEVENDS_MEDIUM_ONLY,
        one_per_band=SEVENDS_ONE_PER_BAND,
        ncores=NCORES,
        overwrite=FORCE_BUILD_DELVE,
    )
    print(f"Built 7DS white detection image: {det_img}")
    return det_img, None  # 7DS white stack has no separate bad-pixel mask


if DETECTION_SOURCE == "7DS":
    detection_image, detection_mask = build_7ds()
elif DETECTION_SOURCE == "DELVE":
    detection_image, detection_mask = build_delve()
else:
    raise ValueError(f"Unknown DETECTION_SOURCE: {DETECTION_SOURCE!r}")

detection_tag = "DELVE" if "DELVE" in os.path.basename(detection_image) else "7DT"

# --- Reference catalog ----------------------------------------------------
reference_catalog = require_gaiaxp_reference(REFERENCE_DIR, tile=tile)

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