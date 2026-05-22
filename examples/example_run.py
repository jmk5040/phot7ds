"""
End-to-end example: DELVE detection (optional) + ``run_photometry``.

Edit the path constants below for your system, then from the repository root::

    pip install -e .          # or: export PYTHONPATH=$PWD:$PYTHONPATH
    python examples/example_run.py

See README.md § "End-to-end example" for prerequisites (SE++, SWarp, Gaia XP,
network when building DELVE). Set ``FORCE_BUILD_DELVE = True`` to rebuild
DELVE image/mask even when cached files already exist.
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
_PHOT7DS_ROOT = Path(__file__).resolve().parents[1]
if str(_PHOT7DS_ROOT) not in sys.path:
    sys.path.insert(0, str(_PHOT7DS_ROOT))

from phot7ds import PhotometryConfig, batch_run, run_photometry
from phot7ds.detection import build_delve_detection_image

# --- Load inputs from your data-search CSV ---
SEPP_CONFIG = "/data/data1/7DS/RIS/config/7ds_sepp.config" # sourcextractor++ --dump-default-config > default.config
SWARP_CONFIG = "/data/data1/7DS/RIS/config/7ds.swarp" # Swarp -dd > default.swarp
TILE_TABLE = "/data/data1/7DS/RIS/config/7DT_tiles.ascii" # 7DS tile information should be prepared in advance
REFERENCE_DIR = "/data/data1/7DS/RIS/catalog/gaiaxp/" # Gaia XP reference catalog directory
OUTPUT_DIR = Path(__file__).resolve().parent / "example_run" # output directory
DETECT_IMG_DIR = f"/data/data1/7DS/DELVE/" # DELVE or 7DS detection image directory

# Set True to rebuild DELVE even when a cached coadd already exists.
FORCE_BUILD_DELVE = False

# number of cores for the photometry pipeline
NCORES = 8

# required science images common header keys: OBJECT, OBJCTRA, OBJCTDEC (for DELVE detection image building)
science_images = ['/data/data1/processed_1x1_gain2750/T06910/7DT14/m625/calib_7DT14_T06910_20250620_024540_m625_300.com.fits',
 '/data/data1/processed_1x1_gain2750/T06910/7DT06/m575/calib_7DT06_T06910_20250620_024534_m575_300.com.fits',
 '/data/data1/processed_1x1_gain2750/T06910/7DT10/m825/calib_7DT10_T06910_20250620_024533_m825_300.com.fits']

for path in science_images:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Science image not found: {path}")
tile = fits.getheader(science_images[0])['OBJECT']

# check if the detection image and mask exist
detection_image = f"{DETECT_IMG_DIR}/{tile}/{tile}_DELVE_DR3_IMAGE_det.fits"
detection_mask = f"{DETECT_IMG_DIR}/{tile}/{tile}_DELVE_DR3_MASK_det.fits"

if os.path.exists(detection_image) and os.path.exists(detection_mask) and not FORCE_BUILD_DELVE:
    print(f"Using existing DELVE detection image: {detection_image} and mask: {detection_mask}")
else:
    print(f"Building DELVE detection image: {detection_image} and mask: {detection_mask}")
    logging.basicConfig(level=logging.INFO)
    tile_tbl = Table.read(TILE_TABLE, format="ascii")
    tile_info = tile_tbl[tile_tbl["tile"] == tile]
    if len(tile_info) == 0:
        raise ValueError(f"No row for tile {tile!r} in {TILE_TABLE}")
    detection_image, _det_weight = build_delve_detection_image(
        tile_info=tile_info,
        imgtype="image",
        output_path=DETECT_IMG_DIR,
        swarp_cfg_path=SWARP_CONFIG,
        ncores=NCORES,
    )
    detection_mask, _det_weight = build_delve_detection_image(
        tile_info=tile_info,
        imgtype="mask",
        output_path=DETECT_IMG_DIR,
        swarp_cfg_path=SWARP_CONFIG,
        ncores=NCORES,
    )
    print(f"built DELVE detection image: {detection_image} and mask: {detection_mask}")
detection_tag = "DELVE" if "DELVE" in os.path.basename(detection_image) else "7DT"

# load the reference catalog
reference_catalog = f"{REFERENCE_DIR}/gaiaxp_dr3_synphot_{tile}.csv"
if not os.path.exists(reference_catalog):
    raise FileNotFoundError(f"Gaia XP catalog not found: {reference_catalog}")

# output catalog name: saved under OUTPUT_DIR (created automatically if missing).
catalog_name = f"test.zp.fits"

# parameters for the photometry pipeline
EXAMPLE = {
    "sepp_config_file": SEPP_CONFIG,
    "reference_catalog": reference_catalog,
    "detection_image": detection_image,
    "badpix_mask": detection_mask,
    "science_images": science_images,
    "catalog_name": catalog_name,
    "output_dir": str(OUTPUT_DIR),
    "detection_threshold": 10.0, # set higher threshold for DELVE detection image
    "thread_count": NCORES,
}

# run the photometry pipeline for a single tile
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
        detection_threshold=EXAMPLE["detection_threshold"],
        fixed_apertures_arcsec=(5.0, 10.0),
        save_residual_plots=True,
        thread_count=EXAMPLE["thread_count"],
        standardize_catalog=False,
    )
    print(f"catalog : {result.catalog_path}")
    print(f"manifest: {result.manifest_path}")
    print(f"log     : {result.log_file}")
    print(f"sources : {result.n_sources}")

# run the photometry pipeline for a batch of tiles
def run_batch() -> None:
    """Reuse a :class:`PhotometryConfig` over a list of jobs."""
    cfg = PhotometryConfig(
        sepp_config_file=EXAMPLE["sepp_config_file"],
        detection_label=detection_tag,
        detection_threshold=10.0,
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

# main function
if __name__ == "__main__":
    run_single()