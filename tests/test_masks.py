"""
Tests for the per-band mask flags (``phot7ds.masks``) and their wiring into
the SE++ command builder and the schema. No SourceExtractor++ run.

The synthetic tile is 12x16 pixels; the "count-map MEF" mimics the py7DT
``<coadd>_counts.fits`` layout (tile-compressed uint8 planes NGEOM, NUSED,
NBAD, NSAT, NTRAIL, NOUTLIER).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Column, Table

from phot7ds.masks import (
    MASKBIT,
    NODATA_BIT,
    PLANE_TO_BIT,
    MaskStaging,
    any_band_nodata,
    build_band_bitmask,
    find_count_mask,
    maskbits_description,
    rename_mask_columns,
    resolve_count_masks,
    sweep_stale_staging,
)

SHAPE = (12, 16)


def _write_coadd(path: Path, band: str, *, nan_at=None, zero_cols: int = 2) -> Path:
    data = np.full(SHAPE, 10.0, dtype=np.float32)
    data[:, :zero_cols] = 0.0                 # off-footprint strip
    if nan_at is not None:
        data[nan_at] = np.nan
    hdr = fits.Header()
    hdr["FILTER"] = band
    hdr["SATLV"] = 50000.0
    fits.PrimaryHDU(data, hdr).writeto(path, overwrite=True)
    return path


def _write_mef(path: Path, band: str, *, shape=SHAPE, compressed: bool = True,
               planes=("NGEOM", "NUSED", "NBAD", "NSAT", "NTRAIL", "NOUTLIER")) -> Path:
    """NBAD at (3,5), NSAT at (4,6), NTRAIL along row 7, NOUTLIER at (3,5) too;
    NUSED == 0 at (9, 9)."""
    prim = fits.PrimaryHDU()
    prim.header["FILTER"] = band
    hdus = [prim]
    arrays = {
        "NGEOM": np.full(shape, 3, np.uint8),
        "NUSED": np.full(shape, 3, np.uint8),
        "NBAD": np.zeros(shape, np.uint8),
        "NSAT": np.zeros(shape, np.uint8),
        "NTRAIL": np.zeros(shape, np.uint8),
        "NOUTLIER": np.zeros(shape, np.uint8),
    }
    if shape == SHAPE:
        arrays["NUSED"][9, 9] = 0
        arrays["NBAD"][3, 5] = 2
        arrays["NSAT"][4, 6] = 1
        arrays["NTRAIL"][7, :] = 1
        arrays["NOUTLIER"][3, 5] = 1
    for name in planes:
        arr = arrays[name]
        hdu = fits.CompImageHDU(arr, name=name) if compressed else fits.ImageHDU(arr, name=name)
        hdus.append(hdu)
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return path


@pytest.fixture
def tile(tmp_path: Path) -> dict:
    coadd = _write_coadd(tmp_path / "T00001_m525_coadd.fits", "m525", nan_at=(10, 10))
    mef = _write_mef(tmp_path / "T00001_m525_coadd_counts.fits", "m525")
    return {"dir": tmp_path, "coadd": coadd, "mef": mef}


# --------------------------------------------------------------------------
# bit folding
# --------------------------------------------------------------------------
def test_bit_table_matches_py7dt_maskbit() -> None:
    assert MASKBIT == {"OUTLIER": 1, "BADPIX": 2, "STRAY": 4, "SATELLITE": 8,
                       "SATURATED": 16, "HOT": 32, "DEAD": 64}
    assert NODATA_BIT == 128
    assert PLANE_TO_BIT["NBAD"] == MASKBIT["BADPIX"]
    assert PLANE_TO_BIT["NTRAIL"] == MASKBIT["SATELLITE"]
    assert maskbits_description().startswith("1=OUTLIER,2=BADPIX") and "128=NODATA" in maskbits_description()


def test_build_band_bitmask_with_mef(tile: dict) -> None:
    out = tile["dir"] / "m525_mask.fits"
    info = build_band_bitmask("m525", str(tile["coadd"]), str(tile["mef"]), str(out), SHAPE)
    bm = fits.getdata(out)
    assert bm.dtype == np.uint8 and bm.shape == SHAPE
    assert info.status == "mef"
    assert set(info.planes) == {"NBAD", "NSAT", "NTRAIL", "NOUTLIER"}
    assert set(info.missing_planes) == {"NSTRAY", "NHOT", "NDEAD"}
    assert "NUSED==0" in info.nodata_sources

    assert bm[3, 5] == MASKBIT["BADPIX"] | MASKBIT["OUTLIER"]      # 3
    assert bm[4, 6] == MASKBIT["SATURATED"]                       # 16
    assert bm[7, 8] == MASKBIT["SATELLITE"]                       # 8
    assert bm[7, 0] == MASKBIT["SATELLITE"] | NODATA_BIT          # trail over the zero strip
    assert bm[9, 9] == NODATA_BIT                                 # NUSED == 0
    assert bm[10, 10] == NODATA_BIT                               # NaN in coadd
    assert np.all(bm[:, :2] & NODATA_BIT)                         # coadd == 0
    assert bm[0, 8] == 0

    hdr = fits.getheader(out)
    assert hdr["MBNODATA"] == 128 and hdr["MBBADPIX"] == 2
    assert hdr["PLANES"] == "NBAD,NSAT,NTRAIL,NOUTLIER"
    assert hdr["SRCMEF"] == tile["mef"].name
    assert info.per_bit_pixels[2] == 1 and info.per_bit_pixels[8] == SHAPE[1]


def test_build_band_bitmask_without_mef_is_coverage_only(tile: dict) -> None:
    out = tile["dir"] / "m525_mask.fits"
    info = build_band_bitmask("m525", str(tile["coadd"]), None, str(out), SHAPE)
    bm = fits.getdata(out)
    assert info.status == "coverage-only" and info.planes == ()
    assert set(np.unique(bm)) == {0, NODATA_BIT}
    assert np.all(bm[:, :2] == NODATA_BIT) and bm[10, 10] == NODATA_BIT
    assert bm[9, 9] == 0                                          # NUSED not available
    assert "SRCMEF" not in fits.getheader(out)


def test_build_band_bitmask_nused_can_be_disabled(tile: dict) -> None:
    out = tile["dir"] / "m525_mask.fits"
    info = build_band_bitmask("m525", str(tile["coadd"]), str(tile["mef"]), str(out), SHAPE,
                              use_nused=False)
    assert fits.getdata(out)[9, 9] == 0
    assert "NUSED==0" not in info.nodata_sources


@pytest.mark.parametrize("breakage", ["wrong_shape", "no_planes", "not_fits", "missing_file"])
def test_broken_mef_degrades_to_coverage_only(tile: dict, breakage: str, caplog) -> None:
    mef = tile["dir"] / "broken_counts.fits"
    if breakage == "wrong_shape":
        _write_mef(mef, "m525", shape=(6, 8))
    elif breakage == "no_planes":
        _write_mef(mef, "m525", planes=("NGEOM",))
    elif breakage == "not_fits":
        mef.write_text("not a fits file")
    else:
        mef = tile["dir"] / "does_not_exist_counts.fits"

    out = tile["dir"] / "m525_mask.fits"
    with caplog.at_level("WARNING", logger="phot7ds.masks"):
        info = build_band_bitmask("m525", str(tile["coadd"]), str(mef), str(out), SHAPE)
    assert info.status.startswith("coverage-only:")
    assert info.count_mask is None and info.planes == ()
    bm = fits.getdata(out)
    assert set(np.unique(bm)) == {0, NODATA_BIT}
    assert "ignored" in caplog.text


def test_coadd_off_detection_grid_raises(tile: dict) -> None:
    with pytest.raises(ValueError, match="detection grid"):
        build_band_bitmask("m525", str(tile["coadd"]), str(tile["mef"]),
                           str(tile["dir"] / "x.fits"), (20, 20))


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------
def test_find_count_mask_conventions(tmp_path: Path) -> None:
    real_dir = tmp_path / "processed"
    real_dir.mkdir()
    link_dir = tmp_path / "tile"
    link_dir.mkdir()
    coadd = _write_coadd(real_dir / "T_m425_coadd.fits", "m425")
    link = link_dir / "T_m425_coadd.fits"
    link.symlink_to(coadd)

    assert find_count_mask(link) is None
    mef = _write_mef(real_dir / "T_m425_coadd_counts.fits", "m425")
    assert find_count_mask(link) == str(mef)                       # next to the symlink target

    other = tmp_path / "masks"
    other.mkdir()
    mef2 = _write_mef(other / "T_m425_coadd_counts.fits", "m425")
    assert find_count_mask(link, mask_dir=other) == str(mef2)      # explicit dir wins
    assert find_count_mask(link, suffix="_cnt.fits") is None


def test_resolve_count_masks_partial_and_forms(tmp_path: Path) -> None:
    imgs, bands = [], ["m425", "m525", "m625"]
    for b in bands:
        imgs.append(str(_write_coadd(tmp_path / f"T_{b}_coadd.fits", b)))
    mef525 = _write_mef(tmp_path / "T_m525_coadd_counts.fits", "m525")

    auto = resolve_count_masks(imgs, bands)
    assert auto == {"m425": None, "m525": str(mef525), "m625": None}

    as_dir = resolve_count_masks(imgs, bands, tmp_path)
    assert as_dir == auto

    mapping = resolve_count_masks(imgs, bands, {"m625": str(mef525), "zzz": "ignored"})
    assert mapping == {"m425": None, "m525": None, "m625": str(mef525)}

    by_header = resolve_count_masks(imgs, bands, [str(mef525)])
    assert by_header["m525"] == str(mef525) and by_header["m425"] is None

    # A path whose FILTER header is unknown falls back to the _<band>_ token.
    odd = _write_mef(tmp_path / "X_m625_counts.fits", "unknown")
    by_name = resolve_count_masks(imgs, bands, [str(odd)])
    assert by_name["m625"] == str(odd)


# --------------------------------------------------------------------------
# staging
# --------------------------------------------------------------------------
def test_mask_staging_partial_mefs_and_cleanup(tmp_path: Path) -> None:
    bands = ["m425", "m525", "m625"]
    imgs = {b: str(_write_coadd(tmp_path / f"T_{b}_coadd.fits", b)) for b in bands}
    mefs = {"m425": None, "m525": str(_write_mef(tmp_path / "T_m525_coadd_counts.fits", "m525")),
            "m625": None}
    imgs["off"] = str(_write_coadd(tmp_path / "T_off_coadd.fits", "off"))  # wrong grid
    with fits.open(imgs["off"], mode="update") as h:
        h[0].data = np.ones((5, 5), np.float32)
    mefs["off"] = None

    stage_root = tmp_path / "shm"
    stage_root.mkdir()
    staging = MaskStaging("run", staging_dir=stage_root, fallback_dir=tmp_path, workers=2)
    with staging:
        infos = staging.stage(imgs, mefs, SHAPE)
        assert set(infos) == set(bands)                      # "off" skipped, run continues
        assert infos["m525"].status == "mef"
        assert infos["m425"].status == "coverage-only"
        for info in infos.values():
            assert Path(info.bitmask).is_file()
            assert Path(info.bitmask).parent == staging.dir
        assert staging.dir.name.startswith("phot7ds_masks_run_") and staging.dir.name.endswith(str(os.getpid()))
    assert not staging.dir.exists()


def test_mask_staging_falls_back_when_tmpfs_unavailable(tmp_path: Path) -> None:
    staging = MaskStaging("run", staging_dir=tmp_path / "nope", fallback_dir=tmp_path, workers=1)
    assert staging.root == tmp_path and staging.location == "disk"


def test_sweep_stale_staging(tmp_path: Path) -> None:
    dead = tmp_path / "phot7ds_masks_old_999999"
    dead.mkdir()
    (dead / "x.fits").write_bytes(b"0")
    alive = tmp_path / f"phot7ds_masks_me_{os.getpid()}"
    alive.mkdir()
    unrelated = tmp_path / "other_dir_123"
    unrelated.mkdir()
    removed = sweep_stale_staging(tmp_path)
    assert removed == [str(dead)]
    assert alive.exists() and unrelated.exists()


# --------------------------------------------------------------------------
# SE++ command and catalog side
# --------------------------------------------------------------------------
def _cmd(**kw) -> str:
    from phot7ds.sepp import build_sepp_command

    return build_sepp_command(
        python_config_file="cfg.py", sepp_config_file="se.config", detection_image="det.fits",
        detection_gain=1.0, detection_saturate=6e4, catalog_path="out.fits", **kw,
    )


def test_build_sepp_command_flag_images() -> None:
    from phot7ds.sepp import FlagImage

    cmd = _cmd(flag_images=[FlagImage("m525", "/dev/shm/m525_mask.fits"),
                            FlagImage("m425", "/dev/shm/m425_mask.fits", "or")])
    assert "--flag-image-m525 /dev/shm/m525_mask.fits --flag-type-m525 or" in cmd
    assert "--flag-image-m425" in cmd and "--flag-image-cover" not in cmd
    assert "ExternalFlags" in cmd

    legacy = _cmd(coverage_mask="/m.fits", badpix_mask="/b.fits")
    assert "--flag-image-cover /m.fits --flag-type-cover or" in legacy
    assert "--flag-image-badpix /b.fits --flag-type-badpix or" in legacy


def test_build_sepp_command_drops_externalflags_without_flag_image() -> None:
    cmd = _cmd()
    assert "--flag-image" not in cmd and "ExternalFlags" not in cmd
    assert "--output-properties" in cmd


def test_build_sepp_command_rejects_bad_labels() -> None:
    from phot7ds.sepp import FlagImage

    with pytest.raises(ValueError, match="Duplicate"):
        _cmd(coverage_mask="/m.fits", flag_images=[FlagImage("cover", "/x.fits")])
    with pytest.raises(ValueError, match="Invalid flag-image label"):
        _cmd(flag_images=[FlagImage("bad label", "/x.fits")])
    with pytest.raises(ValueError, match="Invalid flag-type"):
        _cmd(flag_images=[FlagImage("m525", "/x.fits", "xor")])


def test_rename_mask_columns_and_any_band_nodata() -> None:
    cat = Table({
        "isophotal_image_flags_m525": np.array([0, 128, 3, 130], np.int64),
        "isophotal_image_flags_pixel_count_m525": np.array([0, 5, 2, 9], np.int32),
        "isophotal_image_flags_m425": np.array([0, 0, 0, 0], np.int64),
        "isophotal_image_flags_pixel_count_m425": np.zeros(4, np.int32),
        "isophotal_image_flags_badpix": np.zeros(4, np.int64),
    })
    renamed = rename_mask_columns(cat, ["m525", "m425", "m625"])
    assert renamed == {
        "isophotal_image_flags_m525": "mask_flags_m525",
        "isophotal_image_flags_pixel_count_m525": "mask_npix_m525",
        "isophotal_image_flags_m425": "mask_flags_m425",
        "isophotal_image_flags_pixel_count_m425": "mask_npix_m425",
    }
    assert "isophotal_image_flags_badpix" in cat.colnames        # untouched

    # A schema placeholder (negative sentinel) must not count as flagged.
    cat["mask_flags_m625"] = Column(np.full(4, -99.0))
    nodata = any_band_nodata(cat)
    assert nodata.tolist() == [False, True, False, True]
    assert any_band_nodata(cat, ["m425"]).tolist() == [False] * 4


def test_canonical_schema_has_per_band_mask_columns() -> None:
    from phot7ds.schema import LEGACY_COVER_COLS, build_canonical_schema

    cols = build_canonical_schema(["g", "m525"], ["aper05"])
    assert "mask_flags_g" in cols and "mask_npix_m525" in cols
    assert cols.index("mask_flags_g") < cols.index("aper05_flux_g")
    assert not any(c in cols for c in LEGACY_COVER_COLS)
    assert "isophotal_image_flags_badpix" in cols

    legacy = build_canonical_schema(["g"], ["aper05"], per_band_masks=False)
    assert all(c in legacy for c in LEGACY_COVER_COLS) and "mask_flags_g" not in legacy


def test_config_and_run_photometry_expose_mask_knobs() -> None:
    import inspect

    from phot7ds import PhotometryConfig, run_photometry

    cfg = PhotometryConfig()
    assert cfg.per_band_masks is True and cfg.mask_staging_dir == "/dev/shm"
    assert cfg.count_mask_suffix == "_counts.fits" and cfg.mask_flag_type == "or"
    params = inspect.signature(run_photometry).parameters
    for name in ("count_masks", "per_band_masks", "mask_staging_dir", "mask_use_nused"):
        assert name in params
