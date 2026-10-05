"""
Smoke tests for the ``phot7ds`` package.

These tests don't invoke ``sourcextractor++`` or download anything from the
network; they cover the import surface, the schema, the config-merging
machinery, and a handful of utility helpers.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest
from astropy.table import Table


def test_imports() -> None:
    import phot7ds

    assert phot7ds.__version__
    assert hasattr(phot7ds, "run_photometry")
    assert hasattr(phot7ds, "PhotometryResult")
    assert hasattr(phot7ds, "PhotometryConfig")
    assert hasattr(phot7ds, "batch_run")


def test_default_bands_order() -> None:
    from phot7ds import DEFAULT_BANDS

    assert DEFAULT_BANDS[:3] == ["g", "r", "i"]
    assert DEFAULT_BANDS[3] == "m400"
    assert DEFAULT_BANDS[-1] == "m875"
    assert len(DEFAULT_BANDS) == 23


def test_run_photometry_signature_is_keyword_only() -> None:
    """Every parameter on ``run_photometry`` must be keyword-only."""
    from phot7ds import run_photometry

    sig = inspect.signature(run_photometry)
    for name, p in sig.parameters.items():
        assert p.kind == inspect.Parameter.KEYWORD_ONLY, (
            f"parameter {name!r} should be keyword-only, got {p.kind}"
        )

    required = {
        "science_images",
        "detection_image",
        "reference_catalog",
    }
    assert required.issubset(set(sig.parameters))
    assert "catalog_path" in sig.parameters
    assert "catalog_name" in sig.parameters
    assert "output_dir" in sig.parameters
    assert "config" in sig.parameters
    assert "sepp_config_file" in sig.parameters
    assert sig.parameters["standardize_catalog"].default is False


def test_run_photometry_requires_sepp_config_file() -> None:
    from phot7ds import run_photometry

    with pytest.raises(ValueError, match="sepp_config_file"):
        run_photometry(
            science_images=[],
            detection_image="/tmp/x.fits",
            reference_catalog="/tmp/y.csv",
            catalog_path="/tmp/out/cat_phot.zp.fits",
        )


def test_normalize_catalog_path() -> None:
    """Final catalog name is taken verbatim; no forced ``_phot.zp.fits``."""
    from phot7ds.pipeline import _normalize_catalog_path

    assert _normalize_catalog_path(
        "/tmp/test_zp.fits"
    ).name == "test_zp.fits"
    assert _normalize_catalog_path(
        "/tmp/T01_20260512_DELVE"
    ).name == "T01_20260512_DELVE.fits"
    assert _normalize_catalog_path(
        "/tmp/T01_20260512_DELVE_phot.zp.fits"
    ).name == "T01_20260512_DELVE_phot.zp.fits"


def test_catalog_name_under_output_dir(tmp_path) -> None:
    from phot7ds.pipeline import _resolve_output_paths

    work_dir, raw, zp, _, _, run_name = _resolve_output_paths(
        output_dir=tmp_path / "out",
        catalog_path=None,
        catalog_name="test_zp.fits",
        run_name=None,
        detection_image="/data/det.fits",
        detection_label="DELVE",
    )
    assert work_dir == tmp_path / "out"
    assert zp == tmp_path / "out" / "test_zp.fits"
    assert raw == tmp_path / "out" / "test_zp_raw.fits"
    assert (tmp_path / "out").is_dir()
    assert run_name == "test_zp"

    # Bare stem -> `.fits` is appended.
    _, raw2, zp2, _, _, run_name2 = _resolve_output_paths(
        output_dir=tmp_path / "out2",
        catalog_path=None,
        catalog_name="my_run",
        run_name=None,
        detection_image="/data/det.fits",
        detection_label="DELVE",
    )
    assert zp2.name == "my_run.fits"
    assert raw2.name == "my_run_raw.fits"
    assert run_name2 == "my_run"


def test_photometry_config_default_apertures() -> None:
    """Default apertures must use the zero-padded `aper05` convention."""
    from phot7ds import PhotometryConfig

    cfg = PhotometryConfig()
    assert cfg.apertures[0] == "aper05"
    assert cfg.depth_apertures == ("aper05",)


def test_detection_preset_resolution() -> None:
    """``detection_label`` selects the SE++ tuning preset."""
    from phot7ds import resolve_preset
    from phot7ds.pipeline import _apply_detection_preset
    from phot7ds.config import PhotometryConfig

    delve = resolve_preset("DELVE")
    assert delve["detection_threshold"] == 10.0
    assert delve["auto_kron_min_radius"] == 8.0
    seven_dt = resolve_preset("7DT")
    assert seven_dt["detection_threshold"] == 1.5

    # Unknown labels fall back to the default tuning (7DT) silently.
    unknown = resolve_preset("nonsense")
    assert unknown == seven_dt

    # `None` fields on the merged config get filled from the preset...
    cfg = PhotometryConfig(detection_label="DELVE")
    merged = _apply_detection_preset(cfg, "DELVE")
    assert merged.detection_threshold == 10.0
    assert merged.auto_kron_min_radius == 8.0

    # ... but explicit values always win.
    cfg2 = PhotometryConfig(detection_label="DELVE", detection_threshold=3.3)
    merged2 = _apply_detection_preset(cfg2, "DELVE")
    assert merged2.detection_threshold == 3.3


def test_photometry_config_replace_and_to_dict() -> None:
    from phot7ds import PhotometryConfig

    cfg = PhotometryConfig(sepp_config_file="/tmp/f.config", detection_threshold=3.0)
    assert cfg.detection_threshold == 3.0
    cfg2 = cfg.replace(detection_threshold=10.0, fixed_apertures_arcsec=(5, 10))
    assert cfg2.detection_threshold == 10.0
    assert cfg2.fixed_apertures_arcsec == (5, 10)
    assert cfg.detection_threshold == 3.0  # original is frozen
    d = cfg2.to_dict()
    assert d["detection_threshold"] == 10.0
    assert d["sepp_config_file"] == "/tmp/f.config"


def test_run_photometry_kwargs_override_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """An explicit kwarg on ``run_photometry`` must win over ``config``."""
    import phot7ds.pipeline as pipeline
    from astropy.io import fits as _fits

    det = tmp_path / "det.fits"
    _fits.PrimaryHDU(data=np.ones((4, 4), dtype=np.float32)).writeto(
        str(det), overwrite=True
    )

    def fake_run(*args, **kwargs):  # noqa: ANN001, ANN003
        raise RuntimeError("short-circuit")

    monkeypatch.setattr(pipeline, "run_sepp", fake_run)

    cfg = pipeline.PhotometryConfig(
        sepp_config_file="/tmp/cfg",
        detection_threshold=1.5,
        thread_count=4,
    )
    with pytest.raises(RuntimeError):
        pipeline.run_photometry(
            science_images=[],
            detection_image=str(det),
            reference_catalog="/tmp/r.csv",
            output_dir=str(tmp_path / "out"),
            config=cfg,
            detection_threshold=99.0,
        )

    merged = pipeline._merge_config(  # type: ignore[attr-defined]
        config=cfg, overrides={"detection_threshold": 99.0, "thread_count": None}
    )
    assert merged.detection_threshold == 99.0
    assert merged.thread_count == 4


def test_deg_to_hms_dms() -> None:
    from phot7ds.detection.delve import deg_to_hms_dms

    ra_str, dec_str = deg_to_hms_dms(78.49624060150376, -61.9811320754717)
    assert ra_str == "05:13:59.10"
    assert dec_str == "-61:58:52.08"

    # Carry-over: 219.9999... rounds up but must not produce ":60.00".
    ra_str2, _ = deg_to_hms_dms(219.99999, 0.0)
    assert ra_str2 == "14:40:00.00"


def test_resolve_swarp_center_from_tile_table() -> None:
    from phot7ds.detection.delve import _resolve_swarp_center

    tile_info = Table(
        {"tile": ["T06910"], "ra": [220.0], "dec": [-39.90566037735849]}
    )
    ra_str, dec_str = _resolve_swarp_center(tile_info, None, None)
    assert ra_str == "14:40:00.00"
    assert dec_str == "-39:54:20.52"


def test_aperture_label_zero_padded(tmp_path) -> None:
    """Aperture column labels are always zero-padded (``aper05``)."""
    from phot7ds.sepp import split_array_columns_to_per_filter

    cat = Table(
        {
            "aper_g_mag": np.array([[18.0, 17.5], [19.0, 18.5]]),
            "aper_g_mag_err": np.array([[0.1, 0.2], [0.3, 0.4]]),
        }
    )
    path = tmp_path / "aper.fits"
    cat.write(str(path), format="fits", overwrite=True)
    out = split_array_columns_to_per_filter(
        str(path),
        band_names=["g"],
        fixed_apertures=[5, 10],
        overwrite=False,
    )
    assert out is not None
    assert "aper05_mag_g" in out.colnames
    assert "aper10_mag_g" in out.colnames
    assert "aper5_mag_g" not in out.colnames


def test_build_coverage_mask_records_mskratio_comment(tmp_path) -> None:
    """The MSKRATIO FITS keyword carries a description comment."""
    from astropy.io import fits

    from phot7ds.images import build_coverage_mask

    det = tmp_path / "det.fits"
    sci = tmp_path / "sci.fits"
    data_det = np.ones((20, 20), dtype=np.float32)
    data_det[0:2, :] = 0  # 10 % zero-pixel slab
    data_sci = np.ones_like(data_det)
    fits.PrimaryHDU(data=data_det).writeto(str(det), overwrite=True)
    fits.PrimaryHDU(data=data_sci).writeto(str(sci), overwrite=True)

    out_mask = tmp_path / "mask.fits"
    _, ratio = build_coverage_mask(
        detection_image=str(det),
        science_images=[str(sci)],
        output_path=str(out_mask),
        max_masked_fraction=0.99,
    )
    assert 0.0 < ratio <= 1.0
    with fits.open(str(out_mask)) as hdul:
        card = hdul[0].header.cards["MSKRATIO"]
    assert "masked" in card.comment.lower()


def test_build_coverage_mask_skips_on_shape_mismatch(tmp_path) -> None:
    """A non-standard science-image shape skips the mask (returns None)."""
    from astropy.io import fits

    from phot7ds.images import build_coverage_mask

    det = tmp_path / "det.fits"
    sci = tmp_path / "sci.fits"
    fits.PrimaryHDU(
        data=np.ones((20, 30), dtype=np.float32)
    ).writeto(str(det), overwrite=True)
    # Different shape -> not on the detection grid.
    fits.PrimaryHDU(
        data=np.ones((15, 25), dtype=np.float32)
    ).writeto(str(sci), overwrite=True)

    out_mask = tmp_path / "mask.fits"
    path, ratio = build_coverage_mask(
        detection_image=str(det),
        science_images=[str(sci)],
        output_path=str(out_mask),
    )
    assert path is None
    assert ratio is None
    assert not out_mask.exists()


def test_annotate_catalog_meta_writes_manifest_keys() -> None:
    """The final catalog meta carries informative cards (DETIMG, MSKRATIO, ...)."""
    from phot7ds.config import PhotometryConfig
    from phot7ds.pipeline import _annotate_catalog_meta

    cfg = PhotometryConfig(detection_label="DELVE", detection_threshold=10.0)
    meta: dict = {}
    _annotate_catalog_meta(
        meta,
        detection_image="/path/to/T06910_DELVE_DR3_IMAGE_det.fits",
        coverage_mask="/path/to/mask.fits",
        badpix_mask=None,
        reference_catalog="/path/to/gaiaxp.csv",
        science_images=["/x/img_a.fits", "/x/img_b.fits"],
        detection_label="DELVE",
        mask_ratio=0.0123,
        cfg=cfg,
        run_name="T06910_DELVE",
    )
    assert meta["DETLABEL"][0] == "DELVE"
    assert meta["DETIMG"][0] == "T06910_DELVE_DR3_IMAGE_det.fits"
    assert meta["MSKRATIO"][0] == 0.012
    assert meta["NSCIIMG"][0] == 2
    assert meta["SCIMG000"][0] == "img_a.fits"
    assert meta["SCIMG001"][0] == "img_b.fits"
    assert meta["REFCAT"][0] == "gaiaxp.csv"
    assert meta["PHOTRUN"][0] == "T06910_DELVE"
    assert meta["DETTHR"][0] == 10.0


def _write_header_only(path, **cards) -> str:
    from astropy.io import fits

    hdu = fits.PrimaryHDU(data=np.zeros((4, 4), dtype=np.float32))
    for key, value in cards.items():
        hdu.header[key] = value
    hdu.writeto(str(path), overwrite=True)
    return str(path)


def test_measurement_gain_and_saturation_come_from_each_image(tmp_path) -> None:
    """EGAIN/SATURATE of each measurement image, not SATLV or the detection GAIN."""
    from phot7ds.images import extract_band_names_and_saturation, extract_gain_values

    coadd = _write_header_only(tmp_path / "g.fits", FILTER="g", EGAIN=44.8, SATURATE=775.9)
    single = _write_header_only(tmp_path / "r.fits", FILTER="r", GAIN=1.3, SATLV=60000.0)
    bare = _write_header_only(tmp_path / "i.fits", FILTER="i")
    both = _write_header_only(
        tmp_path / "z.fits", FILTER="z", EGAIN=7.6, GAIN=0.8, SATURATE=4612.0, SATLV=60000.0,
    )
    imgs = [coadd, single, bare, both]

    bands, sats = extract_band_names_and_saturation(imgs, default_saturation=10000)
    gains = extract_gain_values(imgs)
    assert bands == ["g", "r", "i", "z"]
    assert sats == [775.9, 60000.0, 10000.0, 4612.0]
    assert gains == [44.8, 1.3, 0.0, 7.6]


def test_sepp_python_config_sets_per_image_gain_and_saturation(tmp_path) -> None:
    from phot7ds.sepp import generate_sepp_python_config

    cfg = tmp_path / "sepp.py"
    generate_sepp_python_config(
        config_file=str(cfg),
        sciimgs=["/x/g.fits", "/x/r.fits"],
        band_names=["g", "r"],
        saturation_values=[775.9451869169503, 4611.99538016619],
        gain=[44.80280418295595, 7.6062614417710055],
        aper_radius_pix=[2.0],
        fixed_apertures=[5.0],
    )
    text = cfg.read_text()
    assert "load_fits_images(measurement_image_paths)\n" in text
    assert "gain=" not in text
    ns: dict = {}
    exec(text[text.index("gain_values = ["):text.index("for idx, img")], ns)
    assert ns["gain_values"] == [44.80280418295595, 7.6062614417710055]
    assert ns["saturation_values"] == [775.9451869169503, 4611.99538016619]
    assert "img.gain = gain_values[idx]" in text
    assert "img.saturation = saturation_values[idx]" in text

    with pytest.raises(ValueError):
        generate_sepp_python_config(
            config_file=str(cfg), sciimgs=["/x/g.fits", "/x/r.fits"], band_names=["g", "r"],
            saturation_values=[1.0, 2.0], gain=[1.0], aperture_photometry=False,
        )


def test_annotate_catalog_meta_records_measurement_gain() -> None:
    from phot7ds.config import PhotometryConfig
    from phot7ds.pipeline import _annotate_catalog_meta

    meta: dict = {}
    _annotate_catalog_meta(
        meta,
        detection_image="/p/det.fits", coverage_mask=None, badpix_mask=None,
        reference_catalog="/p/ref.csv", science_images=["/x/a.fits", "/x/b.fits"],
        detection_label="7DT", mask_ratio=None, cfg=PhotometryConfig(), run_name="r",
        gain_values=[44.8, 7.6], saturation_values=[775.9, 4612.0],
    )
    assert meta["EGAIN000"][0] == 44.8
    assert meta["EGAIN001"][0] == 7.6
    assert meta["SATUR001"][0] == 4612.0
    assert "DETGAIN" not in meta


def test_detection_gain_prefers_egain(tmp_path) -> None:
    """7DS white stacks use EGAIN (not SWarp's GAIN); DELVE mosaics fall back to GAIN."""
    from phot7ds.images import read_detection_gain

    white = _write_header_only(tmp_path / "white.fits", GAIN=38742.7, EGAIN=3.416)
    delve = _write_header_only(tmp_path / "delve.fits", GAIN=131.6)
    bare = _write_header_only(tmp_path / "bare.fits")
    assert read_detection_gain(white) == (3.416, "EGAIN")
    assert read_detection_gain(delve) == (131.6, "GAIN")
    assert read_detection_gain(bare) == (0.0, None)


def test_zp_solution_is_recorded_and_reproducible(caplog) -> None:
    """Each band x aperture ZP solution is logged and rebuilds the surface exactly."""
    import json
    import logging

    from astropy.modeling import models

    from phot7ds.calibration import calibrate_zeropoints

    rng = np.random.default_rng(1)
    n = 400
    x = rng.uniform(1, 10000, n)
    y = rng.uniform(1, 7000, n)
    ra = 150.0 + x * 1e-5
    dec = 2.0 + y * 1e-5
    ref = rng.uniform(12.5, 15.5, n)
    true_zp = 23.9 + 2e-6 * x - 3e-6 * y + 1e-10 * x * y
    cat = Table({
        "world_centroid_alpha": ra, "world_centroid_delta": dec,
        "pixel_centroid_x": x, "pixel_centroid_y": y,
        "aper05_mag_m500": ref - true_zp + rng.normal(0, 0.005, n),
        "aper05_mag_err_m500": np.full(n, 0.01),
        "aper05_flags_m500": np.zeros(n, dtype=int),
    })
    ref_cat = Table({"ra": ra, "dec": dec, "mag_m500": ref})

    with caplog.at_level(logging.INFO, logger="phot7ds.calibration"):
        calibrate_zeropoints(cat, ref_cat=ref_cat, band_names=["m500"], apertures=["aper05"])

    sol = cat.meta["zp_solutions"][("aper05", "m500")]
    json.dumps(sol)
    sp = sol["spatial"]
    assert sp["degree"] == 2 and sp["n_input"] == n and 0 < sp["n_fit"] <= n
    assert sol["inst_col"] == "aper05_mag_m500"
    rebuilt = models.Polynomial2D(sp["degree"], **sp["coeffs"])
    np.testing.assert_allclose(
        np.asarray(cat["aper05c_mag_m500"]),
        np.asarray(cat["aper05_mag_m500"]) - sol["constant"]["zp"] + rebuilt(x, y),
        atol=1e-9,
    )
    np.testing.assert_allclose(rebuilt(x, y), true_zp, atol=0.005)

    lines = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("ZP m500 aper05: constant") for m in lines)
    assert any(m.startswith("ZP m500 aper05: coeffs") and "c1_1=" in m for m in lines)
    assert not any(m.startswith("Spatial ZP") for m in lines)


def test_provenance_reports_version_and_git() -> None:
    import phot7ds
    from phot7ds.provenance import collect_provenance, format_provenance

    prov = collect_provenance()
    assert prov["phot7ds_version"] == phot7ds.__version__
    assert prov["python_version"]
    line = format_provenance(prov)
    assert line.startswith(f"phot7ds {phot7ds.__version__} | ")
    if prov["git"] is not None:
        assert len(prov["git"]["commit"]) == 40


def test_config_io_required_files_raise(tmp_path) -> None:
    """``require_*`` helpers raise informative errors when files are missing."""
    from phot7ds import require_gaiaxp_reference, require_tile_table

    missing = tmp_path / "nope.ascii"
    with pytest.raises(FileNotFoundError, match="Tile table"):
        require_tile_table(missing)

    with pytest.raises(FileNotFoundError, match="Gaia XP"):
        require_gaiaxp_reference(tmp_path, tile="T00000")


def test_delve_retryable_http_errors() -> None:
    """502/503/504 and connection errors are classified as retryable."""
    import requests

    from phot7ds.detection.delve import _backoff_seconds, _retryable_exception

    class _Resp:
        def __init__(self, status: int) -> None:
            self.status_code = status

    err502 = requests.HTTPError("502 bad gateway")
    err502.response = _Resp(502)  # type: ignore[attr-defined]
    assert _retryable_exception(err502)
    err404 = requests.HTTPError("404 not found")
    err404.response = _Resp(404)  # type: ignore[attr-defined]
    assert not _retryable_exception(err404)

    assert _retryable_exception(requests.ConnectionError("reset"))
    assert _retryable_exception(requests.Timeout("read timeout"))
    assert _retryable_exception(RuntimeError("Bad Gateway from upstream"))
    assert not _retryable_exception(ValueError("schema mismatch"))

    # Backoff grows roughly exponentially and respects the cap.
    short = _backoff_seconds(1, base=2.0, cap=60.0)
    long = _backoff_seconds(5, base=2.0, cap=60.0)
    assert 1.0 <= short <= 3.0
    assert long <= 90.0  # cap*1.5 (jitter upper bound)


def test_canonical_schema_layout() -> None:
    from phot7ds.schema import CANONICAL_BASIC_COLS, build_canonical_schema

    schema = build_canonical_schema(
        bands=["g", "r"], apertures=["auto"], flux_fractions=[0.5, 0.9]
    )
    for col in CANONICAL_BASIC_COLS:
        assert col in schema
    for band in ("g", "r"):
        for quantity in ("flux", "flux_err", "mag", "mag_err", "flags"):
            assert f"auto_{quantity}_{band}" in schema
        assert f"autoc_mag_{band}" in schema
        for suffix in ("50", "90"):
            assert f"flux_rad_{suffix}_{band}" in schema


def test_standardize_catalog_adds_placeholders() -> None:
    from phot7ds.schema import (
        PLACEHOLDER_FILL,
        PLACEHOLDER_TAG,
        build_canonical_schema,
        standardize_catalog,
    )

    cat = Table(
        {
            "source_id": np.array([1, 2, 3], dtype=np.int64),
            "world_centroid_alpha": np.array([10.0, 11.0, 12.0]),
            "world_centroid_delta": np.array([-1.0, -2.0, -3.0]),
            "auto_mag_g": np.array([18.0, 19.0, 20.0]),
            "auto_mag_err_g": np.array([0.01, 0.02, 0.03]),
            "auto_mag-1": np.array([0.0, 0.0, 0.0]),  # SE++ duplicate
        }
    )
    schema = build_canonical_schema(
        bands=["g", "r"], apertures=["auto"], flux_fractions=[0.5, 0.9]
    )
    out = standardize_catalog(cat, schema)
    assert out.meta["NDUPS"] == 1
    assert out.meta["NPLACE"] > 0
    placeholder_col = "auto_mag_r"
    assert placeholder_col in out.colnames
    assert out[placeholder_col].description == PLACEHOLDER_TAG
    assert np.all(np.asarray(out[placeholder_col]) == PLACEHOLDER_FILL)


def test_trim_to_tile_polygon() -> None:
    from phot7ds.tile_geometry import trim_to_tile_polygon

    tile_info = Table(
        {
            "tile": ["T00001"],
            "ra1": [10.0], "dec1": [-1.0],
            "ra2": [11.0], "dec2": [-1.0],
            "ra3": [11.0], "dec3": [0.0],
            "ra4": [10.0], "dec4": [0.0],
        }
    )
    cat = Table(
        {
            "ra": [10.5, 9.0, 10.4, 11.5],
            "dec": [-0.5, -0.5, -0.9, -0.5],
        }
    )
    trimmed = trim_to_tile_polygon(tile_info, cat, margin=0.0)
    assert set(np.asarray(trimmed["ra"]).tolist()) == {10.5, 10.4}


@pytest.mark.parametrize("margin", [0.0, 0.06])
def test_trim_to_tile_polygon_across_ra_zero(margin) -> None:
    from phot7ds.tile_geometry import trim_to_tile_polygon

    tile_info = {
        "ra1": 359.3, "dec1": -0.5, "ra2": 0.7, "dec2": -0.5,
        "ra3": 0.7, "dec3": 0.5, "ra4": 359.3, "dec4": 0.5,
    }
    cat = Table({
        "ra": [359.9, 0.2, 359.5, 0.5, 180.0, 358.0, 2.0],
        "dec": [0.0, 0.1, -0.2, 0.3, 0.0, 0.0, 0.0],
    })
    trimmed = trim_to_tile_polygon(tile_info, cat, margin=margin)
    # RA values are returned as stored, not unwrapped.
    assert sorted(np.asarray(trimmed["ra"]).tolist()) == [0.2, 0.5, 359.5, 359.9]


def test_trim_to_tile_polygon_polar_cap() -> None:
    """The polar-cap tile (corners circling the pole) keeps the pole region."""
    from phot7ds.tile_geometry import trim_to_tile_polygon

    tile_info = {
        "ra1": 56.36, "ra2": 123.64, "ra3": 236.36, "ra4": 303.64,
        "dec1": -89.17, "dec2": -89.17, "dec3": -89.17, "dec4": -89.17,
    }
    cat = Table({
        "ra": [0.0, 90.0, 200.0, 315.0, 0.0, 180.0],
        "dec": [-89.99, -89.7, -89.6, -89.4, -88.5, 89.9],
    })
    trimmed = trim_to_tile_polygon(tile_info, cat, margin=0.0)
    assert sorted(np.asarray(trimmed["dec"]).tolist()) == [-89.99, -89.7, -89.6, -89.4]


def test_white_stack_pixel_scale_from_pc_matrix(tmp_path) -> None:
    """A PC-matrix header with CDELT1 = 1 must not read as 3600 arcsec/pix."""
    from phot7ds.detection.sevends import _reference_frame

    scale = 0.505 / 3600.0
    cards = dict(
        CTYPE1="RA---TAN", CTYPE2="DEC--TAN", CRVAL1=150.0, CRVAL2=2.0,
        CRPIX1=2.0, CRPIX2=2.0, CUNIT1="deg", CUNIT2="deg",
    )
    pc = _write_header_only(
        tmp_path / "pc.fits", **cards, CDELT1=1.0, CDELT2=1.0,
        PC1_1=-scale, PC1_2=0.0, PC2_1=0.0, PC2_2=scale,
    )
    cd = _write_header_only(
        tmp_path / "cd.fits", **cards, CD1_1=-scale, CD1_2=0.0, CD2_1=0.0, CD2_2=scale,
    )
    for path in (pc, cd):
        ra, dec, pix, nx, ny = _reference_frame(path)
        assert (ra, dec, nx, ny) == (150.0, 2.0, 4, 4)
        assert pix == pytest.approx(0.505)


def test_organize_images_by_filter_dict() -> None:
    from phot7ds.images import organize_images_by_filter

    bands_dict = {"g": 4770, "r": 6231, "i": 7625, "m425": 4250}
    images = [
        "/data/T0001_g_X_20260101_010101_300s.fits",
        "/data/T0001_r_X_20260101_010101_300s.fits",
        "/data/T0001_r_X_20260102_010101_300s.fits",
        "/data/T0001_m425_X_20260101_010101_300s.fits",
    ]
    result = organize_images_by_filter(
        images, bands_dict, filter_source="filename",
        output_form="dict", keep_duplicates=False,
    )
    assert isinstance(result["g"], str)
    assert isinstance(result["r"], str)
    assert result["i"] is None
    assert isinstance(result["m425"], str)


def test_batch_run_continues_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import phot7ds.batch as batch

    calls = []

    def fake_run(**kwargs):  # noqa: ANN003
        calls.append(kwargs)
        if kwargs.get("output_dir", "").endswith("fail"):
            raise RuntimeError("boom")
        return batch.PhotometryResult(
            catalog_path="/tmp/cat.fits",
            manifest_path="/tmp/man.json",
            log_file="/tmp/log.log",
            n_sources=42,
        )

    monkeypatch.setattr(batch, "run_photometry", fake_run)

    jobs = [
        dict(science_images=[], detection_image="d", reference_catalog="r",
             output_dir="/tmp/ok", sepp_config_file="/tmp/c"),
        dict(science_images=[], detection_image="d", reference_catalog="r",
             output_dir="/tmp/fail", sepp_config_file="/tmp/c"),
    ]
    results = batch.batch_run(jobs, thread_count=2)
    assert [r.status for r in results] == ["ok", "failed"]
    assert all(c.get("thread_count") == 2 for c in calls)
