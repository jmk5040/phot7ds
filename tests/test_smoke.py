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
    from phot7ds.pipeline import _normalize_catalog_path

    assert _normalize_catalog_path(
        "/tmp/T01_20260512_DELVE_phot.zp.fits"
    ).name == "T01_20260512_DELVE_phot.zp.fits"
    assert _normalize_catalog_path(
        "/tmp/T01_20260512_DELVE"
    ).name == "T01_20260512_DELVE_phot.zp.fits"
    assert _normalize_catalog_path("/tmp/test.zp.fits").name == "test.zp.fits"


def test_catalog_name_under_output_dir(tmp_path) -> None:
    from phot7ds.pipeline import _resolve_output_paths

    work_dir, _, zp, _, _, run_name = _resolve_output_paths(
        output_dir=tmp_path / "out",
        catalog_path=None,
        catalog_name="test.zp.fits",
        run_name=None,
        detection_image="/data/det.fits",
        detection_label="DELVE",
    )
    assert work_dir == tmp_path / "out"
    assert zp == tmp_path / "out" / "test.zp.fits"
    assert (tmp_path / "out").is_dir()
    assert run_name == "test"


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


def test_run_photometry_kwargs_override_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit kwarg on ``run_photometry`` must win over ``config``."""
    import phot7ds.pipeline as pipeline

    captured: dict = {}

    def fake_run(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
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
            detection_image=__file__,  # not opened before short-circuit
            reference_catalog="/tmp/r.csv",
            output_dir="/tmp/phot7ds_test_out",
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


def test_resolve_swarp_center_from_tile_table() -> None:
    from phot7ds.detection.delve import _resolve_swarp_center

    tile_info = Table(
        {"tile": ["T06910"], "ra": [220.0], "dec": [-39.90566037735849]}
    )
    ra_str, dec_str = _resolve_swarp_center(tile_info, None, None)
    assert ra_str == "14:40:00.00"
    assert dec_str == "-39:54:20.52"


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
