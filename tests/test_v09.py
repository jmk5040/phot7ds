"""Filter registry, image selection, output format and header keys (v0.9)."""
from __future__ import annotations

import logging

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table


def _image(path, filter_name=None):
    hdu = fits.PrimaryHDU(data=np.zeros((2, 2), dtype=np.float32))
    if filter_name is not None:
        hdu.header["FILTER"] = filter_name
    hdu.writeto(str(path), overwrite=True)
    return str(path)


def _write_registry(path, rows):
    tbl = Table(rows=rows, names=("name", "kind", "lambda_pivot_nm", "fwhm_nm", "key"))
    tbl.write(str(path), format="ascii.ecsv", overwrite=True)
    return path


# --- registry -------------------------------------------------------------
def test_packaged_registry_contents() -> None:
    from phot7ds.filters import load_filter_registry

    reg = load_filter_registry()
    assert len(reg) == 42
    assert reg.names_of_kind("broad") == ("u", "g", "r", "i", "z")
    assert set(reg.names_of_kind("wide")) == {
        "m375w", "m425w", "m466w", "m692w", "m710w", "m769w", "m832w",
    }
    keys = [f.key for f in reg]
    assert len(set(keys)) == len(keys)
    assert reg["m425"].key == "425" and reg["m425w"].key == "42W"
    assert reg["g"].key == "G" and reg["m875"].key == "875"
    assert reg.canonical("M425W") == "m425w"
    assert reg.canonical("m329w") is None
    pivots = [f.lambda_pivot_nm for f in reg if f.kind != "broad"]
    assert pivots == sorted(pivots)


def test_default_header_key_rule() -> None:
    from phot7ds.filters import default_header_key, default_kind

    assert default_header_key("m329w") == "32W"
    assert default_header_key("m425") == "425"
    assert default_header_key("y") == "Y"
    assert default_kind("m329w") == "wide"
    assert default_kind("m425") == "medium"
    assert default_kind("y") == "broad"


def test_registry_validation(tmp_path) -> None:
    from phot7ds.filters import load_filter_registry

    dup = _write_registry(tmp_path / "dup.ecsv", [
        ("m425", "medium", 425.0, 24.0, "425"),
        ("m4250", "medium", 425.0, 24.0, "425"),
    ])
    with pytest.raises(ValueError, match="shared by"):
        load_filter_registry(dup)
    long_key = _write_registry(tmp_path / "long.ecsv", [
        ("m425w", "wide", 427.0, 48.0, "425W"),
    ])
    with pytest.raises(ValueError, match="1-3"):
        load_filter_registry(long_key)
    bad_kind = _write_registry(tmp_path / "kind.ecsv", [
        ("m425", "narrow", 425.0, 24.0, "425"),
    ])
    with pytest.raises(ValueError, match="kind"):
        load_filter_registry(bad_kind)


def test_registry_env_override(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from phot7ds.filters import REGISTRY_ENV, load_filter_registry

    path = _write_registry(tmp_path / "mini.ecsv", [
        ("g", "broad", 479.0, 142.0, "G"),
        ("m329w", "wide", 330.0, 40.0, "32W"),
    ])
    monkeypatch.setenv(REGISTRY_ENV, str(path))
    reg = load_filter_registry()
    assert reg.names == ("g", "m329w")
    assert reg.header_key("m329w") == "32W"


def test_registry_from_transmission_curves(tmp_path) -> None:
    from phot7ds.filters import FilterInfo, FilterRegistry, registry_from_transmission_curves

    lam = np.linspace(300.0, 1000.0, 1401)
    for name, centre, width in (("g", 480.0, 140.0), ("m425w", 427.0, 48.0),
                                ("m425", 425.0, 24.0), ("m400", 400.0, 24.0)):
        trans = np.where(np.abs(lam - centre) <= width / 2, 0.8, 0.0)
        order = np.arange(lam.size)[::-1]  # unsorted on purpose (cf. m466w.csv)
        Table({"lam": lam[order], "trans": trans[order]}).write(
            str(tmp_path / f"{name}.csv"), format="ascii.csv", overwrite=True,
        )
    tbl = registry_from_transmission_curves(tmp_path)
    assert list(tbl["name"]) == ["g", "m400", "m425", "m425w"]
    assert list(tbl["key"]) == ["G", "400", "425", "42W"]
    assert list(tbl["kind"]) == ["broad", "medium", "medium", "wide"]
    row = tbl[list(tbl["name"]).index("m425w")]
    assert row["lambda_pivot_nm"] == pytest.approx(427.0, abs=0.5)
    assert row["fwhm_nm"] == pytest.approx(48.0, abs=1.0)
    FilterRegistry(FilterInfo(*r) for r in tbl.iterrows())


def test_filter_definitions_follow_registry() -> None:
    from phot7ds import get_filter_definitions

    bands, widths, colors, _, _ = get_filter_definitions(unit="nm")
    assert list(bands)[:5] == ["u", "g", "r", "i", "z"]
    assert bands["m425w"] == pytest.approx(427.22)
    assert widths["m425w"] == pytest.approx(47.7 / 2)
    assert set(bands) == set(widths) == set(colors)


# --- header keys ----------------------------------------------------------
def test_header_keys_wide_and_duplicates() -> None:
    from phot7ds.filters import load_filter_registry

    reg = load_filter_registry()
    keys = reg.header_keys(["g", "m425", "m425w", "m425-1"])
    assert keys == {"g": "G", "m425": "425", "m425w": "42W", "m425-1": "425-1"}
    with pytest.raises(KeyError):
        reg.header_key("m329w")


def test_header_key_collision_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from phot7ds.filters import FilterInfo, FilterRegistry

    reg = FilterRegistry([
        FilterInfo("m425", "medium", 425.0, 24.0, "425"),
        FilterInfo("m426", "medium", 426.0, 24.0, "426"),
    ])
    assert reg.header_keys(["m425", "m426"]) == {"m425": "425", "m426": "426"}
    monkeypatch.setattr(FilterRegistry, "header_key", lambda self, band: "425")
    with pytest.raises(ValueError, match="same header key"):
        reg.header_keys(["m425", "m426"])


def test_zeropoint_and_depth_keys() -> None:
    from phot7ds.depth import depth_results_to_meta, zeropoints_to_meta

    meta: dict = {}
    zps = {("aper05", "g"): 25.0, ("aper05", "m400"): 24.0,
           ("aper05", "m425"): 24.1, ("aper05", "m425w"): 24.9,
           ("auto", "m425w"): 24.8}
    zeropoints_to_meta(meta, zps, {k: 0.01 for k in zps})
    for key in ("ZP05MG", "ZP05M400", "ZP05M425", "ZP05M42W", "ZPAUM42W", "ZE05M42W"):
        assert key in meta
    assert meta["ZP05M425"][0] == pytest.approx(24.1)
    assert meta["ZP05M42W"][0] == pytest.approx(24.9)

    depth = {("aper05", "m425w"): {"curve": {"depth": 21.0},
                                   "empty": {"depth": 21.2, "sky_sigma": 3.0}}}
    depth_results_to_meta(meta, depth)
    assert {"UL5EM42W", "UL5RM42W", "BRMSM42W"} <= set(meta)

    with pytest.raises(ValueError, match="both"):
        zeropoints_to_meta({}, {("aper05", "m425"): 24.0, ("aper05", "x"): 24.0},
                           band_keys={"m425": "425", "x": "425"})


# --- image selection ------------------------------------------------------
def test_select_images_header_first_and_drops(tmp_path) -> None:
    from phot7ds.filters import load_filter_registry
    from phot7ds.images import select_images_by_filter

    names = load_filter_registry().names
    imgs = [
        _image(tmp_path / "T00001_m425w_b.fits", "m425w"),
        _image(tmp_path / "T00001_m425w_a.fits", "m425w"),
        _image(tmp_path / "T00001_g_coadd.fits", "r"),          # header wins
        _image(tmp_path / "T00001_m329w_coadd.fits", "m329w"),  # not in registry
        _image(tmp_path / "T00001_m640_coadd.fits"),            # filename fallback
        _image(tmp_path / "T00001_foo_coadd.fits"),             # nothing
        _image(tmp_path / "T00001_m425_coadd.fits", "m425"),
    ]
    kept, dropped = select_images_by_filter(imgs, names, deduplicate=True)
    assert [(f.band, f.source) for f in kept] == [
        ("r", "header"), ("m425", "header"), ("m425w", "header"), ("m640", "filename"),
    ]
    assert kept[2].path.endswith("m425w_a.fits")
    reasons = {d["image"].rsplit("/", 1)[-1]: (d["reason"], d["filter"]) for d in dropped}
    assert reasons == {
        "T00001_m329w_coadd.fits": ("not_in_registry", "m329w"),
        "T00001_foo_coadd.fits": ("unknown_filter", None),
        "T00001_m425w_b.fits": ("duplicate", "m425w"),
    }

    kept_all, dropped_all = select_images_by_filter(imgs, names, deduplicate=False)
    assert [f.band for f in kept_all].count("m425w") == 2
    assert {d["reason"] for d in dropped_all} == {"not_in_registry", "unknown_filter"}


def test_select_measurement_images_manifest_and_errors(
    tmp_path, caplog: pytest.LogCaptureFixture,
) -> None:
    from phot7ds.filters import load_filter_registry
    from phot7ds.pipeline import _select_measurement_images

    reg = load_filter_registry()
    good = _image(tmp_path / "T00001_m692w_coadd.fits", "m692w")
    bad = _image(tmp_path / "T00001_m329w_coadd.fits", "m329w")
    with caplog.at_level(logging.WARNING, logger="phot7ds.pipeline"):
        paths, bands, dropped = _select_measurement_images(
            [bad, good], registry=reg, deduplicate=True,
        )
    assert paths == [good] and bands == ["m692w"]
    assert dropped == [{"image": bad, "filter": "m329w", "reason": "not_in_registry"}]
    assert "not in the filter registry" in caplog.text

    with pytest.raises(ValueError, match="filter registry"):
        _select_measurement_images([bad], registry=reg, deduplicate=True)


def test_band_names_use_registry_names(tmp_path) -> None:
    from phot7ds.images import extract_band_names_and_saturation

    imgs = [_image(tmp_path / "a.fits", "M425W"), _image(tmp_path / "b.fits", "m425w")]
    bands, _ = extract_band_names_and_saturation(imgs, bands=["m425w", "m425w"])
    assert bands == ["m425w", "m425w-1"]


def test_uncalibrated_bands() -> None:
    from phot7ds.pipeline import _uncalibrated_bands

    zps = {("aper05", "g"): 25.0, ("auto", "g"): 25.1, ("aper05", "m386"): float("nan")}
    assert _uncalibrated_bands(zps, ["g", "m386", "m438"], ["aper05", "auto"]) == {
        "m386": ["aper05", "auto"], "m438": ["aper05", "auto"],
    }


def test_medium_only_keeps_wide_bands() -> None:
    from phot7ds.detection.sevends import _is_broad_band

    assert _is_broad_band("g") and _is_broad_band("u") and _is_broad_band("z")
    assert not _is_broad_band("m425w")
    assert not _is_broad_band("m386")


# --- empty-aperture depth -------------------------------------------------
def test_empty_aperture_subtracts_mesh_background(tmp_path) -> None:
    from phot7ds.depth import depth_from_empty_apertures, mesh_background

    rng = np.random.default_rng(0)
    ny = nx = 1024
    yy, xx = np.mgrid[0:ny, 0:nx]
    sky = 0.5 * np.sin(xx / 200.0) * np.cos(yy / 260.0) + 0.0005 * xx
    data = (sky + rng.normal(0.0, 1.0, (ny, nx))).astype(np.float32)
    data[:, :40] = 0.0  # no-data strip
    path = tmp_path / "img.fits"
    fits.PrimaryHDU(data).writeto(str(path))

    bkg = mesh_background(data, data != 0, cell_size=128, smoothing_box_size=1)
    inner = (slice(150, 850), slice(150, 850))
    assert np.std(bkg(xx[inner], yy[inner]) - sky[inner]) < 0.03
    assert np.std(bkg(xx[:, 40:], yy[:, 40:]) - sky[:, 40:]) < 0.04

    r = 4.95
    out = depth_from_empty_apertures(
        str(path), r, zeropoint=25.0, n_apertures=1500, seed=1,
        background_cell_size=128, smoothing_box_size=1,
    )
    n_pix = 69  # pixels in an r = 4.95 disc
    assert out["sky_sigma"] == pytest.approx(np.sqrt(n_pix), rel=0.12)
    assert out["depth"] == pytest.approx(25.0 - 2.5 * np.log10(5 * np.sqrt(n_pix)), abs=0.15)


# --- output format --------------------------------------------------------
def test_resolve_output_format() -> None:
    from phot7ds.catalog_io import FITS_MAX_COLUMNS, resolve_output_format

    assert resolve_output_format(FITS_MAX_COLUMNS) == "fits"
    assert resolve_output_format(FITS_MAX_COLUMNS + 1) == "parquet"
    assert resolve_output_format(10, "parquet") == "parquet"
    with pytest.raises(ValueError, match="at most 999"):
        resolve_output_format(1000, "fits")
    with pytest.raises(ValueError, match="output_format"):
        resolve_output_format(10, "hdf5")


def _wide_table(n_cols: int) -> Table:
    from astropy.table import MaskedColumn

    tbl = Table()
    for j in range(n_cols):
        tbl[f"c{j:04d}"] = np.arange(3, dtype=np.float32) + j
    tbl["c0000"].unit = "mag"
    tbl["c0001"].description = "a described column"
    tbl["c0002"] = MaskedColumn([1.0, 2.0, 3.0], mask=[False, True, False])
    tbl.meta["ZP05M42W"] = (24.9, "ZP aper05 m425w [mag]")
    tbl.meta["PHOTVER"] = ("0.9.0", "phot7ds package version")
    return tbl


def test_write_catalog_auto_switches_to_parquet(tmp_path) -> None:
    from phot7ds.catalog_io import catalog_nrows, find_existing_catalog, read_catalog, write_catalog

    narrow = write_catalog(_wide_table(5), tmp_path / "cat.fits")
    assert narrow.suffix == ".fits" and narrow.exists()

    wide = write_catalog(_wide_table(1000), tmp_path / "cat.fits", remove_stale=True)
    assert wide.name == "cat.parquet"
    assert not narrow.exists()
    assert find_existing_catalog(tmp_path / "cat.fits") == wide
    assert catalog_nrows(wide) == 3

    back = read_catalog(wide)
    assert len(back.colnames) == 1000
    assert str(back["c0000"].unit) == "mag"
    assert back["c0001"].description == "a described column"
    assert list(back["c0002"].mask) == [False, True, False]
    assert back.meta["ZP05M42W"] == pytest.approx(24.9)
    assert read_catalog(wide, plain_meta=False).meta["ZP05M42W"][1] == "ZP aper05 m425w [mag]"

    with pytest.raises(ValueError, match="at most 999"):
        write_catalog(_wide_table(1000), tmp_path / "x.fits", "fits")

    forced = write_catalog(_wide_table(5), tmp_path / "small.fits", "parquet")
    assert forced.suffix == ".parquet"


def test_catalog_name_parquet_suffix(tmp_path) -> None:
    from phot7ds.pipeline import _normalize_catalog_path, _resolve_output_paths

    assert _normalize_catalog_path("/tmp/x.parquet").name == "x.parquet"
    _, raw, zp, _, _, run_name = _resolve_output_paths(
        output_dir=tmp_path, catalog_path=None, catalog_name="run.parquet",
        run_name=None, detection_image="/d/det.fits", detection_label="7DT",
    )
    assert zp.name == "run.parquet" and run_name == "run"
    assert raw.name == "run_raw.fits"


def test_load_unified_catalog_reads_parquet(tmp_path) -> None:
    from phot7ds import build_canonical_schema, load_unified_catalog, standardize_catalog
    from phot7ds.catalog_io import write_catalog
    from phot7ds.filters import DEFAULT_BANDS

    schema = build_canonical_schema(
        bands=DEFAULT_BANDS, apertures=("aper05", "aper10", "auto"),
        flux_fractions=(0.5, 0.9), per_band_masks=True,
    )
    cat = Table({"source_id": [1, 2], "aper05_mag_g": [20.0, 21.0]})
    out = standardize_catalog(cat, schema)
    assert len(out.colnames) > 999
    path = write_catalog(out, tmp_path / "unified.fits")
    assert path.suffix == ".parquet"
    back = load_unified_catalog(str(path))
    assert back.colnames == out.colnames
    assert np.isnan(back["aper05_mag_m425w"]).all()
