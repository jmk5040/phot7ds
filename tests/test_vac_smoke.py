"""
Smoke tests for the ``phot7ds.vac`` subpackage.

These avoid the heavy optional dependencies (``eazy``, ``sfdmap``,
``extinction``) and the FAST++ binary. They cover the ported helpers
(:func:`phot7ds.matching`, :func:`phot7ds.mag_to_flux`,
:func:`phot7ds.filter_colorization`) and :class:`VACConfig` validation.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from astropy.table import Table


# --- ported core helpers ------------------------------------------------
def test_matching_join_modes() -> None:
    from phot7ds import matching

    a = Table({"id": [1, 2, 3], "ra": [10.0, 20.0, 30.0], "dec": [0.0, 0.0, 0.0]})
    b = Table({"name": ["x", "y"], "RA": [10.00001, 20.00001], "DE": [0.0, 0.0]})

    inner = matching(a, b, a["ra"], a["dec"], b["RA"], b["DE"],
                     sep=2.0, join_type="inner", duplicate="closest", ref_prefix="ref_")
    assert len(inner) == 2

    left = matching(a, b, a["ra"], a["dec"], b["RA"], b["DE"],
                    sep=2.0, join_type="left", duplicate="closest", ref_prefix="ref_")
    assert len(left) == 3
    assert "ref_sep" in left.colnames
    assert int(np.sum(np.isfinite(left["ref_sep"].filled(np.nan)))) == 2


def test_matching_all_duplicate() -> None:
    from phot7ds import matching

    a = Table({"id": [1], "ra": [10.0], "dec": [0.0]})
    b = Table({"name": ["x", "y"], "RA": [10.00001, 10.00002], "DE": [0.0, 0.0]})
    allpairs = matching(a, b, a["ra"], a["dec"], b["RA"], b["DE"],
                        sep=2.0, join_type="inner", duplicate="all", ref_prefix="ref_")
    assert len(allpairs) == 2


def test_mag_to_flux_edge_cases() -> None:
    from phot7ds import AB2Jy, mag_to_flux, mag_to_flux_err

    fluxes = mag_to_flux([20.0, 99.0, -5.0])
    assert fluxes[0] == pytest.approx(AB2Jy(20.0, "FAST"), rel=1e-3)
    assert fluxes[1] == -99  # out of (5, 30) range
    assert fluxes[2] == -99

    errs = mag_to_flux_err([20.0, 99.0], [0.1, 0.1])
    assert errs[0] > 0
    assert errs[1] == -99

    masked = mag_to_flux(np.ma.array([20.0], mask=[True]))
    assert masked[0] == -99


def test_filter_colorization_bands() -> None:
    from phot7ds import filter_colorization

    bands, widths, colors, l2c, l2b = filter_colorization(unit="angstrom")
    assert len(bands) == 23  # g, r, i + 20 medium
    assert "g" in bands and "m400" in bands and "m875" in bands
    assert set(bands) == set(widths) == set(colors)


# --- VACConfig ----------------------------------------------------------
def test_vacconfig_filters_toggle() -> None:
    from phot7ds.vac import VACConfig

    cfg = VACConfig(lib_dir="/tmp/lib", catalog_dir="/tmp/cat", output_root="/tmp/out",
                    use_medium=True, use_broad=False, use_vhs=True, use_galex=True)
    filters = cfg.filters()
    assert "f_7DS_m400" in filters
    assert "f_FUV" in filters and "f_NUV" in filters
    assert "f_VHS_J" in filters
    assert all(not f.startswith("f_SDSS") for f in filters)

    cfg2 = VACConfig(lib_dir="/tmp/lib", catalog_dir="/tmp/cat", output_root="/tmp/out",
                     use_medium=False, use_broad=True, use_vhs=False, use_galex=False)
    assert cfg2.filters() == ["f_7DS_g", "f_7DS_r", "f_7DS_i", "f_W1", "f_W2"]


def test_vacconfig_validate_missing(tmp_path) -> None:
    from phot7ds.vac import VACConfig

    cfg = VACConfig(lib_dir=str(tmp_path / "nope"),
                    catalog_dir=str(tmp_path), output_root=str(tmp_path))
    with pytest.raises(FileNotFoundError):
        cfg.validate(require_fastpp=False)


def test_vacconfig_derived_paths() -> None:
    from phot7ds.vac import VACConfig

    cfg = VACConfig(lib_dir="/lib", catalog_dir="/cat", output_root="/out",
                    detection_ref="DELVE")
    assert str(cfg.sfd_path) == "/lib/sfddata"
    assert str(cfg.filters_res) == "/lib/FILTER.RES.latest"
    assert str(cfg.photoz_dir("T1")) == "/out/eazy/T1"
    assert str(cfg.regalade_path("T1")).endswith("T1_regalade.fits")


# --- lazy import surface ------------------------------------------------
def test_vac_import_does_not_require_extras() -> None:
    import importlib

    mod = importlib.import_module("phot7ds.vac")
    assert hasattr(mod, "run_value_added")
    assert hasattr(mod, "VACConfig")
    assert hasattr(mod, "build_galaxy_catalog")
    assert hasattr(mod, "detect_filters")
    assert hasattr(mod, "ensure_external_catalog")
    assert hasattr(mod, "write_run_log")


# --- new VAC behavior ---------------------------------------------------
def test_vacconfig_prior_defaults() -> None:
    from phot7ds.vac import VACConfig
    from phot7ds.vac.config import PACKAGED_PRIOR

    cfg = VACConfig(lib_dir="/lib", catalog_dir="/cat", output_root="/out")
    assert cfg.prior_band == "m625"
    # The default prior ships with the package, so it does not depend on
    # which prior happens to sit in the user's LIB tree.
    assert cfg.prior_path == PACKAGED_PRIOR
    assert cfg.auto_download is False

    cfg2 = VACConfig(lib_dir="/lib", catalog_dir="/cat", output_root="/out",
                     prior_band="r", prior_file="/lib/templates/prior_R_extend.dat")
    assert str(cfg2.prior_path).endswith("prior_R_extend.dat")


def test_packaged_prior_is_installed_and_readable() -> None:
    import numpy as np

    from phot7ds.vac.config import PACKAGED_PRIOR

    assert PACKAGED_PRIOR.is_file(), f"packaged prior missing: {PACKAGED_PRIOR}"
    arr = np.loadtxt(PACKAGED_PRIOR)
    # z column + one p(z|m) column per magnitude bin in the header.
    kbins = PACKAGED_PRIOR.read_text().splitlines()[0].split()[2:]
    assert arr.shape[1] == len(kbins) + 1
    assert np.all(arr[:, 1:] >= 0)


def test_vacconfig_binaries_are_discovered_not_hardcoded(monkeypatch) -> None:
    """No machine-specific default paths: env var, then $PATH, then None."""
    from phot7ds.vac import VACConfig

    monkeypatch.setenv("PHOT7DS_FASTPP_BIN", "/somewhere/fast++")
    monkeypatch.setenv("PHOT7DS_EAZY_BIN", "/somewhere/eazy")
    cfg = VACConfig(lib_dir="/lib", catalog_dir="/cat", output_root="/out",
                    photoz_engine="eazy-py")
    assert str(cfg.fastpp_bin) == "/somewhere/fast++"
    assert str(cfg.eazy_bin) == "/somewhere/eazy"

    monkeypatch.delenv("PHOT7DS_FASTPP_BIN")
    monkeypatch.delenv("PHOT7DS_EAZY_BIN")
    monkeypatch.setattr("shutil.which", lambda name: None)
    cfg2 = VACConfig(lib_dir="/lib", catalog_dir="/cat", output_root="/out",
                     photoz_engine="eazy-py")
    assert cfg2.fastpp_bin is None
    assert cfg2.eazy_bin is None
    assert cfg2.fastpp_bin_ok() is False
    assert cfg2.eazy_bin_ok() is False
    # An unset binary is reported as unconfigured rather than crashing on a
    # None path while deriving fastpp_share.
    assert cfg2.fastpp_share is None
    problems = cfg2.check_requirements(do_photoz=False, do_sedfit=True,
                                       deep=False)
    assert any("FAST++ binary" in p and "not configured" in p for p in problems)

    # An explicit value always wins over the environment.
    monkeypatch.setenv("PHOT7DS_FASTPP_BIN", "/from/env")
    cfg3 = VACConfig(lib_dir="/lib", catalog_dir="/cat", output_root="/out",
                     fastpp_bin="/explicit/fast++", photoz_engine="eazy-py")
    assert str(cfg3.fastpp_bin) == "/explicit/fast++"


def test_external_enabled_respects_toggles() -> None:
    from phot7ds.vac import VACConfig
    from phot7ds.vac.fluxes import _external_enabled

    cfg = VACConfig(lib_dir="/lib", catalog_dir="/cat", output_root="/out",
                    use_vhs=False, use_galex=True, use_wise=True)
    assert _external_enabled("f_VHS_J", cfg) is False
    assert _external_enabled("f_NUV", cfg) is True
    assert _external_enabled("f_W1", cfg) is True


def test_vizier_preset_mapping() -> None:
    from phot7ds.vac.vizier import CATALOG_PRESETS

    for key in ("regalade", "vhs", "vhs_ap6", "galex", "catwise"):
        assert key in CATALOG_PRESETS
        assert CATALOG_PRESETS[key].key == key
        assert CATALOG_PRESETS[key].vizier_id  # non-empty VizieR id
    assert CATALOG_PRESETS["regalade"].vizier_id == "J/A+A/706/A284/regalade"
    assert CATALOG_PRESETS["catwise"].vizier_id == "II/365/catwise"

    # The ap6 staging must be a superset of the plain VHS one, so a single
    # matched table can serve both magnitude sets.
    plain = set(CATALOG_PRESETS["vhs"].columns or ())
    ap6 = set(CATALOG_PRESETS["vhs_ap6"].columns or ())
    assert plain <= ap6
    assert {"Japc6", "Hapc6", "Ksapc6"} <= ap6


# --- VHS magnitude sets -------------------------------------------------
def test_vhs_mag_set_selects_columns_and_staging() -> None:
    from phot7ds.vac import VACConfig, resolve_vhs_mag_set
    from phot7ds.vac.crossmatch import VHS_VEGA_TO_AB
    from phot7ds.vac.fluxes import external_columns
    from phot7ds.vac.vizier import vhs_preset_key

    # A fixed 7DS aperture pairs with the VHS 5.7" aperture; total-light
    # magnitudes pair with the Petrosian ones.
    assert resolve_vhs_mag_set("aper05c") == "ap6"
    assert resolve_vhs_mag_set("autoc") == "petro"
    assert resolve_vhs_mag_set("auto") == "petro"

    petro = VACConfig(lib_dir="/lib", catalog_dir="/cat", output_root="/out",
                      photoz_engine="eazy-py")
    assert external_columns(petro)["f_VHS_J"] == ("vhs_Jpmag", "vhs_e_Jpmag")
    assert petro.vhs_staging[0] == "vhs_dr5"
    assert vhs_preset_key(petro) == "vhs"

    ap6 = VACConfig(lib_dir="/lib", catalog_dir="/cat", output_root="/out",
                    vhs_mag_set="ap6", photoz_engine="eazy-py")
    assert external_columns(ap6)["f_VHS_J"] == ("vhs_Japc6", "vhs_e_Jap6")
    # ap6 columns are absent from VizieR's default set, so they are staged
    # separately rather than overwriting the shared VHS files.
    assert ap6.vhs_staging[0] == "vhs_dr5_ap6"
    assert str(ap6.vhs_path("T1")).endswith("vhs_dr5_ap6/T1_vhs_dr5_ap6.fits")
    assert vhs_preset_key(ap6) == "vhs_ap6"

    # The staging has to follow the magnitude set through dataclasses.replace,
    # or a split run looks for ap6 columns in the Petrosian files.
    back = replace(ap6, vhs_mag_set="petro")
    assert str(back.vhs_path("T1")).endswith("vhs_dr5/T1_vhs_dr5.fits")
    assert external_columns(back)["f_VHS_J"] == ("vhs_Jpmag", "vhs_e_Jpmag")

    # An explicit staging location is respected, and pins both sets to it.
    custom = VACConfig(lib_dir="/lib", catalog_dir="/cat", output_root="/out",
                       vhs_mag_set="ap6", vhs_subdir="mine",
                       vhs_template="{tile}.fits", photoz_engine="eazy-py")
    assert str(custom.vhs_path("T1")).endswith("mine/T1.fits")
    assert str(replace(custom, vhs_mag_set="petro").vhs_path("T1")).endswith(
        "mine/T1.fits")

    # Both magnitude sets need Vega->AB offsets, or the NIR points silently
    # stay on the Vega scale.
    for mcol, _err in external_columns(ap6).values():
        if mcol.startswith("vhs_"):
            assert mcol[len("vhs_"):] in VHS_VEGA_TO_AB

    with pytest.raises(ValueError):
        VACConfig(lib_dir="/lib", catalog_dir="/cat", output_root="/out",
                  vhs_mag_set="nope", photoz_engine="eazy-py")


def test_split_match_prefers_staged_vhs(tmp_path) -> None:
    from phot7ds.vac import VACConfig
    from phot7ds.vac.pipeline import _match_vhs_mag_set

    def cfg(mag_set):
        return VACConfig(lib_dir="/lib", catalog_dir=str(tmp_path),
                         output_root="/out", vhs_mag_set=mag_set,
                         photoz_engine="eazy-py")

    ap6, petro = cfg("ap6"), cfg("petro")

    # Nothing staged: keep the preferred (superset) set so auto_download has
    # something to fetch.
    assert _match_vhs_mag_set(ap6, petro, "T1") == "ap6"

    # Only the Petrosian file is staged. Matching the absent ap6 file would
    # drop the NIR bands from both stages; matching petro keeps them for the
    # stage that asked for Petrosian magnitudes.
    petro_path = petro.vhs_path("T1")
    petro_path.parent.mkdir(parents=True, exist_ok=True)
    petro_path.write_text("placeholder")
    assert _match_vhs_mag_set(ap6, petro, "T1") == "petro"

    # Once ap6 is staged it wins again: it is a superset, so it serves both.
    ap6_path = ap6.vhs_path("T1")
    ap6_path.parent.mkdir(parents=True, exist_ok=True)
    ap6_path.write_text("placeholder")
    assert _match_vhs_mag_set(ap6, petro, "T1") == "ap6"

    # When neither stage wants ap6, the shared staging is used as-is.
    assert _match_vhs_mag_set(petro, petro, "T1") == "petro"


def test_reference_path_falls_back_to_generic_layout() -> None:
    from phot7ds.vac import VACConfig

    cfg = VACConfig(lib_dir="/lib", catalog_dir="/cat", output_root="/out",
                    photoz_engine="eazy-py")
    assert cfg.reference_path("regalade", "T1") == cfg.regalade_path("T1")
    assert cfg.reference_path("vhs", "T1") == cfg.vhs_path("T1")
    assert cfg.reference_path("galex", "T1") == cfg.galex_path("T1")
    assert str(cfg.reference_path("catwise", "T1")).endswith(
        "catwise/T1_catwise.fits")


# --- dead bands and the coverage cut -----------------------------------
def test_drop_dead_bands_removes_all_nan_columns() -> None:
    from phot7ds.vac import drop_dead_bands, live_bands

    n = 100
    cat = Table({
        "aper05c_mag_m625": np.full(n, 18.0),
        "aper05c_mag_err_m625": np.full(n, 0.05),
        # never observed on this tile: column exists but is entirely NaN
        "aper05c_mag_g": np.full(n, np.nan),
        "aper05c_mag_err_g": np.full(n, np.nan),
        # measured for a single source: below the live threshold
        "aper05c_mag_i": np.concatenate([[19.0], np.full(n - 1, np.nan)]),
        "aper05c_mag_err_i": np.concatenate([[0.1], np.full(n - 1, np.nan)]),
        "autoc_mag_m625": np.full(n, 17.5),
        "autoc_mag_err_m625": np.full(n, 0.04),
    })

    live, dead = live_bands(cat, "aper05c")
    assert live == ["m625"]
    assert sorted(dead) == ["g", "i"]

    prepared, live_by_aperture = drop_dead_bands(cat, ("aper05c", "autoc"))
    assert "aper05c_mag_g" not in prepared.colnames
    assert "aper05c_mag_err_g" not in prepared.colnames
    assert "aper05c_mag_i" not in prepared.colnames
    assert "aper05c_mag_m625" in prepared.colnames
    assert live_by_aperture == {"aper05c": ["m625"], "autoc": ["m625"]}
    # The input is left untouched.
    assert "aper05c_mag_g" in cat.colnames


def test_required_filter_count_modes() -> None:
    from phot7ds.vac import VACConfig
    from phot7ds.vac.fluxes import required_filter_count

    # 4 x 7DS + 2 external = 6 filters.
    flux = Table({"#id": [1]})
    for col in ("f_7DS_g", "f_7DS_r", "f_7DS_m625", "f_7DS_m675",
                "f_VHS_J", "f_NUV"):
        flux[col] = [1.0]

    plain = VACConfig(lib_dir="/lib", catalog_dir="/cat", output_root="/out",
                      min_filter_fraction=0.80, photoz_engine="eazy-py")
    # int(6 * 0.8) == 4, counted over every filter.
    assert required_filter_count(flux, plain) == (4, 6)

    # Against the 7DS bands alone: round(0.8 * 4) == 3, externals are a bonus.
    seven = VACConfig(lib_dir="/lib", catalog_dir="/cat", output_root="/out",
                      min_7ds_band_fraction=0.80, photoz_engine="eazy-py")
    assert required_filter_count(flux, seven) == (3, 6)

    # Never demand more filters than exist, nor fewer than one.
    greedy = VACConfig(lib_dir="/lib", catalog_dir="/cat", output_root="/out",
                       min_7ds_band_fraction=5.0, photoz_engine="eazy-py")
    assert required_filter_count(flux, greedy) == (6, 6)
    tiny = VACConfig(lib_dir="/lib", catalog_dir="/cat", output_root="/out",
                     min_7ds_band_fraction=0.0, photoz_engine="eazy-py")
    assert required_filter_count(flux, tiny) == (1, 6)


def test_validity_cut_keeps_sources_with_enough_bands() -> None:
    from phot7ds.vac import VACConfig
    from phot7ds.vac.fluxes import _apply_validity_cut

    flux = Table({
        "#id": [1, 2, 3],
        "f_7DS_g": [1.0, 1.0, -99.0],
        "e_7DS_g": [0.1, 0.1, -99.0],
        "f_7DS_r": [1.0, 1.0, -99.0],
        "e_7DS_r": [0.1, 0.1, -99.0],
        "f_7DS_m625": [1.0, -99.0, -99.0],
        "e_7DS_m625": [0.1, -99.0, -99.0],
        "f_7DS_m675": [1.0, -99.0, 1.0],
        "e_7DS_m675": [0.1, -99.0, 0.1],
    })
    cfg = VACConfig(lib_dir="/lib", catalog_dir="/cat", output_root="/out",
                    min_filter_fraction=0.75, photoz_engine="eazy-py")
    kept, mask = _apply_validity_cut(flux, cfg)
    # int(4 * 0.75) == 3 measured filters required.
    assert list(kept["#id"]) == [1]
    assert list(mask) == [True, False, False]


@pytest.mark.parametrize("use_zspec", [False, True])
def test_regalade_zspec_toggle(monkeypatch, use_zspec) -> None:
    """z_spec is written only when use_regalade_zspec is on, from spec codes only."""
    from phot7ds.vac import VACConfig
    from phot7ds.vac import fluxes

    monkeypatch.setattr(fluxes, "detect_filters", lambda magtbl, cfg: ["f_7DS_m625"])
    monkeypatch.setattr(fluxes, "_central_wavelengths", lambda cfg, f: {"f_7DS_m625": 6250.0})
    monkeypatch.setattr(fluxes, "_tile_center", lambda tile_info: (0.0, 0.0))
    monkeypatch.setattr(fluxes, "_extinction_by_filter", lambda cfg, lc, ra, dec: {"f_7DS_m625": 0.0})

    magtbl = Table({
        "aper05c_mag_m625": [18.0, 18.0, 18.0, 18.0],
        "aper05c_mag_err_m625": [0.05, 0.05, 0.05, 0.05],
        # spec code, photometric code, spec code with no z, spec code
        "regalade_r_DistInput": [4, 1, 7, 0],
        "regalade_z": [0.05, 0.10, np.nan, 0.02],
    })
    cfg = VACConfig(lib_dir="/lib", catalog_dir="/cat", output_root="/out",
                    photoz_engine="eazy-py", use_regalade_zspec=use_zspec)
    flux, _, info = fluxes.build_flux_catalog(magtbl, cfg, "T1", None, write=False)

    if use_zspec:
        assert list(flux["z_spec"]) == [0.05, -1.0, -1.0, 0.02]
        assert info["n_zspec"] == 2
    else:
        assert "z_spec" not in flux.colnames
        assert info["n_zspec"] == 0


def test_regalade_zspec_is_off_by_default() -> None:
    from phot7ds.vac import VACConfig

    cfg = VACConfig(lib_dir="/lib", catalog_dir="/cat", output_root="/out",
                    photoz_engine="eazy-py")
    assert cfg.use_regalade_zspec is False
    assert cfg.regalade_spec_codes == (4, 5, 6, 7, 8, 0, 2)


# --- tile query geometry ------------------------------------------------
def test_tile_query_box_deprojects_ra_span() -> None:
    from phot7ds.vac.vizier import tile_query_box

    # A ~1.4 deg wide tile at dec = -83 spans a much larger raw RA range;
    # querying VizieR with the raw span pulls in far too much sky.
    dec = -83.0
    half_width = 0.7 / np.cos(np.radians(dec))
    tile = Table({
        "ra1": [180.0 - half_width], "ra2": [180.0 + half_width],
        "ra3": [180.0 + half_width], "ra4": [180.0 - half_width],
        "dec1": [dec - 0.5], "dec2": [dec - 0.5],
        "dec3": [dec + 0.5], "dec4": [dec + 0.5],
    })
    ra_c, dec_c, width, height = tile_query_box(tile)
    assert ra_c == pytest.approx(180.0)
    assert dec_c == pytest.approx(dec, abs=0.05)
    # Angular width of the wider (equatorward) edge: 1.4 * cos(82.5)/cos(83).
    assert width == pytest.approx(1.5, abs=0.01)
    assert height == pytest.approx(1.0, abs=0.01)
    # The raw span is many times the angular width at this declination.
    assert 2 * half_width > 5 * width

    _, _, padded_w, padded_h = tile_query_box(tile, margin_deg=0.05)
    assert padded_w == pytest.approx(width + 0.05)
    assert padded_h == pytest.approx(height + 0.05)


def test_tile_query_box_across_ra_zero() -> None:
    """A tile straddling RA 0 is queried at RA ~0 with its true width, not at RA 180."""
    from phot7ds.vac.vizier import tile_query_box

    tile = {"ra1": 359.3, "ra2": 0.7, "ra3": 0.7, "ra4": 359.3,
            "dec1": -0.5, "dec2": -0.5, "dec3": 0.5, "dec4": 0.5}
    ra_c, dec_c, width, height = tile_query_box(tile)
    assert min(ra_c, 360.0 - ra_c) == pytest.approx(0.0, abs=1e-9)
    assert dec_c == pytest.approx(0.0, abs=1e-9)
    assert width == pytest.approx(1.4, abs=1e-3)
    assert height == pytest.approx(1.0, abs=1e-3)


def test_time_limit_raises_and_restores_handler() -> None:
    import signal
    import time

    from phot7ds.vac.vizier import time_limit

    before = signal.getsignal(signal.SIGALRM)
    with pytest.raises(TimeoutError):
        with time_limit(1):
            time.sleep(3)
    assert signal.getsignal(signal.SIGALRM) is before

    # A block that finishes in time cancels the alarm.
    with time_limit(5):
        pass
    assert signal.getsignal(signal.SIGALRM) is before


def test_prefetch_references_skips_staged_and_reports(tmp_path, monkeypatch) -> None:
    from phot7ds.vac import VACConfig, prefetch_references
    from phot7ds.vac import vizier as vac_vizier

    tile_table = Table({
        "tile": ["T1"],
        "ra1": [179.5], "ra2": [180.5], "ra3": [180.5], "ra4": [179.5],
        "dec1": [-0.5], "dec2": [-0.5], "dec3": [0.5], "dec4": [0.5],
    })
    cfg = VACConfig(lib_dir="/lib", catalog_dir=str(tmp_path), output_root="/out",
                    auto_download=True, vizier_attempts=2,
                    photoz_engine="eazy-py")

    # GALEX is already staged; VHS is not and its download keeps failing.
    galex = cfg.galex_path("T1")
    galex.parent.mkdir(parents=True, exist_ok=True)
    galex.write_text("placeholder")

    attempts = {"n": 0}

    def _boom(*args, **kwargs):
        attempts["n"] += 1
        raise RuntimeError("vizier is down")

    monkeypatch.setattr(vac_vizier, "download_catalog_for_tile", _boom)
    staged = prefetch_references(["T1", "T_absent"], tile_table, cfg,
                                 keys=("vhs", "galex"))

    assert staged["T1"]["galex"] is True
    assert staged["T1"]["vhs"] is False
    # A failed optional reference is retried, then reported and skipped
    # rather than raising and taking the whole batch down.
    assert attempts["n"] == cfg.vizier_attempts
    assert staged["T_absent"] == {}


def test_ensure_external_catalog_noop_when_present(tmp_path) -> None:
    from phot7ds.vac import VACConfig, ensure_external_catalog

    existing = tmp_path / "T1_regalade.fits"
    existing.write_text("placeholder")
    cfg = VACConfig(lib_dir="/lib", catalog_dir=str(tmp_path), output_root="/out",
                    auto_download=False)
    assert ensure_external_catalog("regalade", "T1", None, existing, cfg) is True

    missing = tmp_path / "T1_vhs_dr5.fits"
    # auto_download disabled -> stays absent, returns False (no download attempt)
    assert ensure_external_catalog("vhs", "T1", None, missing, cfg) is False


def test_write_run_log_contents(tmp_path) -> None:
    from phot7ds.vac import VACConfig, write_run_log

    cfg = VACConfig(lib_dir="/lib", catalog_dir="/cat", output_root=str(tmp_path),
                    detection_ref="7DS")
    log_path = write_run_log(
        cfg, "T1", tmp_path / "T1_7DS_vac.log",
        flux_info={"filters": ["f_7DS_g", "f_7DS_m625"],
                   "lambda_c": {"f_7DS_g": 4711.0, "f_7DS_m625": 6247.0},
                   "extinction": {"f_7DS_g": 0.12, "f_7DS_m625": 0.08},
                   "n_input": 10, "n_pass": 8, "min_filter_fraction": 0.8,
                   "error_margin": 0.03, "aperture": "aper05c"},
        eazy_info={"apply_prior": True, "prior_band": "m625",
                   "zphot_column": "z_m2", "n_targets": 8, "params": {}},
        fastpp_info={"name_zphot": "z_m2", "n_fits": 8, "params": {}},
    )
    text = open(log_path).read()
    assert "z_m2" in text
    assert "prior_band" in text
    assert "f_7DS_m625" in text
