"""
Smoke tests for the ``phot7ds.vac`` subpackage.

These avoid the heavy optional dependencies (``eazy``, ``sfdmap``,
``extinction``) and the FAST++ binary. They cover the ported helpers
(:func:`phot7ds.matching`, :func:`phot7ds.mag_to_flux`,
:func:`phot7ds.filter_colorization`) and :class:`VACConfig` validation.
"""
from __future__ import annotations

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

    cfg = VACConfig(lib_dir="/lib", catalog_dir="/cat", output_root="/out")
    assert cfg.prior_band == "m625"
    assert str(cfg.prior_path) == "/lib/templates/prior_m6250_extend.dat"
    assert cfg.auto_download is False

    cfg2 = VACConfig(lib_dir="/lib", catalog_dir="/cat", output_root="/out",
                     prior_band="r", prior_file="/lib/templates/prior_R_extend.dat")
    assert str(cfg2.prior_path).endswith("prior_R_extend.dat")


def test_external_enabled_respects_toggles() -> None:
    from phot7ds.vac import VACConfig
    from phot7ds.vac.fluxes import _external_enabled

    cfg = VACConfig(lib_dir="/lib", catalog_dir="/cat", output_root="/out",
                    use_vhs=False, use_galex=True, use_wise=True)
    assert _external_enabled("f_VHS_J", cfg) is False
    assert _external_enabled("f_NUV", cfg) is True
    assert _external_enabled("f_W1", cfg) is True


def test_vizier_preset_mapping() -> None:
    from phot7ds.vac.vizier import _PRESET_KEYS

    assert _PRESET_KEYS["regalade"] == "regalade"
    assert _PRESET_KEYS["vhs"] == "vhs"
    assert _PRESET_KEYS["galex"] == "galex"


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
