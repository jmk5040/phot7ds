"""
Configuration for the value-added catalog (VAC) pipeline.

:class:`VACConfig` bundles every path and tuning knob used by
:func:`phot7ds.vac.run_value_added`. Required external files are validated
up front (see :meth:`VACConfig.validate`) with helpful error messages,
mirroring :mod:`phot7ds.config_io`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Default 7DS medium-band set (m400..m875, 25 nm steps).
DEFAULT_MEDIUM_BANDS: tuple[str, ...] = tuple(f"m{w}" for w in range(400, 900, 25))
DEFAULT_BROAD_BANDS: tuple[str, ...] = ("g", "r", "i")


@dataclass
class VACConfig:
    """Paths and knobs for the value-added catalog pipeline.

    Parameters
    ----------
    lib_dir
        EAzY/FAST++ ``LIB`` tree (templates, ``FILTER.RES.latest``,
        priors, ``default.translate``, parameter templates).
    sfd_dir
        Directory of SFD dust maps (``SFD_dust_4096_*.fits``). Defaults to
        ``{lib_dir}/sfddata``.
    catalog_dir
        Root holding external reference catalogs (REGALADE/VHS/GALEX),
        each in a named subdirectory.
    output_root
        Root for per-tile VAC products (photo-z, SED, value-added catalog).
    fastpp_bin
        Path to the compiled ``fast++`` executable.
    detection_ref
        Detection-image tag used in file names (e.g. ``DELVE``, ``7DS``).
    aperture
        Catalog aperture prefix used for the SED fluxes (e.g. ``aper05c``).
    cover_flag_column
        Column flagging bad coverage; rows with non-zero values are dropped.
    medium_bands, broad_bands
        Band lists; ``use_medium``/``use_broad`` toggle inclusion.
    match_radius_arcsec
        Cross-match radius for all reference catalogs.
    error_margin
        Magnitude error floor added in quadrature-free fashion to all bands.
    min_filter_fraction
        Minimum fraction of measured filters required to keep a source.
    z_min, z_max, z_step
        Redshift grid shared by EAzY and FAST++.
    n_proc
        Worker count for eazy-py / FAST++.
    eazy_params, fastpp_params
        Optional override dicts merged over the built-in parameter sets.
    """

    # Required config tree / binaries
    lib_dir: str | Path
    catalog_dir: str | Path
    output_root: str | Path
    fastpp_bin: str | Path = "/home/jmkastro/fastpp/bin/fast++"
    sfd_dir: str | Path | None = None

    # Naming / catalog columns
    detection_ref: str = "DELVE"
    aperture: str = "aper05c"
    ra_column: str = "world_centroid_alpha"
    dec_column: str = "world_centroid_delta"
    id_column: str = "source_id"
    cover_flag_column: str = "isophotal_image_flags_cover"

    # External reference catalogs (subdir name + filename template, RA/Dec keys)
    regalade_subdir: str = "regalade"
    regalade_template: str = "{tile}_regalade.fits"
    regalade_ra_key: str = "RAJ2000"
    regalade_dec_key: str = "DEJ2000"
    regalade_name_key: str = "Name"
    regalade_mag_key: str = "rmag"
    vhs_subdir: str = "vhs_dr5"
    vhs_template: str = "{tile}_vhs_dr5.fits"
    galex_subdir: str = "galex"
    galex_template: str = "{tile}_galex_ais.fits"

    # Bands
    medium_bands: tuple[str, ...] = DEFAULT_MEDIUM_BANDS
    broad_bands: tuple[str, ...] = DEFAULT_BROAD_BANDS
    use_medium: bool = True
    use_broad: bool = False
    use_vhs: bool = True
    use_galex: bool = True
    use_wise: bool = True  # WISE bands come bundled in the REGALADE columns

    # Matching / flux assembly
    match_radius_arcsec: float = 2.0
    error_margin: float = 0.03
    min_filter_fraction: float = 0.80
    dedup_by_brightness: bool = True

    # Redshift grid
    z_min: float = 0.01
    z_max: float = 1.0
    z_step: float = 0.001

    # Execution
    n_proc: int = 8
    eazy_params: dict[str, Any] = field(default_factory=dict)
    fastpp_params: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Derived paths
    # ------------------------------------------------------------------
    @property
    def lib_path(self) -> Path:
        return Path(self.lib_dir)

    @property
    def sfd_path(self) -> Path:
        return Path(self.sfd_dir) if self.sfd_dir else self.lib_path / "sfddata"

    @property
    def filters_res(self) -> Path:
        return self.lib_path / "FILTER.RES.latest"

    @property
    def filters_res_info(self) -> Path:
        return self.lib_path / "FILTER.RES.latest.info"

    @property
    def translate_file(self) -> Path:
        return self.lib_path / "default.translate"

    @property
    def eazy_param_template(self) -> Path:
        return self.lib_path / "zphot.param"

    @property
    def fastpp_param_template(self) -> Path:
        return self.lib_path / "fastpp.param"

    def filters(self) -> list[str]:
        """Return the EAzY-style filter names selected by the toggles."""
        names: list[str] = []
        if self.use_galex:
            names += ["f_FUV", "f_NUV"]
        if self.use_medium:
            names += [f"f_7DS_{b}" for b in self.medium_bands]
        if self.use_broad:
            names += [f"f_SDSS_{b}" for b in self.broad_bands]
        if self.use_vhs:
            names += [f"f_VHS_{b}" for b in ("J", "H", "K")]
        if self.use_wise:
            names += ["f_W1", "f_W2"]
        return names

    def photoz_dir(self, tile: str) -> Path:
        return Path(self.output_root) / "eazy" / tile

    def sedfit_dir(self, tile: str) -> Path:
        return Path(self.output_root) / "fastpp" / tile

    def vac_dir(self) -> Path:
        return Path(self.output_root) / "value_added"

    def regalade_path(self, tile: str) -> Path:
        return Path(self.catalog_dir) / self.regalade_subdir / self.regalade_template.format(tile=tile)

    def vhs_path(self, tile: str) -> Path:
        return Path(self.catalog_dir) / self.vhs_subdir / self.vhs_template.format(tile=tile)

    def galex_path(self, tile: str) -> Path:
        return Path(self.catalog_dir) / self.galex_subdir / self.galex_template.format(tile=tile)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self, *, require_fastpp: bool = True) -> None:
        """Check that required config files / binaries exist.

        Raises :class:`FileNotFoundError` with a helpful message when a
        required input is missing. External per-tile catalogs are checked
        lazily during the run (REGALADE required, VHS/GALEX optional).
        """
        required = [
            (self.lib_path, "EAzY/FAST++ LIB directory"),
            (self.filters_res, "EAzY FILTER.RES.latest"),
            (self.filters_res_info, "EAzY FILTER.RES.latest.info"),
            (self.translate_file, "EAzY default.translate"),
            (self.eazy_param_template, "EAzY zphot.param template"),
            (self.sfd_path, "SFD dust map directory"),
        ]
        for path, what in required:
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"{what} not found at {path}. Set VACConfig.lib_dir / sfd_dir "
                    "to the directory holding your EAzY/FAST++ LIB tree."
                )
        if require_fastpp and not Path(self.fastpp_bin).exists():
            raise FileNotFoundError(
                f"FAST++ binary not found at {self.fastpp_bin}. Install FAST++ "
                "and set VACConfig.fastpp_bin, or run with do_sedfit=False."
            )


__all__ = ["VACConfig", "DEFAULT_MEDIUM_BANDS", "DEFAULT_BROAD_BANDS"]
