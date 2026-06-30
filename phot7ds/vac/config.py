"""
Configuration for the value-added catalog (VAC) pipeline.

:class:`VACConfig` bundles every path and tuning knob used by
:func:`phot7ds.vac.run_value_added`. Required external files are validated
up front (see :meth:`VACConfig.validate`) with helpful error messages,
mirroring :mod:`phot7ds.config_io`.
"""
from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

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
        Legacy band lists. The 7DS filters that enter the fit are now
        auto-detected from the catalog's ``<aperture>_mag_*`` columns, so
        ``use_medium`` / ``use_broad`` are ignored (kept for back-compat).
        ``medium_bands`` is still used to pick the dedup reference band.
    auto_download
        When True, fetch absent REGALADE/VHS/GALEX catalogs via VizieR
        (``vizier_query_path``) before matching.
    prior_band, prior_file
        Magnitude-prior band (default ``m625``) and prior table (default
        ``templates/prior_m6250_extend.dat``).
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
    fastpp_share_dir: str | Path | None = None
    fastpp_library_dir: str | Path | None = None
    sfd_dir: str | Path | None = None
    # Compiled EAzY binary (used when photoz_engine == "binary").
    eazy_bin: str | Path = "/data/data1/7DS/RIS/config/eazy/src/eazy"

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
    #
    # The 7DS filters that enter the SED fit are auto-detected from the
    # catalog's ``{aperture}_mag_*`` columns (see fluxes.build_flux_catalog),
    # so ``use_medium`` / ``use_broad`` / ``medium_bands`` / ``broad_bands``
    # are NOT used to select 7DS bands. ``medium_bands`` is kept only as the
    # reference list for the dedup band; ``use_*`` toggles below still gate
    # the external-survey joins.
    medium_bands: tuple[str, ...] = DEFAULT_MEDIUM_BANDS
    broad_bands: tuple[str, ...] = DEFAULT_BROAD_BANDS
    use_medium: bool = True  # deprecated: kept for back-compat, ignored
    use_broad: bool = False  # deprecated: kept for back-compat, ignored
    use_vhs: bool = True
    use_galex: bool = True
    use_wise: bool = True  # WISE bands come bundled in the REGALADE columns

    # External-catalog auto-download (VizieR). When a per-tile reference
    # catalog is absent at its expected path and ``auto_download`` is True,
    # it is fetched via the Query/Vizier_Query.py helper before matching.
    auto_download: bool = False
    vizier_query_path: str | Path = "/data/data1/7DS/RIS/script/Query/Vizier_Query.py"

    # Magnitude prior (eazy). Default is the 7DS m625 prior; when the prior
    # band's flux is absent the run proceeds without a prior (see photoz.py).
    prior_band: str = "m625"
    prior_file: str | Path | None = None  # default: templates/prior_m6250_extend.dat

    # Matching / flux assembly
    match_radius_arcsec: float = 2.0
    error_margin: float = 0.03
    min_filter_fraction: float = 0.80
    dedup_by_brightness: bool = True

    # Redshift grid
    z_min: float = 0.01
    z_max: float = 1.0
    z_step: float = 0.001
    z_step_type: int = 1  # 0 = Z_STEP, 1 = Z_STEP*(1+z) (log; matches eazy)

    # FAST++ SED-fit grid / performance knobs. Defaults reproduce the fast,
    # photo-z-anchored configuration of the legacy RIS_catalog_fastpp.py; the
    # raw LIB/fastpp.param template ships much slower defaults (RESOLUTION
    # 'hr', FORCE_ZPHOT 0), which is why an un-tuned vac run is far slower than
    # the original. Override any of these (or anything else) via fastpp_params.
    fastpp_resolution: str = "lr"          # 'pr'/'lr'/'hr'; 'hr' is far slower
    fastpp_force_zphot: bool = True        # fit only at the EAzY photo-z
    fastpp_parallel: str = "generators"    # suits many models + few sources
    fastpp_metal: tuple[float, ...] = (0.004, 0.008, 0.02, 0.05)

    # Photo-z engine. "binary" shells out to the compiled EAzY executable
    # (``eazy_bin``) and is the default: the pure-Python eazy-py TemplateGrid
    # build is pathologically slow for the 7DS medium-band filter set (it can
    # take longer to integrate a single template through the ~27 filters x
    # redshift grid than the binary takes to fit the whole catalog). Set to
    # "eazy-py" only if you specifically need the pure-Python path.
    photoz_engine: str = "binary"  # "binary" | "eazy-py"

    # Execution
    n_proc: int = 8  # FAST++ threads (and default object count is small)
    # eazy-py parallelism. Its TemplateGrid (n_proc<0 == serial) and
    # fit_catalog (n_proc==0 == serial) use *inconsistent* conventions, and
    # the multiprocessing template-grid build is prone to fork deadlocks
    # (it hangs then raises multiprocessing TimeoutError). Default <=0 runs
    # both stages serially, which is robust and fast for the small VAC
    # catalogs. Set >0 only if you know the parallel build works for you.
    eazy_n_proc: int = -1
    eazy_params: dict[str, Any] = field(default_factory=dict)
    fastpp_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Early, non-fatal heads-up: when the (default) binary photo-z engine
        # is selected but no executable is found at ``eazy_bin``, warn now so
        # the user can fix the path before a long run reaches the photo-z
        # stage. ``validate(require_eazy_bin=True)`` still hard-fails later.
        if self.photoz_engine == "binary" and not self.eazy_bin_ok():
            warnings.warn(
                f"VACConfig.photoz_engine='binary' but no EAzY executable was "
                f"found at eazy_bin={self.eazy_bin!s}. Build/point to the "
                f"compiled 'eazy' binary, or set photoz_engine='eazy-py'.",
                stacklevel=2,
            )

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
    def fastpp_share(self) -> Path:
        """FAST++ ``share`` tree (template error fn + SPS libraries).

        Defaults to ``<fastpp_bin>/../../share`` (the standard install
        layout) when not set explicitly.
        """
        if self.fastpp_share_dir:
            return Path(self.fastpp_share_dir)
        return Path(self.fastpp_bin).resolve().parent.parent / "share"

    @property
    def fastpp_libraries(self) -> Path:
        """BC03/SPS ``ised`` library tree for FAST++.

        Defaults to ``{lib_dir}/Libraries`` (where the 7DS config tree keeps
        the Bruzual & Charlot models) rather than the FAST++ install share,
        which is often left empty.
        """
        if self.fastpp_library_dir:
            return Path(self.fastpp_library_dir)
        return self.lib_path / "Libraries"

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
    def prior_path(self) -> Path:
        """eazy magnitude-prior file (default: 7DS m625 prior)."""
        if self.prior_file:
            return Path(self.prior_file)
        return self.lib_path / "templates" / "prior_m6250_extend.dat"

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
            names += [f"f_7DS_{b}" for b in self.broad_bands]
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
    # Validation / preflight
    # ------------------------------------------------------------------
    def eazy_bin_ok(self) -> bool:
        """True when ``eazy_bin`` exists and is an executable file."""
        path = str(self.eazy_bin)
        return os.path.isfile(path) and os.access(path, os.X_OK)

    def _requirement_list(
        self, *, do_photoz: bool, do_sedfit: bool, deep: bool
    ) -> list[tuple[str, Path, bool, str]]:
        """Every required input as ``(label, path, ok, kind)`` tuples.

        ``kind`` is ``"dir"``, ``"file"`` or ``"exe"``; ``ok`` reflects
        existence (and the executable bit for ``"exe"``). Per-tile external
        catalogs (REGALADE/VHS/GALEX) are intentionally excluded — they are
        tile-specific and fetched/checked during the run.
        """
        reqs: list[tuple[str, Path, bool, str]] = []

        def add(label: str, path, kind: str) -> None:
            p = Path(path)
            if kind == "dir":
                ok = p.is_dir()
            elif kind == "exe":
                ok = p.is_file() and os.access(str(p), os.X_OK)
            else:
                ok = p.is_file()
            reqs.append((label, p, ok, kind))

        # Always needed (config tree shared by both stages).
        add("LIB directory", self.lib_path, "dir")
        add("FILTER.RES.latest", self.filters_res, "file")
        add("FILTER.RES.latest.info", self.filters_res_info, "file")
        add("default.translate", self.translate_file, "file")
        add("SFD dust map directory", self.sfd_path, "dir")

        if do_photoz:
            tmpl = self.lib_path / "templates"
            add("EAzY zphot.param template", self.eazy_param_template, "file")
            add("EAzY magnitude prior", self.prior_path, "file")
            add("EAzY templates definition",
                tmpl / "eazy_v1.2_dusty.spectra.param", "file")
            add("EAzY wavelength grid",
                tmpl / "EAZY_v1.1_lines" / "lambda_v1.1.def", "file")
            add("EAzY template-error file", tmpl / "TEMPLATE_ERROR.eazy_v1.0", "file")
            add("EAzY IGM LAF coefficients", tmpl / "LAFcoeff.txt", "file")
            add("EAzY IGM DLA coefficients", tmpl / "DLAcoeff.txt", "file")
            if self.photoz_engine == "binary":
                add("EAzY binary (photoz_engine='binary')", self.eazy_bin, "exe")
            if deep:
                reqs += self._eazy_template_spectra_reqs(
                    tmpl / "eazy_v1.2_dusty.spectra.param"
                )

        if do_sedfit:
            add("FAST++ binary", self.fastpp_bin, "exe")
            add("FAST++ fastpp.param template", self.fastpp_param_template, "file")
            # NB: resolved from the fast++ install's share/ dir by default; a
            # mismatched install prefix is a common cross-machine failure.
            add("FAST++ template-error file (fastpp_share)",
                self.fastpp_share / "TEMPLATE_ERROR.fast.v0.2", "file")
            add("FAST++ SPS library directory (fastpp_libraries)",
                self.fastpp_libraries, "dir")

        return reqs

    def _eazy_template_spectra_reqs(
        self, spectra_file
    ) -> list[tuple[str, Path, bool, str]]:
        """Check each template spectrum path listed in the SED definition.

        ``eazy_v1.2_dusty.spectra.param`` references the individual template
        ``.dat`` files by (often absolute) path; copying the LIB tree to a
        different root on another machine silently breaks these. Returns one
        requirement per listed template (empty if the definition is absent —
        that absence is already reported separately).
        """
        reqs: list[tuple[str, Path, bool, str]] = []
        p = Path(spectra_file)
        if not p.is_file():
            return reqs
        try:
            for line in p.read_text().splitlines():
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                parts = s.split()
                if len(parts) < 2:
                    continue
                tpath = Path(parts[1])
                reqs.append(
                    (f"EAzY template spectrum #{parts[0]}", tpath, tpath.is_file(),
                     "file")
                )
        except OSError:
            pass
        return reqs

    def check_requirements(
        self, *, do_photoz: bool = True, do_sedfit: bool = True, deep: bool = True
    ) -> list[str]:
        """Return a list of missing/invalid required inputs (empty == all OK).

        Non-raising counterpart of :meth:`preflight`; useful for programmatic
        checks. Also verifies the eazy-py import when
        ``photoz_engine == "eazy-py"``.
        """
        problems: list[str] = []
        nouns = {"dir": "directory", "exe": "executable", "file": "file"}
        for label, path, ok, kind in self._requirement_list(
            do_photoz=do_photoz, do_sedfit=do_sedfit, deep=deep
        ):
            if not ok:
                problems.append(f"{label}: missing {nouns[kind]} -> {path}")
        if do_photoz and self.photoz_engine == "eazy-py":
            try:
                import eazy  # noqa: F401
            except Exception as exc:  # pragma: no cover - import-time/env issue
                problems.append(
                    f"eazy-py not importable (photoz_engine='eazy-py'): {exc!r}"
                )
        return problems

    def preflight(
        self, *, do_photoz: bool = True, do_sedfit: bool = True,
        deep: bool = True, strict: bool = True, verbose: bool = True,
    ) -> list[str]:
        """Check every required config file / template / binary up front.

        Logs a per-item ``OK``/``MISS`` checklist (``verbose``) and, when
        ``strict`` (default), raises :class:`FileNotFoundError` listing **all**
        problems at once — so path issues surface before any long-running
        stage instead of mid-run. Returns the list of problems.
        """
        reqs = self._requirement_list(
            do_photoz=do_photoz, do_sedfit=do_sedfit, deep=deep
        )
        if verbose:
            log.info("VAC preflight: %d required inputs (photo-z=%s, SED-fit=%s, "
                     "engine=%s)", len(reqs), do_photoz, do_sedfit,
                     self.photoz_engine)
            for label, path, ok, _kind in reqs:
                log.info("  [%s] %s: %s", "OK  " if ok else "MISS", label, path)
        problems = self.check_requirements(
            do_photoz=do_photoz, do_sedfit=do_sedfit, deep=deep
        )
        if problems and strict:
            bullets = "\n  - ".join(problems)
            raise FileNotFoundError(
                f"VAC preflight failed: {len(problems)} required input(s) missing "
                f"or invalid:\n  - {bullets}\n"
                "Fix the paths above via VACConfig (lib_dir / sfd_dir / fastpp_bin "
                "/ fastpp_share_dir / fastpp_library_dir / eazy_bin), then re-run."
            )
        return problems

    def validate(self, *, require_fastpp: bool = True,
                 require_eazy_bin: bool = False) -> None:
        """Back-compat thin wrapper around :meth:`preflight`.

        Kept for existing callers; new code should call :meth:`preflight`
        directly. ``require_eazy_bin`` is accepted for compatibility but the
        EAzY binary is now checked automatically whenever the photo-z stage
        uses ``photoz_engine == "binary"``.
        """
        self.preflight(do_photoz=True, do_sedfit=require_fastpp, deep=False,
                       strict=True, verbose=False)


__all__ = ["VACConfig", "DEFAULT_MEDIUM_BANDS", "DEFAULT_BROAD_BANDS"]
