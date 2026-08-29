# Changelog

All notable changes to `phot7ds`. Versions follow
[semantic versioning](https://semver.org/) loosely: the minor number moves on
new features or behaviour changes, the patch number on fixes.

## v0.5.0 — 2026-08-29

Everything that accumulated on `main` after the 0.4.0 version bump, plus a
cleanup pass making the package portable across machines.

### Changed — behaviour worth knowing about before you re-run

- **Default magnitude prior replaced.** `VACConfig.prior_file` now defaults to
  a prior that **ships with the package**,
  `phot7ds/vac/data/prior_m6250_desi.dat`, instead of expecting
  `lib_dir/templates/prior_m6250_extend.dat`. The new table is a Benitez (2000)
  `p(z|m625)` fitted jointly to SDSS DR16 (bright anchor on the
  `m625 = r_SDSS - 0.2` system, complete to `m625 < 17.55`) and DESI DR1
  BGS_BRIGHT (flux-limited `r < 19.5`, extending the empirical anchor to
  `m625 < 19.25`), with a quadratic `ln zm(m)` to follow the flattening of
  `d ln z / dm` that a linear law misses. The old EL-COSMOS-derived prior is
  mis-calibrated at the bright end and biases low-z galaxies high, so
  **photo-z results will shift** relative to 0.4.0. Pass `prior_file=` to pin
  any other table, including the old one.
- **FAST++ / EAzY binaries are discovered, not hard-coded.** `fastpp_bin` and
  `eazy_bin` now default to `None` and are resolved from
  `$PHOT7DS_FASTPP_BIN` / `$PHOT7DS_EAZY_BIN` and then `$PATH`. Previously they
  defaulted to absolute paths from the original author's machine, which
  silently pointed every other installation at nonexistent files. An
  unresolved binary is reported by `preflight()` as `not configured` instead of
  failing mid-run.
- **Zero-point calibration no longer applies a global `source_flags == 0`
  cut.** That flag comes from the detection image and discarded too many
  well-measured calibration stars in crowded fields, leaving some
  band/aperture pairs without enough stars to fit a ZP surface. The per-band,
  per-aperture flag cut is now the only flag cut and is tunable via the new
  `band_flag_cut` argument (default `0`, i.e. previous strictness per band).

### Added

- `vac/photoz_binary.py`: photo-z via the compiled EAzY executable, and
  `VACConfig.photoz_engine` to choose it. It is now the **default** (`"binary"`)
  because the pure-Python eazy-py `TemplateGrid` build is pathologically slow
  for the 7DS medium-band filter set — it could not finish one of 8 templates
  in 242 s on a 803-row, 27-filter tile. The binary's native `.zout` already
  carries `z_m2`/`z_a` and the `l68/u68`, `l95/u95`, `l99/u99` intervals
  FAST++ needs, so it is reused verbatim.
- `VACConfig.preflight()` / `check_requirements()`: up-front validation of
  every required config file, template, SPS library and binary for the
  requested stages, reporting all problems at once rather than failing mid-run.
  With `deep=True` it also checks the individual template spectra listed in
  `eazy_v1.2_dusty.spectra.param`, which catches a LIB tree copied to a
  different root.
- `vac/vizier.py`: self-contained on-demand VizieR download of the per-tile
  REGALADE / VHS / GALEX references (`auto_download=True`), with polygon trim.
  Replaces the dependency on an external query script; `astroquery` is imported
  lazily and lives in the `vac` extra.
- `vac/report.py`: per-run log recording the cross-match settings, detected
  filters with central wavelengths and extinction, target counts and the full
  EAzY / FAST++ configuration.
- FAST++ grid and performance knobs on `VACConfig` (`fastpp_resolution`,
  `fastpp_force_zphot`, `fastpp_parallel`, `fastpp_metal`, `z_step_type`,
  `fastpp_best_fit`, `fastpp_intrinsic_best_fit`) with defaults that reproduce
  the fast legacy configuration instead of inheriting the much slower
  `LIB/fastpp.param` defaults.
- `detection/sevends.py`: `MEDIAN` combine for the 7DS white detection image.
  Being unweighted, it neither requires nor uses weight maps, so coadds without
  a `*_weight.fits` sibling can be stacked too (`require_weights=False`).
- `detection/delve.py`: overlap-aware automatic patch grid, reusable
  `download_delve_patches()`, and `fill_delve_detection_gaps()` for closing
  grid-induced and brick-boundary gaps in existing mosaics (both now exported
  from `phot7ds.detection`).
- Per-image `DATE-{nnn}` observation dates recorded in the output catalog
  header alongside the existing provenance keys.

### Fixed

- Constant-ZP fitting no longer crashes or silently mis-fits when a
  band/aperture has non-finite residuals; the band is skipped with a log line
  instead. Spatial-ZP logging now names the instrumental column being fitted
  rather than only the reference.
- `fastpp_share` returns `None` instead of raising when it cannot be derived,
  and the FAST++ stage reports the missing binary clearly.

### Documentation / packaging

- `examples/example_vac.py` restored to a self-contained single-tile example;
  it had drifted into a personal batch loop with `main()` defined inside it.
- Machine-specific absolute paths removed from the README examples.
- `pyproject.toml` ships `phot7ds/vac/data/*.dat` as package data.
- Smoke tests cover the packaged prior's shape and the binary-discovery
  precedence (explicit value, environment variable, `$PATH`, unset).

## v0.4.0 — 2026-06-05

- `phot7ds.vac` subpackage: value-added catalogs from a phot7ds photometric
  catalog — REGALADE/VHS/GALEX cross-matching with dedup, extinction-corrected
  flux assembly, eazy-py photo-z, FAST++ SED fitting and the merged output
  catalog (`run_value_added()`).
- Filters entering the fit are auto-detected from the catalog's
  `{aperture}_mag_*` columns, retiring the `use_medium` / `use_broad` toggles.

## v0.3.1 — 2026-05-26

- FITS header key reorganisation; `aper05`-style zero-padded aperture columns.
- Final catalog name kept verbatim (`{any suffix}.fits`) instead of forcing
  `_phot.zp.fits`.
- Config directories created relative to the calling script.
- Explicit detection-label handling; SE++ presets per detection image.
- Depth estimation made robust to science images off the detection grid.
- 7DS native white detection-image builder (`detection/sevends.py`).

## v0.3.0 — 2026-05-24

- 5-sigma depth estimation (magnitude-error curve fit + empty-aperture sky
  sigma) written to the log, manifest and FITS header.
- ZP and depth header keys; `detection_label` required so the DELVE minimum
  Kron radius is correct.
- DELVE SWarp center defaults to the tile RA/Dec in sexagesimal.

## v0.2.0 — 2026-05-21

- Initial public release: SE++-driven forced photometry on 7DS images guided
  by a single detection image, Gaia XP zero-point calibration, bad-pixel
  masking, DELVE detection-image builder, canonical output schema and the
  batch runner.
