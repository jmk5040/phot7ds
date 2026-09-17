# Changelog

All notable changes to `phot7ds`. Versions follow
[semantic versioning](https://semver.org/) loosely: the minor number moves on
new features or behaviour changes, the patch number on fixes.

## v0.7.1 - 2026-09-17

### Added

- **`segmentation_filter`** (`PhotometryConfig`, `run_photometry`,
  `build_sepp_command`) — path of an SE++ `.conv` detection kernel, passed as
  `--segmentation-filter`. Empty (default) leaves the kernel to the
  `--config-file`, i.e. SE++'s built-in 3x3 unless that file names one.

## v0.7.0 — 2026-09-11

Per-band mask flags. The 7DT image pipeline (py7DT) now delivers a count-map
MEF next to every coadd (`<coadd>_counts.fits`: `NBAD`, `NSAT`, `NTRAIL`,
`NOUTLIER`, … as `uint8` frame counts). SE++ cannot consume those directly —
the planes are tile-compressed, `[EXTNAME]` selectors are silently ignored,
and six planes per band would mean hundreds of columns — so phot7ds folds
each MEF into a single bitmask per band and hands that to SE++ as a flag
image. The catalog gains two columns per band and loses the union coverage
column. Validated on T08147 (23 bands): the bitmask path changes no
photometric value, costs ~3 s of staging and ~2.5 min of extra SE++ time
(+39 % on a 6.5 min run at 16 threads), and 31 % / 48 % of m525 sources come
out with `mask_flags == 0` / `< 4`.

### Added

- **`phot7ds.masks`** — `build_band_bitmask()` (MEF + coadd → `uint8`
  bitmask), `resolve_count_masks()` / `find_count_mask()` (discovery by the
  `<coadd stem>_counts.fits` convention, following symlinks; or a directory,
  a `{band: path}` mapping, or a list matched by the MEF `FILTER` header),
  `MaskStaging` (parallel build into `/dev/shm`, `finally` cleanup, sweep of
  stale `phot7ds_masks_*_<pid>` directories from killed runs, disk fallback),
  `rename_mask_columns()`, `any_band_nodata()`.
- **Bit convention** = py7DT `MaskBit`: 1 OUTLIER, 2 BADPIX, 4 STRAY,
  8 SATELLITE, 16 SATURATED, 32 HOT, 64 DEAD, plus **128 NODATA** (phot7ds):
  coadd pixel `== 0` or non-finite, or `NUSED == 0`. A defect bit is set where
  the count is `> 0`; severity is left to `mask_npix`. Mild conditions have the
  low bits so `mask_flags < 4` is a usable "clean enough" cut.
- **New catalog columns** `mask_flags_<band>` (OR over the detection
  isophote) and `mask_npix_<band>` (isophote pixels with any bit set), one
  pair per measurement band, part of the canonical schema. Header cards
  `MBOUTLIE … MBNODATA` (bit values), `NBANDMSK`, `NCNTMSK`, `MSKnnn`
  (`<band>:<MEF>` or `<band>:coverage-only`), `MSKSTAGE`; manifest section
  `band_masks` with per-band status, planes found/missing and pixel counts.
- **`run_photometry(count_masks=...)`** and `PhotometryConfig` fields
  `per_band_masks` (default `True`), `count_mask_suffix`, `mask_use_nused`,
  `mask_staging_dir` (`/dev/shm`; `""` → work dir), `mask_workers`,
  `mask_flag_type`.
- **`sepp.FlagImage`** and `build_sepp_command(flag_images=[...])` for an
  arbitrary set of `--flag-image-<label>` / `--flag-type-<label>` pairs;
  labels are validated for uniqueness and SE++-safe characters.

### Changed

- **`isophotal_image_flags_cover` / `..._pixel_count_cover` are gone** from
  the default output and the canonical schema. Bit 128 of `mask_flags_<band>`
  is the per-band replacement: a source outside one band's footprint is no
  longer flagged in all bands (on T08147 the union column over-flagged
  1.7k–12.2k sources per band). The union coverage mask is still built for
  the depth estimate; `per_band_masks=False` restores the legacy flag image
  and columns. The VAC coverage cut (`vac.pipeline._load_catalog`) accepts
  both: it uses `cover_flag_column` when present, otherwise rejects sources
  with bit 128 in any band, which reproduces the old selection.
- **`{run_name}_mask.fits` is no longer written to the output directory.**
  The union coverage mask is still built — it places the empty apertures of
  the depth estimate and yields `MSKRATIO` — but in the mask staging
  directory (tmpfs), and it is removed with it once the run is over. Set
  `save_coverage_mask=True` (kwarg or `PhotometryConfig`) to keep it in
  `output_dir` as before; `COVMASK` is written only when the file survives.
  Per-band bitmasks are freed as soon as SE++ returns.
- `build_coverage_mask` treats non-finite pixels as no-data, not only zeros.
- `build_canonical_schema(..., per_band_masks=True)` appends the mask
  columns after the basic block.
- `resolve_count_masks` logs a WARNING (not INFO) listing the bands that
  fell back to a coverage-only bitmask.

### Fixed

- `build_sepp_command` requested `ExternalFlags` even when no flag image was
  given, which makes SE++ abort; the property is now dropped in that case.

### Robustness (absent / partial MEFs)

- A band with no MEF, an unreadable one, one on a different grid, or one
  without any known plane gets a **coverage-only** bitmask (bit 128 from the
  coadd alone), a warning, and `status = "coverage-only: <reason>"` in the
  manifest and `MSKWHY` in the bitmask header. The run continues. Checked on
  T08147 with a single real MEF (m525) and 22 bands without.
- A coadd that is not on the detection grid is skipped for masking (no
  `mask_flags` for that band; placeholder in the canonical schema) instead of
  aborting; if no bitmask at all could be built the legacy coverage flag
  image is used.

## v0.6.0 — 2026-08-29

Promotes the machinery that had accumulated in the RIS production driver
(`RIS_vac.py`) into the package, so a run no longer depends on a script
monkey-patching module-level tables. The driver dropped from ~1030 lines to
~470, and the library gained the aperture-matching, prefetch, dead-band and
split-stage logic it was reaching around.

### Fixed

- **VizieR tile queries no longer over-fetch near the poles.** The query box
  used the raw RA span of the tile polygon, which is `1/cos(dec)` times the
  angular width — a 1.4°-wide tile at dec = −83° was queried over ~12° of RA,
  roughly an order of magnitude too much sky, enough to stall a dense catalog
  like VHS DR5. The span is now deprojected by `cos(dec)`
  (`vizier.tile_query_box`).
- **VizieR queries can no longer hang a run.** astroquery does not reliably
  honour its own `timeout` and can stall indefinitely on a half-closed
  connection. Every query now also runs under a hard `SIGALRM` wall clock
  (`VACConfig.vizier_timeout`, default 180 s) and is retried
  `vizier_attempts` (2) times.
- **VHS Vega→AB offsets are registered for both magnitude sets.** Only the
  Petrosian column names were listed, so any run using the 5.7″ aperture
  columns would have left the NIR points on the Vega scale (J off by
  0.94 mag).
- The filter-coverage cut is vectorized instead of looping over table rows.

### Added

- **`VACConfig.vhs_mag_set`** — `"petro"` (Petrosian, total light) or `"ap6"`
  (5.7″ aperture, `Japc6`/`Hapc6`/`Ksapc6`), selecting the VHS columns joined
  onto the SED, the Vega→AB offsets, the VizieR preset and the staging
  directory in one field. A VHS band has to measure the same kind of light as
  the 7DS band next to it: pairing a fixed 7DS aperture with total-light VHS
  magnitudes tilts the optical-to-NIR slope by aperture alone and the photo-z
  absorbs the difference as redshift. `resolve_vhs_mag_set(aperture)` returns
  the matching set. The two sets are staged separately
  (`config.VHS_STAGING`), since the ap6 columns are outside VizieR's default
  VHS column set.
- **`VACConfig.min_7ds_band_fraction`** — expresses the coverage cut against
  the 7DS bands alone, with the external surveys counted as a bonus rather
  than in the denominator. `min_filter_fraction` alone can demand more
  filters than a tile will ever have when band coverage varies: for a tile
  with no `g`/`r`/`i` coadds the 0.80 all-filter threshold left 40 of 958
  galaxies.
- **`fluxes.drop_dead_bands` / `live_bands`**, reachable as
  `run_value_added(..., drop_empty_bands=True)` — drops magnitude columns
  that are entirely empty before the filters are detected. An unobserved band
  still has its column, filled with NaN, and detection works from column
  presence, so without this the dead bands sat in the coverage denominator.
- **`vizier.prefetch_references`** — stages every tile's external references
  up front, under the bounded time limit, so a flaky query cannot stall a
  long batch part-way through and `auto_download` can be left off during
  fitting.
- **`pipeline.run_value_added_split`** — fits the photo-z and SED-fit stages
  on different photometry (different aperture, VHS magnitude set and `use_*`
  toggles), cutting both flux catalogs to the galaxies that survive both
  coverage cuts, since everything downstream is aligned by row order.
- VizieR presets `vhs_ap6` (a superset of `vhs`, so one matched table can
  serve both magnitude sets) and `catwise` (CatWISE2020).
- `VACConfig.reference_path(key, tile)`, `VACConfig.vhs_staging`,
  `fluxes.external_columns`, `fluxes.write_flux_inputs`,
  `fluxes.required_filter_count`, `vizier.time_limit`,
  `pipeline.select_tile_row` — the helpers the driver had been reaching for
  privately.

### Changed

- `VACConfig.vhs_subdir` / `vhs_template` now default to `None`, meaning
  "follow `vhs_mag_set`", and are resolved on access (`vhs_staging`) rather
  than pinned at construction. Pinning them in `__post_init__` would leave
  `dataclasses.replace(cfg, vhs_mag_set=...)` with a magnitude set and a
  staging directory that disagree. Setting either field explicitly still wins.
- `build_flux_catalog(..., write=False)` skips writing the stage inputs, for
  callers that need to intersect the two stages' row sets first.

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
