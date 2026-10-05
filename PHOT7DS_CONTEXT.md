# phot7ds — Session Context / Handoff

Working memory for the `phot7ds` package and the 7DS/RIS scripts around it.
Read this first when starting a new session. Version at time of writing:
**phot7ds 0.9.1** (`phot7ds/__init__.py`, `pyproject.toml`).

---

## 1. What this is

`phot7ds` is a reusable Python package (Python API only) for processing 7DS
survey data into zero-point-calibrated photometric catalogs and value-added
catalogs. It refactors older one-off scripts into a modular package hosted on
GitHub. The workspace root is `/lyman/data1/7DS/RIS/script`; the package lives
in the sibling `Phot7DS/` dir (`Phot7DS/phot7ds/`).

## 2. Environment & how to run

- Conda env: **`7dt`** → interpreter `/home/jmkastro/miniconda3/envs/7dt/bin/python`
  (numpy 2.4.2). Always run scripts with this interpreter.
- External binaries: **`SWarp`** at `/usr/bin/SWarp`; **FAST++** at
  `/home/jmkastro/fastpp/bin/fast++`; **EAzY** at
  `/lyman/data1/7DS/RIS/config/eazy/src/eazy`; SourceExtractor++
  (`sourcextractor++`). Since v0.5.0 `VACConfig` no longer hard-codes the
  FAST++/EAzY paths — export `PHOT7DS_FASTPP_BIN` / `PHOT7DS_EAZY_BIN` (or pass
  `fastpp_bin=` / `eazy_bin=`) on this machine.
- Set `MPLCONFIGDIR=/tmp/mpl` to avoid matplotlib cache warnings.
- Import without install: scripts insert `/lyman/data1/7DS/RIS/script/Phot7DS`
  onto `sys.path` (see `phot7ds_IMS.py`). The basedpyright "could not be
  resolved" warning for `phot7ds` in such scripts is a false positive.

### Sandbox / filesystem gotchas (IMPORTANT)
- The agent shell runs sandboxed: **writes are only allowed inside the
  workspace** (`/lyman/data1/7DS/RIS/script`). Everything else (e.g.
  `/lyman/data1/7DS/IMS/DELVE`, `/lyman/data2/RIS/data`, `/lyman/data1/7DS/RIS/config`,
  `/lyman/data1/7DS/RIS/catalog`) is **read-only** unless you pass
  `required_permissions: ["all"]`.
- Network (NOIRLab SIA, Vizier, etc.) needs `["all"]` (or `full_network`).
- Those data dirs are often **root-owned**; the user runs as root but the
  sandbox still blocks writes without `["all"]`.

## 3. Key external paths

| Purpose | Path |
|---|---|
| Config / LIB tree (EAzY, FILTER.RES, priors, SFD, swarp cfg) | `/lyman/data1/7DS/RIS/config/` and `.../config/LIB/` |
| SWarp config | `/lyman/data1/7DS/RIS/config/7ds.swarp` |
| SE++ config | `/lyman/data1/7DS/RIS/config/7ds_sepp.config` |
| Tile table | `/lyman/data1/7DS/RIS/config/7DT_tiles.fits` (also `.ascii`) |
| Band coverage table | `/lyman/data1/7DS/RIS/config/coadd_band_coverage.csv` |
| Gaia XP reference CSVs | `/lyman/data1/7DS/RIS/catalog/gaiaxp/` |
| Per-tile science coadds (+ `_weight.fits`) | `/lyman/data2/RIS/lyman/{tile}/*_coadd.fits` |
| Output catalogs | `/lyman/data1/7DS/RIS/catalog/7ds/{tile}/` |
| IMS DELVE detection images | `/lyman/data1/7DS/IMS/DELVE/{tile}_DELVE_DR3_{IMAGE,MASK}_det.fits` |

## 4. Package layout

```
phot7ds/
  __init__.py        # version + public API
  config.py          # PhotometryConfig (dataclass); apertures default ("aper05", ...)
  config_io.py       # ensure_/require_ helpers for config & reference files
  pipeline.py        # run_photometry() — main entry; _annotate_catalog_meta()
  batch.py           # batch_run()
  calibration.py     # calibrate_zeropoints(), apply_spatial_zeropoint(); ZP per band×aperture
  depth.py           # depth estimation + ZP/depth header meta; WCS-based empty-aperture sky sigma
  images.py          # organize_images_by_filter(), build_coverage_mask() (shape-checked → (None,None))
  masks.py           # per-band bitmasks from py7DT count-map MEFs; MaskStaging (/dev/shm); v0.7.0
  sepp.py            # SE++ command (FlagImage list) + output parsing; aperture labels zero-padded ("05")
  schema.py          # canonical catalog schema
  filters.py         # 7DS filter definitions / DEFAULT_BANDS
  presets.py         # detection-label SE++ tuning presets (PRESET_TUNING_FIELDS)
  crossmatch.py      # matching() sky cross-match (ported from Utils_7DT)
  photconv.py        # mag/flux conversions, filter_colorization
  diagnostics.py     # residual map plots
  tile_geometry.py
  _logging.py
  detection/
    delve.py         # DELVE-DR3 mosaic builder + gap-fill (see §6)
    sevends.py       # 7DS native white detection image builder
    __init__.py
  vac/               # value-added catalog subpackage (see §7)
    config.py photoz.py photoz_binary.py sedfit.py fluxes.py crossmatch.py
    vizier.py catalog.py report.py pipeline.py __init__.py
```

## 5. Photometry pipeline conventions

- Entry: `run_photometry(science_images=..., detection_image=..., reference_catalog=..., ...)`.
  Settings can be passed as kwargs or via a `PhotometryConfig` (`config=`); explicit
  kwargs override the config.
- Apertures use **zero-padded** labels: `aper05`, `aper10` (not `aper5`).
- ZP calibration (`calibrate_zeropoints`) runs **once per (band × aperture)**:
  fits a 2-D polynomial spatial ZP (`{aper}c_mag_{band}`) + a constant ZP
  (`{aper}_mag_{band}`). This is why ZP log lines repeat per band (one per
  aperture). The "Spatial ZP: fitting…" line prints the instrumental column
  (`aper05_mag_m875`) to disambiguate apertures.
- Catalog primary-header metadata is injected by `_annotate_catalog_meta()`
  (pipeline.py). Cards are short, FITS-safe. Includes: `PHOTVER/PHOTRUN/PHOTDATE`,
  `DETLABEL/DETIMG`, `REFCAT`, `NSCIIMG`, `SCIMG{nnn}` (basenames),
  tuning fields (`DETTHR`, `DETMINAR`, …), `PIXSCALE`, `MSKRATIO`.
  - **Observation dates (added 2026-06-16):** per science image
    `DATE-{nnn}` = that image's `DATE-OBS`, paired with `SCIMG{nnn}`; plus an
    **exposure-averaged `DATE-OBS` (ISO)** and **`MJD-OBS`** computed by
    averaging each image's `DATE-OBS` via `astropy.time.Time` (fallback to
    header `MJD`/`MJD-OBS`). NOTE: in 7DS coadds, header `MJD` matches
    `DATE-OBS` while header `MJD-OBS` is offset ~0.5 d — so we average from
    `DATE-OBS`, not `MJD-OBS`.
- `batch_run()` / `phot7ds_IMS.py` loop tiles; per-tile errors are caught so the
  batch continues. `phot7ds_IMS.py` builds a detection image (7DS white stack or
  DELVE) then calls `run_photometry`.
- **Per-band mask flags (v0.7.0, `masks.py`):** for every measurement band a
  `uint8` bitmask is built from the coadd (bit 128 = `==0 | ~isfinite`) plus,
  if found, the py7DT count-map MEF `<coadd>_counts.fits` (bits 1–64 = py7DT
  `MaskBit`: OUTLIER 1, BADPIX 2, STRAY 4, SATELLITE 8, SATURATED 16, HOT 32,
  DEAD 64; a bit is set where the count `> 0`; `NUSED == 0` also → 128).
  Staged in `/dev/shm/phot7ds_masks_<run>_<pid>/` (1.6 GB for 23 bands, ~3 s,
  8 workers), passed as `--flag-image-<band> … --flag-type-<band> or`, removed
  in `finally`; stale dirs of dead PIDs swept at start. Columns
  `mask_flags_<band>` (OR over isophote) / `mask_npix_<band>`. The union
  `isophotal_image_flags_cover` column is gone (bit 128 is per band); the
  union coverage mask is still built (depth empty apertures, `MSKRATIO`) but
  into the staging dir and deleted at the end unless `save_coverage_mask=True`
  (then `{run_name}_mask.fits` in work_dir + `COVMASK`). Absent/partial/broken MEFs →
  coverage-only bitmask (128 only) + warning + `status` in manifest
  `band_masks`; never fatal. `count_masks=` accepts None (auto, follows
  symlinks) / dir / `{band: path}` / list (matched by `FILTER`).
  `per_band_masks=False` = legacy behaviour. SE++ facts learned: `[EXTNAME]`
  selectors are silently ignored, tile-compressed HDUs unreadable, `[N]`
  index works, `.fits.gz` works, requesting `ExternalFlags` with no flag
  image is FATAL (now dropped automatically).

## 6. Detection images — `phot7ds.detection`

### DELVE (`delve.py`)
- `build_delve_detection_image(...)`: partitions the tile FOV into a patch grid,
  queries NOIRLab SIA (`delve_dr3`) per patch, downloads, SWarps into one mosaic
  (`-COMBINE_TYPE MAX`, never MEDIAN). Robust retries w/ exponential backoff.
- **Overlap-aware auto grid:** pass `n_cols=None, n_rows=None, patch_size_deg=,
  overlap=` to size the grid from the field span so patches overlap in
  *coordinate* degrees. Needed at high |dec| (e.g. IMS tiles dec≈−61°,
  cosδ≈0.48) where a fixed 9×6 grid leaves thin **RA-direction** gaps. Defaults
  remain `9×6, overlap=0.0` (backward compatible for equatorial RIS tiles).
- `download_delve_patches(centers, ..., all_matches=False)`: reusable threaded
  downloader. `all_matches=True` downloads **every** overlapping brick per
  center (deduped by URL) — needed at brick boundaries/tile edges where the
  first-returned brick has a hole but a neighbour covers it.
- `fill_delve_detection_gaps(image_path, ..., imgtype, coverage_reference=None,
  overlap=0.5, max_passes=5, all_matches via internal)`: **repairs existing
  mosaics in place**. Finds empty pixels, places query centers at the
  **centroid of gap pixels** per bin (gaps lie on brick boundaries, so a
  bin-center query returns the wrong brick), downloads all overlapping bricks,
  SWarps onto the *same frame*, merges into empty pixels only. Iterates passes
  (each pass re-targets the shrinking residual). For **science images** coverage
  = `data!=0`; for **masks** coverage = SWarp weight>0 (a mask value of 0 is a
  valid pixel, not "no data"). For masks, pass the sibling IMAGE as
  `coverage_reference`.

### 7DS white (`sevends.py`)
- `build_7ds_detection_image(image_dir, ..., medium_only=, one_per_band=,
  combine_type="WEIGHTED"|"MEDIAN")`: stacks per-band coadds into a white
  detection image with SWarp + weight maps. `one_per_band` keeps the
  sharpest-SEEING image per band (FWHM fallback; skip if unavailable),
  parallelized header reads (default `n_workers=1`). Records provenance in the
  header (`DETIMGnn`, `DETSEEnn`, `SEEMIN/MED/MAX`, `MEDONLY`, `ONEPRBND`).
  `MEDIAN` combine does not require weight maps.

### IMS work (done 2026-06-16) — `IMS_detect.py`
- 7 IMS tiles (`T02666 T02665 T02524 T02523 T02386 T02385 T02252`),
  output `/lyman/data1/7DS/IMS/DELVE/`. Filled grid-induced + brick-boundary gaps
  to **0.0000** on all tiles. T02386's MASK had been generated as science data
  (pre-existing bug) → regenerated from scratch as a proper mask. `IMS_detect.py`
  gap-fills existing images (cheap) or `FULL_REGEN=True` rebuilds with the auto grid.

## 7. Value-added catalog — `phot7ds.vac`

- Entry: `run_value_added(...)` orchestrated in `vac/pipeline.py`; config
  `VACConfig` (`vac/config.py`). Example: `Phot7DS/examples/example_vac.py`.
  Install extra: `pip install -e ".[vac]"` (eazy, sfdmap2, extinction).
- Stages: galaxy match + optional external-catalog download (REGALADE, VHS,
  GALEX, WISE via Vizier — `vac/vizier.py`, `auto_download=True`) →
  flux catalog (`vac/fluxes.py`, **auto-detects filters** from catalog columns,
  no more `use_medium/use_broad`) → photo-z (see engine below) →
  SED fit FAST++ (`vac/sedfit.py`) → assemble (`vac/catalog.py`) → run log
  (`vac/report.py`).
- **Photo-z engine (`VACConfig.photoz_engine`, added 2026-06-30):** default
  `"binary"` → `vac/photoz_binary.py:run_eazy_binary()` shells out to the
  compiled EAzY at `VACConfig.eazy_bin` (on this machine
  `/lyman/data1/7DS/RIS/config/eazy/src/eazy`, now supplied via
  `$PHOT7DS_EAZY_BIN` rather than a hard-coded default); ~2 min for ~800
  sources. The
  binary's native `.zout` already has `z_m2`/`z_a` + `l68/u68/l95/u95/l99/u99`
  for FAST++, so it is copied verbatim into the SED-fit dir.
  `"eazy-py"` → `vac/photoz.py:run_eazy()` (pure Python) is kept but is
  **effectively unusable** for the 7DS medium-band set: the eazy-py
  `TemplateGrid` build (`PhotoZ.__init__`, serial) failed to finish even one
  of 8 templates in 242 s on T22956 (803 rows, 27 filters), long before
  `fit_catalog` started. It is the grid build that is slow, NOT the ZP
  offsets (eazy-py never iterates ZP here) and NOT `fit_catalog`.
- VizieR auto-download is **self-contained** in `vac/vizier.py` (presets
  `CATALOG_PRESETS` for regalade/vhs/vhs_ap6/galex/catwise +
  `query_vizier_catalog_for_tile` / `download_catalog_for_tile`); `astroquery`
  is imported lazily and is in the `vac` extra. No longer depends on the
  external `Query/Vizier_Query.py` (the old `VACConfig.vizier_query_path`
  field was removed). Polygon trim reuses
  `phot7ds.tile_geometry.trim_to_tile_polygon`.
- **VizieR query box** (`tile_query_box`, v0.6.0): the RA span is deprojected
  by `cos(dec)`. Before that fix a 1.4°-wide tile at dec = −83° was queried
  over ~12° of RA — an order of magnitude too much sky, enough to stall VHS
  DR5. Every query also runs under a hard `SIGALRM` cap
  (`vizier_timeout`, 180 s) with `vizier_attempts` retries, because astroquery
  can ignore its own timeout and hang. For a batch, call
  `prefetch_references(...)` first and leave `auto_download=False` during
  fitting.
- **VHS aperture matching** (`vhs_mag_set`, v0.6.0): `"petro"` pairs with
  `auto`/`autoc`, `"ap6"` (`Japc6`/`Hapc6`/`Ksapc6`, 5.7″) pairs with
  `aper05c`; `resolve_vhs_mag_set(aperture)` does the pairing. Mixing them
  tilts the optical-to-NIR slope by aperture alone and the photo-z absorbs it
  as redshift. The two sets stage separately (`vhs_dr5/`, `vhs_dr5_ap6/`) since
  the ap6 columns are outside VizieR's default column set. `vhs_subdir` /
  `vhs_template` default to `None` and resolve on access, so
  `dataclasses.replace(cfg, vhs_mag_set=...)` moves the staging with the
  magnitude set. Vega→AB offsets are keyed by column name and registered for
  **both** sets.
- **Coverage cut** (v0.6.0): `min_7ds_band_fraction` counts only the 7DS bands
  in the denominator, externals as a bonus. Use it with
  `run_value_added(..., drop_empty_bands=True)` when tiles differ in band
  coverage — an unobserved band still carries an all-NaN column and filter
  detection works from column presence, so a plain `min_filter_fraction=0.80`
  left 40 of 958 galaxies on a tile missing `g`/`r`/`i`.
- **Split stages** (`run_value_added_split`, v0.6.0): photo-z and SED fit on
  different photometry (aperture / VHS set / `use_*`). Both flux catalogs are
  cut to the galaxies surviving both coverage cuts, because FAST++ reads the
  `.zout` EAzY wrote and the merge `hstack`s positionally.
- Filters: broad bands use `f_7DS_g/r/i` (not `f_SDSS_*`). `FILTER.RES.latest`
  and `default.translate` were updated by `update_7ds_filters.py`; keep the two
  files in sync (a prior desync was missing `f_7DS_g/r/i`).
- Prior: since v0.5.0 the default is the **packaged**
  `phot7ds/vac/data/prior_m6250_desi.dat` (band m625) — the joint SDSS DR16 +
  DESI DR1 BGS fit built by `IMS/script/ris_build_m625_prior_desi.py`. It
  replaces `prior_m6250_extend.dat` (EL-COSMOS), which is mis-calibrated at the
  bright end. `prior_file=` still overrides. If a prior applies → redshift
  column `z_m2`, else `z_a`. Warn (don't fail) if `aper05c_mag_m625` missing.
- **eazy-py multiprocessing deadlock:** `VACConfig.eazy_n_proc = -1` (default)
  forces serial `TemplateGrid` build and serial `fit_catalog` to avoid a fork
  timeout. Don't set it positive unless you know it's safe.
- numpy 2.x: use `sfdmap2` (or the `_sfd_ebv` shim) — legacy `sfdmap` uses
  removed `np.int`.

## 8. Conventions / gotchas summary

- Sexagesimal: `delve.deg_to_hms_dms()` carries seconds/minutes correctly (no
  `:60.00`).
- FITS keyword cards are ≤8 chars; hyphenated keys like `DATE-OBS`, `DATE-000`
  are valid and survive astropy `Table.write(format="fits")` with comments.
- `pytest` is installed in the `7dt` env: run the smoke tests in `tests/` with
  `MPLCONFIGDIR=/tmp/mpl python -m pytest tests/ -q`.
- `build_coverage_mask` returns `(None, None)` (and logs) when science image
  shapes don't match the detection image, instead of raising.
- Per-band bitmasks live in `/dev/shm` only while SE++ runs; if a run is
  SIGKILLed, `phot7ds_masks_*_<pid>` may linger until the next run sweeps it
  (`masks.sweep_stale_staging`). `mask_staging_dir=""` stages in the work dir.
- Diagnostic figures off: `save_residual_plots=False` (run_photometry) /
  `plot_residuals=False`.

## 9. Recent changes (2026-06-16)

1. `detection/delve.py`: overlap-aware auto grid; reusable
   `download_delve_patches(all_matches=)`; new `fill_delve_detection_gaps`
   (centroid-centered queries, all-bricks, data-vs-weight coverage, multi-pass).
   Filled all 7 IMS tiles to 0 gap; regenerated T02386 mask.
2. `pipeline._annotate_catalog_meta`: added per-image `DATE-{nnn}` +
   exposure-averaged `DATE-OBS` / `MJD-OBS`. Verified on T02386 (23 images).
3. `calibration.py`: ZP "fitting" log now shows the instrumental column
   (per-aperture clarity); pre-filter NaNs before `sigma_clipped_stats` to
   silence repeated "invalid values" warnings.

## 9b. Recent changes (2026-06-30)

1. `vac`: added a **binary EAzY photo-z backend** (`vac/photoz_binary.py:
   run_eazy_binary`) and made it the default via `VACConfig.photoz_engine
   = "binary"` (+ `VACConfig.eazy_bin`). `vac/pipeline.py` dispatches on the
   engine; `validate(require_eazy_bin=...)` checks the executable; `report.py`
   logs the engine. Benchmarked on T22956 (803 rows): binary ≈115 s vs.
   eazy-py not finishing the template-grid build in 242 s. Diagnosis: the
   eazy-py slowness is the serial `TemplateGrid` build, not ZP offsets.

2. `vac/sedfit.py`: the FAST++ run was inheriting the slow shared
   `LIB/fastpp.param` defaults (`RESOLUTION='hr'`, `FORCE_ZPHOT=0`,
   `Z_STEP_TYPE=0`), making it many× slower than the legacy script.
   `_build_overrides` now sets the perf-critical keys from new `VACConfig`
   knobs (`fastpp_resolution='lr'`, `fastpp_force_zphot=True`,
   `fastpp_parallel='generators'`, `fastpp_metal=(.004,.008,.02,.05)`,
   `z_step_type=1`). T22956: ~3m11s (grid 1,625,888 = ntau4·nmetal4·nage11·
   nav31·nz298), matching the original.

3. Live logs: both external tools now stream stdout/stderr to a saved file
   (`eazy/<tile>/<tile>_<ref>_eazy.log`, `fastpp/<tile>/<catalog>_fastpp.log`)
   instead of buffering in memory. The `*_vac.log` run log is still written
   only once at the end of a successful run.

4. `vac/config.py`: added an up-front **preflight** sanity check
   (`VACConfig.preflight()` / `.check_requirements()`), called at the start of
   `run_value_added`. It collects **all** missing config/template/binary
   inputs at once (LIB tree, SFD, EAzY templates+prior+IGM+binary, FAST++
   binary + `fastpp.param` + `TEMPLATE_ERROR.fast.v0.2` from `fastpp_share`
   + SPS library dir; deep mode also checks each spectrum in
   `eazy_v1.2_dusty.spectra.param`) and raises one `FileNotFoundError` listing
   them — fixing cross-machine path failures surfacing mid-run. `validate()`
   is now a back-compat wrapper over `preflight()`.

5. `vac/vizier.py`: internalized the VizieR querying (presets + tile-polygon
   query/download) into the package; dropped the dependency on the external
   `Query/Vizier_Query.py` and removed `VACConfig.vizier_query_path`. Added
   `astroquery` to the `vac` extra (imported lazily). Verified a live REGALADE
   download for T22956 (4071 sources).

## 9c. Recent changes (2026-08-29, v0.6.0)

Promoted the machinery that had accumulated in the RIS production driver
(`RIS/script/RIS_vac.py`) into the package. The driver went from ~1030 to
~470 lines and no longer monkey-patches `vac.fluxes` / `vac.crossmatch`
module tables, nor imports private names (`_select_tile_row`).

1. `VACConfig.vhs_mag_set` (+ `resolve_vhs_mag_set`, `VHS_STAGING`,
   `vhs_staging`, `fluxes.VHS_MAG_COLUMNS`, `fluxes.external_columns`): the
   VHS aperture pairing described in §7. Replaces a context manager in the
   driver that rebound `fluxes._EXTERNAL_COLUMNS` per stage.
2. `vizier.prefetch_references` + `time_limit` + `tile_query_box`: bounded
   up-front staging, the `cos(dec)` query-box fix, and the `SIGALRM` cap.
   Presets `vhs_ap6` (superset of `vhs`) and `catwise` added.
3. `fluxes.drop_dead_bands` / `live_bands` +
   `run_value_added(drop_empty_bands=True)`, and
   `VACConfig.min_7ds_band_fraction` + `fluxes.required_filter_count`: the
   dead-band trim and the 7DS-relative coverage cut. The driver's
   "fraction offset by half a filter to land on the intended integer" hack is
   gone — the requirement is now an integer count computed directly.
4. `pipeline.run_value_added_split` is official API, along with
   `select_tile_row`, `fluxes.write_flux_inputs` and
   `build_flux_catalog(write=False)`. When the two stages want different VHS
   sets, `_match_vhs_mag_set` prefers the ap6 staging (a superset) but yields
   to whichever file is actually staged: matching an absent ap6 file would
   drop the NIR bands from *both* stages.
5. Fixed: Vega→AB offsets were registered only for the Petrosian column
   names, so an ap6 run would have left the NIR points on the Vega scale
   (J off by 0.94 mag).
6. `VACConfig.reference_path(key, tile)` for per-tile reference paths by
   preset key, with a generic `{catalog_dir}/{key}/{tile}_{key}.fits`
   fallback.

Verified end-to-end on T16088 (32558 rows → 1642 matched galaxies, 30
filters, coverage cut ≥18): full EAzY + FAST++ run, and the split path with
photo-z on `aper05c` 7DS-only vs. SED fit on `autoc` + VHS/GALEX/WISE.

## 9d. Recent changes (2026-09-11, v0.7.0)

Per-band mask flags from py7DT count-map MEFs (request from the mask
producer; design agreed: pipeline keeps MEF count maps, phot7ds converts to a
bitmask internally). Details in §5 and README "Per-band mask flags".

1. `masks.py`: `build_band_bitmask`, `resolve_count_masks`/`find_count_mask`,
   `MaskStaging`, `sweep_stale_staging`, `rename_mask_columns`,
   `any_band_nodata`. `sepp.FlagImage` + `build_sepp_command(flag_images=)`.
   `PhotometryConfig`: `per_band_masks`, `count_mask_suffix`,
   `mask_use_nused`, `mask_staging_dir`, `mask_workers`, `mask_flag_type`.
   `run_photometry(count_masks=)`.
2. Schema: `mask_flags_<band>`, `mask_npix_<band>` canonical;
   `isophotal_image_flags_cover` pair removed (`schema.LEGACY_COVER_COLS`,
   only with `per_band_masks=False`). VAC `_load_catalog` falls back to
   "bit 128 in any band" when the cover column is absent.
3. `build_coverage_mask` also flags non-finite pixels.
4. Experiment record: `RIS/script/test/bitmask_experiment/` (`notes.txt`,
   catalogs, `figures/`), prototype `RIS/script/test/bitmasktest.py`. T08147
   coadds under `/lyman/data2/RIS/data/T08147/` were **NaN-filled copies**
   (symlinks replaced; originals listed in `ORIGINAL_SYMLINKS.txt`) because
   the delivered coadds still had NaN outside/inside the footprint (pipeline
   version issue; producer was asked to interpolate).
5. Per-band results (m525, 33 129 sources): flags==0 31 %, <4 48 %, NODATA
   17 180, BADPIX 5 314, OUTLIER 1 549; inside every footprint 6 639 (union
   flag had over-flagged 1.7k–12.2k per band). Partial-MEF run (only m525 MEF)
   in `catalog_v070_partial/`.

## 9e. Recent changes (2026-10-02, v0.8.0)

1. **SE++ measurement gain / saturation fix.** Measurement images used the
   detection image's `GAIN` (SWarp "maximum equivalent gain", 28039 on
   T08147) and `SATLV` (absent in 7DS coadds → 10000 for every band). Now
   per image: gain = `EGAIN` → `GAIN` → 0 (`images.extract_gain_values`),
   saturation = `SATURATE` → `SATLV` → default
   (`extract_band_names_and_saturation`); `generate_sepp_python_config`
   sets `img.gain` / `img.saturation` per image. 7DS coadd headers have
   `EGAIN` + `SATURATE` only (no `GAIN`, no `SATLV`). Header `EGAINnnn` /
   `SATURnnn`, manifest `measurement_images`. A/B on T08147 (g, m875):
   fluxes identical, Δσ² = flux/EGAIN, SATURATED flag g 0→71, m875 14→0.
   Detection-image gain/saturation unchanged (still detection `GAIN` /
   `SATURATE`). The same fix was applied to the standalone
   `RIS/script/Utils_7DT.py` (not in git; backup
   `legacy/Utils_7DT_261002.py`); `RIS_catalog_sepp.py` still passes the
   detection gain as a scalar.
2. **REGALADE z_spec** (`VACConfig.use_regalade_zspec`, default `False`;
   `regalade_spec_codes`): spectroscopic-code galaxies get `z_spec` in the
   flux `.cat` (others −1); FAST++ fits at z_spec, EAzY only reports it.
   Off → no `z_spec` column. `RIS_vac.py` turns it on by default
   (`USE_ZSPEC = True`, `--no-zspec` to disable).

## 9f. Recent changes (2026-10-02, v0.8.1) — log / manifest provenance

- `phot7ds/provenance.py`: `collect_provenance()` (version, git commit /
  describe / dirty, SE++ `--version`, Python) → log `Software:` line +
  manifest `"provenance"`. Releases are annotated tags `vX.Y.Z` (v0.8.1 on
  its release commit; v0.6.0, v0.7.1 and v0.8.0 are untagged), so
  `describe` matches the version on a tagged checkout.
- `calibration.calibrate_zeropoints` stores `cat.meta["zp_solutions"]
  [(aper, band)]` (constant ZP/err/n, `Polynomial2D` coeffs by name, RMSE,
  n_input/n_fit/n_rmse, fit x/y range, surface min/med/max, selection) and
  logs `ZP <band> <aper>: constant ...` + `... coeffs ...`; the pipeline
  copies it to manifest `"zeropoints"` (`"aper05__m875"`) before the
  dict-valued meta is dropped. Verified on T08147 (g, m875): rebuilding
  `aper05c_mag_m875` from raw + manifest coeffs gives max |diff| = 0.
- Open (user to decide): the full-catalog constant-ZP column
  `{aper}_mag_err_*` adds `zp_rmse` (spatial RMSE), not `zperr` (constant
  scatter), unlike the calibration subset and the docstring.

## 9g. Recent changes (2026-10-05, v0.8.2) — user report fixes

From a user report (`RIS/bin/phot7ds_claim*.png`, items 2 and 6):

- Detection gain = `EGAIN` → `GAIN` → 0 (`images.read_detection_gain`),
  same order as the measurement images; was `GAIN` → 1.0 (DELVE leftover).
  No branching on `detection_label`: DELVE mosaics have only `GAIN`
  (SWarp max-equivalent, e.g. 131.6) and fall through; phot7ds SWarp white
  stacks have both (`GAIN` ≈ 3.9e4, `EGAIN` ≈ 3.4) and now use `EGAIN`.
  `DETGAIN` header card, manifest `detection_gain_keyword`.
- `tile_geometry`: polygon trim on the gnomonic plane about the spherical
  mean of the corners (`tile_center`, `gnomonic`, `tile_plane`). Fixes the
  145 tiles straddling RA 0 (kept 0 reference stars) and polar tiles
  (`T00000` kept 0; `T00001` kept the wrong side of the pole).
  `vac.vizier.tile_query_box` uses the same tangent-plane extent. Ordinary
  tiles: <1 % edge differences.
- `detection.sevends._reference_frame`: pixel scale from
  `proj_plane_pixel_scales` (PC + `CDELT1 = 1` headers read 3600″/pix).
- Report items 1 (new 41-filter set), 3–5 deferred; item 1 targeted at v0.9.

## 9h. Recent changes (2026-10-05, v0.9.0) — 42-filter set

- **Filter registry** `phot7ds/data/filters_7ds.ecsv` (42 rows: name, kind,
  pivot, FWHM, header key) is the single source for filters; built from
  `RIS/config/Filter_transmission/*.csv` with `python -m phot7ds.filters`.
  `filters.py` (`DEFAULT_BANDS`, `get_filter_definitions`),
  `photconv.filter_colorization`, `depth` header keys and
  `sevends medium_only` all read it. Override: `PhotometryConfig.filter_registry`
  or `$PHOT7DS_FILTER_REGISTRY`. Header-key rule for new rows
  (`filters.default_header_key`): `g`→`G`, `m425`→`425`, `m425w`→`42W`.
- Not migrated on purpose: `vac.config.DEFAULT_MEDIUM_BANDS` (legacy 20
  bands). It only picks the crossmatch dedup band (`medium_bands[mid]` ≈
  `m650`); switching it to the registry would move that band.
- Image selection (`images.select_images_by_filter`): `FILTER` card first,
  filename token as fallback, registry gate, optional dedup; every drop is a
  WARNING plus `dropped_images` in the manifest. Run raises only when none left.
- `catalog_io`: `output_format` auto/fits/parquet (999-column limit), suffix
  follows the format, stale other-format sibling removed; `read_catalog()`
  normalises Parquet `(value, comment)` meta. Schema with all 42 bands is
  1081 columns → Parquet under auto.
- `UNCALBND` / manifest `uncalibrated_bands` for bands without a ZP (e.g.
  the Gaia XP synphot reference lacks the 14 new filters: m386 m438 m466w
  m483 m534 m561 m586 m615 m640 m661 m692w m710w m769w m832w). The
  reference-catalog location for the new filters is still to be updated by
  the user.

## 9i. Recent changes (2026-10-05, v0.9.1) — empty-aperture depth fix

- Bug: `empty_aperture_sky_sigma` summed raw pixels without subtracting a
  background, so large-scale sky structure (strongest in red bands)
  inflated σ_aper. UL5RM ran up to 0.5 mag too shallow (m850 old 16.34 vs
  header 16.85).
- Fix: `depth.mesh_background` (SExtractor mode estimator per cell, median
  filter, bicubic spline) uses the run's `background_cell_size` /
  `smoothing_box_size`, as SE++ does. Apertures that touch no-data pixels
  are rejected. `background_cell_size=None` gives the old behaviour.
- The remaining offset (T00236 median UL5RM − header = −0.15) is
  correlated noise: the aperture/pixel σ ratio is 1.0 at r=1 px and ~1.2
  at r=5″ apertures. It is real. The coadd `UL5_*` and the error curve
  both assume white noise. New `UL{N}WM{key}` (white-noise depth) agrees
  with the header within +0.05 mag. Manifest depth entries gain
  `depth_white`, `white_sigma`, `pixel_sigma`, `correlation_ratio`,
  `background`.

## 10. Open / possible next steps

- Gaia XP reference with the 14 new filters (location TBD by the user).

- Consider lowering `fill_delve_detection_gaps` default overlap for *full*
  rebuilds (overlap 0.5 → 192 patches for an IMS tile; fine for gap-fill, heavy
  for full builds).
- `IMS_detect.py` `FULL_REGEN` path uses the auto grid; tune overlap if used.
