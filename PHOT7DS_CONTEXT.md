# phot7ds — Session Context / Handoff

Working memory for the `phot7ds` package and the 7DS/RIS scripts around it.
Read this first when starting a new session. Version at time of writing:
**phot7ds 0.5.0** (`phot7ds/__init__.py`, `pyproject.toml`).

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
  sepp.py            # SE++ output parsing; aperture labels zero-padded ("05")
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
  `CATALOG_PRESETS` for regalade/vhs/galex + `query_vizier_catalog_for_tile`
  / `download_catalog_for_tile`); `astroquery` is imported lazily and is in
  the `vac` extra. No longer depends on the external `Query/Vizier_Query.py`
  (the old `VACConfig.vizier_query_path` field was removed). Polygon trim
  reuses `phot7ds.tile_geometry.trim_to_tile_polygon`.
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

## 10. Open / possible next steps

- Consider lowering `fill_delve_detection_gaps` default overlap for *full*
  rebuilds (overlap 0.5 → 192 patches for an IMS tile; fine for gap-fill, heavy
  for full builds).
- `IMS_detect.py` `FULL_REGEN` path uses the auto grid; tune overlap if used.
