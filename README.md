# phot7ds

Multi-band photometric catalog pipeline for the **7DS** survey suite
(RIS — Reference Imaging Survey, WFS — Wide-Field Survey, IMS — Intensive
Monitoring Survey).

Given a list of 7DS science images, a detection image, a SourceExtractor++
config and a Gaia XP synphot reference catalog, `phot7ds.run_photometry()`
returns a single zero-point-calibrated FITS catalog. By default the table
keeps the SE++ column layout plus calibration columns; pass
`standardize_catalog=True` if you want the fixed canonical column order with
placeholder columns for missing bands.

## Layout

```
phot7ds/          # repository root (this package)
├── README.md
├── pyproject.toml
├── examples/
│   ├── example_run.py                  # end-to-end photometry example
│   ├── example_vac.py                  # end-to-end value-added-catalog example
│   └── config/                         # SE++/SWarp configs, tile table, Gaia XP, DELVE cache
│       ├── README.md
│       └── column_convention.md
├── tests/
│   ├── test_smoke.py                   # smoke tests (no SE++/network)
│   └── test_vac_smoke.py               # vac smoke tests (no eazy/sfdmap/FAST++)
└── phot7ds/
    ├── __init__.py
    ├── _logging.py
    ├── config.py                       # PhotometryConfig (frozen dataclass)
    ├── config_io.py                    # ensure_* / require_* helpers
    ├── presets.py                      # per-detection-image SE++ presets
    ├── filters.py                      # 7DS filter definitions, DEFAULT_BANDS
    ├── tile_geometry.py                # tile polygon trim
    ├── images.py                       # organise-by-filter, coverage-mask builder
    ├── sepp.py                         # SourceExtractor++ config, command, post-process
    ├── calibration.py                  # Gaia XP loader, NN matching, spatial ZP
    ├── depth.py                        # 5-sigma depth (curve fit / empty-aper) + ZP header keys
    ├── diagnostics.py                  # residual-map plot
    ├── schema.py                       # canonical schema, standardize, load
    ├── crossmatch.py                   # sky cross-matching (matching())
    ├── photconv.py                     # mag/flux conversions, filter colorization
    ├── pipeline.py                     # run_photometry()  ← main entry
    ├── batch.py                        # batch_run(jobs=[...])
    ├── detection/
    │   ├── delve.py                    # DELVE patch download + SWarp coadd
    │   └── sevends.py                  # 7DS native white detection image (SWarp)
    └── vac/                            # value-added catalog subpackage (optional extra)
        ├── config.py                   # VACConfig (dataclass)
        ├── vizier.py                   # on-demand external-catalog download
        ├── crossmatch.py               # REGALADE/VHS/GALEX matching + dedup
        ├── fluxes.py                   # auto-detect filters, extinction-corrected fluxes
        ├── photoz.py                   # eazy-py photo-z (m625 prior, z_m2/z_a)
        ├── sedfit.py                   # FAST++ SED fitting
        ├── catalog.py                  # merge photo-z + SED fits
        ├── report.py                   # per-run log file
        └── pipeline.py                 # run_value_added()  ← VAC entry
```

## Installation

Clone from GitHub ([jmk5040/phot7ds](https://github.com/jmk5040/phot7ds)) and
install in editable mode (recommended):

```bash
git clone https://github.com/jmk5040/phot7ds.git
cd phot7ds
pip install -e ".[dev]"
python -c "import phot7ds; print(phot7ds.__version__)"
```

Without installing, add the repo root to `PYTHONPATH`:

```bash
export PYTHONPATH=/path/to/phot7ds:$PYTHONPATH
python -c "import phot7ds; print(phot7ds.__version__)"
```

Run the smoke tests:

```bash
cd phot7ds
python -m pytest tests/ -v
```

External (non-Python) dependencies:

- [`sourcextractor++`](https://sextractor.readthedocs.io/) — the
  `sourcextractor++` command must be on `$PATH`.
- [`SWarp`](https://www.astromatic.net/software/swarp/) — only required if
  you ask `phot7ds.detection` to build a DELVE detection image.

Python dependencies: `numpy`, `astropy`, `pandas`, `scipy`, `matplotlib`,
`requests`, `pyvo`.

## Usage

### Single run

```python
from phot7ds import run_photometry

result = run_photometry(
    science_images=[
        "/data/.../T01234_g.fits",
        "/data/.../T01234_r.fits",
        "/data/.../T01234_m500.fits",
    ],
    detection_image="/data/.../detection.fits",
    reference_catalog="/data/.../gaiaxp_T01234.csv",
    catalog_path="/data/.../catalog/7ds/T01234_20260512_DELVE.fits",
    output_dir="/data/.../results/sepp/T01234",
    sepp_config_file="/data/.../config/7ds_sepp.config",
    # detection_label selects an SE++ preset (see "Tuning SE++"):
    detection_label="DELVE",
    # any preset value can be overridden inline:
    fixed_apertures_arcsec=(5.0, 10.0),
    save_residual_plots=True,
    thread_count=8,
)

print(result.catalog_path)   # /data/.../catalog/7ds/T01234_20260512_DELVE.fits
print(result.manifest_path)  # JSON snapshot of the inputs + config
print(result.log_file)
print(result.n_sources)
```

The raw SE++ output is written alongside the final catalog as
`{run_name}_raw.fits`, so the two are easy to tell apart on disk.

Required keyword arguments:

1. `science_images` — list of science FITS paths (filter from `FILTER` header).
2. `detection_image` — detection FITS path.
3. `reference_catalog` — Gaia XP synphot CSV path.
4. `sepp_config_file` — SourceExtractor++ `--config-file` path.
5. `output_dir` — working directory (created automatically if missing).
6. Name the final catalog with **either**:
   - `catalog_name` — basename only, written inside `output_dir` (typical).
     The leaf is kept **verbatim** (no forced `_phot.zp.fits` suffix);
     `.fits` is appended when missing:

```python
run_photometry(
    ...,
    output_dir="/data/.../results/sepp/T01234",
    catalog_name="test_zp.fits",       # -> output_dir/test_zp.fits
    # or simply:                       # -> output_dir/T01234_run01.fits
    catalog_name="T01234_run01",
)
```

   - `catalog_path` — full path (alternative; do not pass both). Again
     `.fits` is appended if absent.

Alternatively, omit both and pair `run_name` with `output_dir`; the
default name becomes `{output_dir}/{run_name}.fits`.

See `help(phot7ds.run_photometry)` for all optional knobs.

### Canonical output schema (optional)

`standardize_catalog` defaults to **`False`**. When left at the default, the
written FITS file is the SE++ catalog after per-filter column splitting,
Gaia XP zero-point calibration, and unit cleanup — no placeholder columns are
added.

Set `standardize_catalog=True` to reshape the table to the unified schema
(same column order every time, missing bands filled with `-99.0`
placeholders):

```python
result = run_photometry(
    ...,
    standardize_catalog=True,
)
```

You can also call `phot7ds.standardize_catalog()` on an existing catalog
outside the pipeline.

### Tile-polygon trim of the reference catalog

If your tile geometry is known, pass a single-row table with
`ra1..ra4/dec1..dec4` as `tile_info=`; the Gaia XP reference is trimmed to
the polygon before matching:

```python
from astropy.table import Table
tile_tbl = Table.read("/data/.../7DT_tiles.fits")
tile_info = tile_tbl[tile_tbl["tile"] == "T01234"]

result = run_photometry(
    ...,
    tile_info=tile_info,
)
```

### Reusing the same knobs (`PhotometryConfig`)

For repeatable runs, bundle the tuning knobs into a
`PhotometryConfig`. Any explicit `run_photometry` kwarg always overrides
the corresponding config field.

```python
from phot7ds import PhotometryConfig, run_photometry

cfg = PhotometryConfig(
    sepp_config_file="/data/data1/7DS/RIS/config/7ds_sepp.config",
    detection_label="DELVE",
    detection_threshold=10.0,
    fixed_apertures_arcsec=(5.0, 10.0),
    save_residual_plots=True,
    thread_count=8,
)

result = run_photometry(
    science_images=[...],
    detection_image=...,
    reference_catalog=...,
    output_dir=...,
    config=cfg,
    # any kwarg here overrides the same field on cfg:
    detection_threshold=8.0,
)
```

### Batch processing

`batch_run` is a thin loop with shared defaults and error handling:

```python
from phot7ds import PhotometryConfig, batch_run

cfg = PhotometryConfig(
    sepp_config_file="/data/data1/7DS/RIS/config/7ds_sepp.config",
    detection_threshold=10.0,
)

jobs = [
    dict(
        science_images=[...],
        detection_image="/data/.../T01_det.fits",
        reference_catalog="/data/.../gaia_T01.csv",
        output_dir="/data/.../out/T01",
        run_name="T01_run01",
    ),
    dict(
        science_images=[...],
        detection_image="/data/.../T02_det.fits",
        reference_catalog="/data/.../gaia_T02.csv",
        output_dir="/data/.../out/T02",
        run_name="T02_run01",
    ),
]

results = batch_run(jobs, config=cfg, thread_count=8)
# - `config=` and `thread_count=` are applied to every job
# - per-job keys always win over shared defaults
# - on_error="continue" (default) records failures and proceeds; "raise" stops

for r in results:
    label = r.job.get("run_name")
    if r.status == "ok":
        print(f"OK   {label} -> {r.result.catalog_path}")
    else:
        print(f"FAIL {label}: {r.error}")
```

### Detection-image construction

The DELVE detection-image builder is callable directly:

```python
from astropy.table import Table
from phot7ds.detection import build_delve_detection_image

tile_tbl = Table.read("/data/.../7DT_tiles.fits")
tile_info = tile_tbl[tile_tbl["tile"] == "T01234"]

detection_img, weight_img = build_delve_detection_image(
    tile_info=tile_info,
    imgtype="image",
    output_path="/data/.../detect_imgs/T01234",
    swarp_cfg_path="/data/data1/7DS/RIS/config/7ds.swarp",
)
# SWarp center defaults to round(tile_info['ra'], 4) / round(tile_info['dec'], 4)
# converted to sexagesimal. Override with ra_center= / dec_center= if needed.
```

`imgtype="mask"` builds a bad-pixel mask suitable for passing as
`badpix_mask=` to `run_photometry`.

### Reading the output catalog

If the catalog was written with `standardize_catalog=True`, missing canonical
columns use a finite sentinel (`PLACEHOLDER_FILL = -99.0`) so they read back
as plain Columns (not `MaskedColumn`). Use `load_unified_catalog` if you'd
rather see those as NaN:

```python
from phot7ds import load_unified_catalog
cat = load_unified_catalog("/data/.../T01234_..._phot.zp.fits", fill_nan=True)
print(cat.meta)   # {'NPLACE': ..., 'NEXTRA': ..., 'NDUPS': ...}
```

## Output schema

When `standardize_catalog=True`, every catalog has the same columns in the
same order:

1. A fixed set of detection / geometry columns
   (`phot7ds.schema.CANONICAL_BASIC_COLS`): IDs, world/pixel centroids, error
   ellipse, ellipse parameters, source flags, area, elongation, etc.
2. For each `aperture` x `band`:
   `{aperture}_{quantity}_{band}` where `quantity ∈ {flux, flux_err, mag,
   mag_err, flags}`.
3. For each magnitude column, a spatially-corrected counterpart
   `{aperture}c_mag_{band}` and `{aperture}c_mag_err_{band}`.
4. For each `flux_fraction`: `flux_rad_{int(f*100)}_{band}`.
5. Any extra columns SE++ produced beyond the schema are kept at the end
   and counted in `meta["NEXTRA"]`.

Bands missing from the input image set are inserted as **placeholder
columns** (value `-99.0`, `description == "PLACEHOLDER"`); the count is in
`meta["NPLACE"]`.

With the default `standardize_catalog=False`, column names and order follow
SourceExtractor++ and the calibration step instead; only the bands present in
`science_images` appear (plus spatially corrected `{aperture}c_mag_{band}`
columns where calibration ran).

## Calibration

For each `(band, aperture)`:

1. Targets are matched to the Gaia XP synphot reference within
   `match_radius_arcsec` (default 1″).
2. The calibration subset requires:
   - matched within the radius,
   - `source_flags == 0` and per-band per-aperture `flags == 0`,
   - reference magnitude inside `mag_range` (default 12–16).
3. A 2-D polynomial zero-point surface is fit (`spatial_poly_degree`,
   default 2). The spatially-corrected magnitude is written to
   `{aperture}c_mag_{band}` on the full catalog.
4. A constant zero-point (sigma-clipped median offset) is computed and
   applied to `{aperture}_mag_{band}`; its scatter is added in quadrature
   to `{aperture}_mag_err_{band}`.

Set `save_residual_plots=True` to drop residual maps into
`<output_dir>/figures/` (one PNG per band/aperture).

After calibration, the constant ZP and its sigma-clipped scatter are
written to the output FITS header as `ZP<APER>M<BAND>` and
`ZE<APER>M<BAND>` (e.g. `ZP05MG`, `ZE05M575`). Aperture tokens are `05`
(5″), `10` (10″), `AU` (`auto`); band tokens are `G`/`R`/`I`/`Z` for
broadband and the 3-digit central wavelength for medium bands.

## 5-sigma depth

`run_photometry()` estimates the 5-sigma photometric depth per band right
after calibration. Two independent methods are computed and logged side
by side:

1. **Magnitude-error curve fit** — fits `magerr = a · exp(b · mag)` to
   point-like, well-detected sources and returns the magnitude where
   `magerr ≈ 1.0857/N` (= 0.217 mag for `N=5`). Pure-catalog, no I/O.
2. **Empty-aperture sky sigma** — drops ~2000 random circular apertures
   on each science image at positions that avoid catalog sources, takes
   the sigma-clipped stddev of the summed aperture fluxes as `σ_aper`,
   and reports `ZP − 2.5·log10(N · σ_aper)`. Captures correlated
   background noise without the cost of a full background-RMS check
   image.

Results are written to:

- The run **log** (formatted table side-by-side).
- The **manifest JSON** (`*_manifest.json` -> `"depths"`).
- The **FITS catalog header**, as 8-character keys for the canonical 5″
  aperture:
  - `UL{N}EM{BAND}` -- N-sigma depth from the magnitude-error curve fit
    [mag].
  - `UL{N}RM{BAND}` -- N-sigma depth from the empty-aperture / background
    RMS sampling [mag].
  - `BRMSM{BAND}`   -- the empty-aperture sky sigma in ADU, useful for
    recomputing the limiting magnitude with an updated ZP.

Tune via `depth_n_sigma`, `depth_apertures`, `depth_n_empty_apertures`,
or `depth_seed`; turn the empty-aperture pass off with
`depth_empty_aperture=False`, or disable depth estimation entirely with
`estimate_depth=False`.

## Tuning SE++ (detection presets)

`PhotometryConfig.detection_label` selects a built-in SE++ tuning preset
(see `phot7ds.presets`). When a tuning field is left at its default
(`None`) on the merged config, it is filled from the preset; any value
you set explicitly **always** wins.

| Field                          | `DELVE` (default) | `7DT`    |
| ------------------------------ | ----------------- | -------- |
| `detection_threshold`          | `10.0`            | `1.5`    |
| `detection_minimum_area`       | `9`               | `9`      |
| `auto_kron_min_radius`         | `8.0`             | `3.5`    |
| `partition_threshold_count`    | `32`              | `32`     |
| `partition_minimum_contrast`   | `5.0e-4`          | `1.0e-5` |

```python
# DELVE detection image (defaults applied automatically)
cfg = PhotometryConfig(
    sepp_config_file=...,
    detection_label="DELVE",
)

# 7DT detection coadd
cfg = PhotometryConfig(
    sepp_config_file=...,
    detection_label="7DT",
)

# Override a single value (the rest comes from the preset):
cfg = PhotometryConfig(
    sepp_config_file=...,
    detection_label="DELVE",
    detection_threshold=8.0,
)
```

Add a new preset by extending `phot7ds.presets.DETECTION_PRESETS`.

## FITS catalog header

The final calibrated catalog carries provenance / coverage metadata in
its primary header. Selected keys (all 8-char, no `HIERARCH`):

| Key        | Meaning                                          |
| ---------- | ------------------------------------------------ |
| `PHOTVER`  | phot7ds package version                          |
| `PHOTRUN`  | Run name (stem of this catalog)                  |
| `PHOTDATE` | Catalog write timestamp (UTC, ISO-8601)          |
| `PHOTUSR`  | Username that produced the catalog               |
| `PHOTHOST` | Hostname that produced the catalog               |
| `DETLABEL` | `'DELVE'` or `'7DT'`                             |
| `DETIMG`   | Detection image basename                         |
| `COVMASK`  | Coverage mask basename                           |
| `BADPMASK` | Bad-pixel mask basename (if any)                 |
| `REFCAT`   | Reference catalog basename                       |
| `NSCIIMG`  | Number of measurement (science) images           |
| `SCIMGNNN` | Per-image basename (`NNN` = zero-padded index)   |
| `MSKRATIO` | Ratio of pixels masked in the coverage mask     |
| `DETTHR`   | SE++ detection threshold (σ)                     |
| `DETMINAR` | SE++ detection minimum area (pix)                |
| `KRNMINR`  | SE++ auto-kron minimum radius (pix)              |
| `PARTMINC` | SE++ partition minimum contrast                  |
| `FIXAPER`  | Fixed apertures (arcsec, comma-separated)        |
| `PIXSCALE` | Pixel scale (arcsec/pix)                         |

Per-(aperture, band) ZP, ZP scatter and 5-σ depths follow the
`ZP{AP}M{BND}`, `ZE{AP}M{BND}`, `UL{N}EM{BND}`, `UL{N}RM{BND}` patterns
(see [Calibration](#calibration) and [5-sigma depth](#5-sigma-depth)).

> Aperture column names use the **zero-padded** convention: `aper05_`,
> `aper10_`, `auto_`, `autoc_`. See
> `examples/config/column_convention.md` for the full cheat-sheet.

## Value-added catalog (`phot7ds.vac`)

`phot7ds.vac` turns a phot7ds photometric catalog into a **value-added
catalog**: it cross-matches galaxies against external references, builds an
extinction-corrected flux catalog, runs photometric redshifts with
[`eazy-py`](https://github.com/gbrammer/eazy-py), runs SED fitting with the
compiled [`FAST++`](https://github.com/cschreib/fastpp) binary, and merges
everything back onto the input catalog.

### Installation (extra dependencies)

The subpackage needs heavy optional dependencies plus the external `fast++`
binary. The heavy imports are deferred to call time, so plain
`import phot7ds` (and even `import phot7ds.vac`) keeps working without them;
a clear error is raised only when a stage that needs them actually runs.

```bash
pip install -e ".[vac]"     # eazy, sfdmap2, extinction
```

You also need:

- the compiled `fast++` binary (default path
  `/home/jmkastro/fastpp/bin/fast++`, override via `VACConfig.fastpp_bin`),
- the EAzY/FAST++ config tree under `lib_dir` (templates, `FILTER.RES.latest`,
  `default.translate`, priors, SPS libraries),
- SFD dust maps (default `lib_dir/sfddata`, override via `sfd_dir`).

### Quick start

```python
from phot7ds.vac import VACConfig, run_value_added

config = VACConfig(
    lib_dir="/data/data1/7DS/RIS/config/LIB",
    catalog_dir="/data/data1/7DS/RIS/catalog",
    output_root="/data/data1/7DS/RIS/results/vac",
    fastpp_bin="/home/jmkastro/fastpp/bin/fast++",
    detection_ref="7DS",
    aperture="aper05c",
    auto_download=True,   # fetch missing REGALADE/VHS/GALEX via VizieR
    prior_band="m625",    # default 7DS m625 magnitude prior
    use_vhs=True, use_galex=True, use_wise=True,
)

result = run_value_added(
    catalog_path="/data/.../T00236/T00236_7DS_phot.fits",
    tile="T00236",
    tile_table="/data/data1/7DS/RIS/config/7DT_tiles.fits",
    config=config,
    do_photoz=True,
    do_sedfit=True,
)

print(result.value_added_path)  # merged catalog
print(result.log_path)          # human-readable run log
print(result.n_matched, result.n_flux)
```

A runnable version is in `examples/example_vac.py`.

### Preflight checks (config sanity)

`run_value_added` calls `config.preflight(...)` before any work begins, so
missing paths fail fast with a single, complete report instead of cryptic
mid-run errors (e.g. FAST++ being unable to open
`TEMPLATE_ERROR.fast.v0.2`). It verifies, for the requested stages, the LIB
tree (`FILTER.RES.latest[.info]`, `default.translate`, SFD maps), the EAzY
templates/prior/IGM files and binary, and the FAST++ binary, `fastpp.param`
template, template-error file (resolved from `fastpp_share`), and SPS
library dir — and (with `deep=True`) the individual template spectra listed
in `eazy_v1.2_dusty.spectra.param`, which catches a LIB tree copied to a
different root. Per-tile reference catalogs (REGALADE/VHS/GALEX) are checked
during the run, not here.

Run it yourself before a batch:

```python
problems = config.preflight(do_photoz=True, do_sedfit=True, strict=False)
if problems:
    for p in problems:
        print("MISSING:", p)
```

With `strict=True` (the default) it raises `FileNotFoundError` listing every
problem; `config.check_requirements(...)` is the non-logging, non-raising
variant. Common fixes are `VACConfig.fastpp_share_dir` (when the `fast++`
install prefix differs from the default `<fastpp_bin>/../../share`),
`fastpp_library_dir`, `eazy_bin`, `lib_dir`, and `sfd_dir`.

### Pipeline stages

1. **Cross-match** (`crossmatch.py`) — match the catalog to REGALADE
   (required; WISE `W1`/`W2` come bundled in its columns), then optionally
   VHS (NIR, Vega→AB corrected) and GALEX (UV). Duplicate REGALADE
   associations collapse to the brightest 7DS match.
2. **Flux assembly** (`fluxes.py`) — build the EAzY/FAST++ `.cat`
   (AB zeropoint 25) with Galactic-extinction correction (SFD + Fitzpatrick99
   at the tile center).
3. **Photo-z** — on the auto-detected filter set, with the default m625
   prior. Two interchangeable backends (`VACConfig.photoz_engine`):
   - `"binary"` (**default**, `photoz_binary.py`) — shells out to the
     compiled EAzY executable (`VACConfig.eazy_bin`). Fast (≈2 min for ~800
     sources) and the recommended path.
   - `"eazy-py"` (`photoz.py`) — the pure-Python `eazy.photoz.PhotoZ`. Kept
     for reference, but its `TemplateGrid` build is pathologically slow for
     the 7DS medium-band filter set (a single template can take longer to
     integrate than the binary needs for the whole catalog), so it is not
     recommended for production runs.
4. **SED fit** (`sedfit.py`) — FAST++ for stellar mass, SFR, age, Av, ….
   The fit is anchored at the EAzY photo-z (`FORCE_ZPHOT=1`) and uses the
   fast low-resolution BC03 grid by default. The grid/performance knobs
   (`fastpp_resolution`, `fastpp_force_zphot`, `fastpp_parallel`,
   `fastpp_metal`, `z_step_type`) have defaults that reproduce the fast
   legacy configuration — without them the run inherits the slow
   `LIB/fastpp.param` defaults (`RESOLUTION='hr'`, `FORCE_ZPHOT=0`) and is
   many times slower. FAST++ stdout/stderr is streamed live to
   `<output_root>/fastpp/<tile>/<catalog>_fastpp.log`.
5. **Merge** (`catalog.py`) — `hstack` the photo-z and SED-fit outputs onto a
   carried-over id table and write the final FITS catalog plus a run log.

### Auto-detected filter set

The 7DS filters that enter the fit are read **directly from the catalog**:
every `{aperture}_mag_<band>` column with a matching `f_7DS_<band>` entry in
`default.translate` is used (ordered by central wavelength). External bands
(`f_W1/f_W2`, `f_VHS_*`, `f_FUV/f_NUV`) are added when their reference columns
are present **and** the corresponding `use_wise` / `use_vhs` / `use_galex`
toggle is on. The legacy `use_medium` / `use_broad` toggles are ignored.

### On-demand external catalogs (VizieR)

With `auto_download=True`, a per-tile reference catalog that is **absent** at
its expected path is fetched by the in-package VizieR querier
(`phot7ds.vac.vizier`, which needs `astroquery` from the `vac` extra) before
matching: the catalog is queried inside the tile bounding box and trimmed to
the tile polygon. Supported keys: `regalade`, `vhs`, `galex` (see
`vac.vizier.CATALOG_PRESETS`). Download failures are non-fatal for the
optional catalogs (the run proceeds and skips those bands); a missing
**REGALADE** catalog still raises.

### Magnitude prior and the redshift column

The default prior is the 7DS **m625** prior
(`templates/prior_m6250_extend.dat`, band `m625`). When the prior band's flux
(`f_7DS_m625`, i.e. `aper05c_mag_m625`) is present the prior is applied and
the redshift column handed to FAST++ is **`z_m2`**; when it is absent a
warning is logged, the run proceeds **without** a prior, and the column is
**`z_a`**. FAST++ `NAME_ZPHOT` is pointed at whichever column is produced.
Override with `prior_band=` / `prior_file=`.

### Photo-z engine (`photoz_engine`)

`VACConfig.photoz_engine` selects the redshift backend and defaults to
`"binary"`, which runs the compiled EAzY executable at
`VACConfig.eazy_bin` (default `/data/data1/7DS/RIS/config/eazy/src/eazy`).
The binary writes its native `.zout` — which already carries `z_m2`/`z_a`
plus the `l68/u68`, `l95/u95`, `l99/u99` confidence intervals FAST++ needs —
and that file is copied straight into the SED-fit directory. Set
`photoz_engine="eazy-py"` to use the pure-Python path instead (see below);
the `eazy_n_proc` knob only applies to that path.

### eazy-py parallelism (`eazy_n_proc`)

eazy-py's `TemplateGrid` (serial when `n_proc < 0`) and `fit_catalog`
(serial when `n_proc == 0`) use **inconsistent** conventions, and the
parallel template-grid build is prone to fork deadlocks (it hangs, then
raises a multiprocessing `TimeoutError`). `VACConfig.eazy_n_proc` defaults to
`-1`, which runs **both** stages serially — robust and fast for the small VAC
catalogs. Set it `>0` only if the parallel build is known to work in your
environment. (`n_proc` still controls FAST++ threads.)

### Outputs

```
<output_root>/
├── eazy/<tile>/      # EAzY .cat / zphot.param / OUTPUT/*.zout
├── fastpp/<tile>/    # FAST++ .cat / .zout / .param / .fout
└── value_added/
    ├── <tile>_<ref>_value_added.fits   # merged catalog
    └── <tile>_<ref>_vac.log            # run log
```

The run log records the timestamp, cross-match settings, the detected filter
list with central wavelengths and extinction, target counts, the full
eazy/photo-z configuration (engine, prior choice, redshift column), and the
full FAST++ configuration. Note this `*_vac.log` is written **once at the
end** of a successful run; for live progress during the (multi-minute)
photo-z and SED-fit stages, tail the per-stage logs that the external tools
stream as they run: `<output_root>/eazy/<tile>/<tile>_<ref>_eazy.log` and
`<output_root>/fastpp/<tile>/<catalog>_fastpp.log`.

### vac smoke tests

`tests/test_vac_smoke.py` covers the lightweight pieces (matching, flux
conversions, `VACConfig` validation/derived paths, filter auto-detection
toggles, VizieR preset mapping, the run-log writer) without importing the
heavy extras or invoking FAST++.

## Testing and examples

### Smoke tests (no data, no SE++, no network)

The included smoke tests cover the import surface, the schema, the config
machinery, the kwarg-vs-config merge, the polygon trim, and the batch loop.

```bash
cd phot7ds          # repository root
python -m pytest tests/ -v
```

### End-to-end example (`examples/example_run.py`)

`examples/example_run.py` is a runnable script for a **full** pipeline
on one tile: optional DELVE detection + mask build, then
`run_photometry` with the DELVE-built bad-pixel mask and residual plots.

All paths are **relative to the script** so the example works without
editing if the matching layout exists in `examples/config/`:

```
examples/config/
├── 7ds_sepp.config         # SE++ defaults    (auto-generated if missing)
├── 7ds.swarp               # SWarp defaults   (auto-generated if missing)
├── 7DT_tiles.ascii         # tile table       (USER-supplied, required)
├── gaiaxp/                 # Gaia XP CSVs     (USER-supplied, required)
│   └── gaiaxp_dr3_synphot_<TILE>.csv
└── DELVE/                  # cache for DELVE mosaics (auto-created)
    └── <TILE>/
        ├── <TILE>_DELVE_DR3_IMAGE_det.fits
        └── <TILE>_DELVE_DR3_MASK_det.fits
```

The script uses the bootstrap helpers exposed from the package:

```python
from phot7ds import (
    ensure_sepp_config,       # dumps SE++ defaults if config absent
    ensure_swarp_config,      # dumps SWarp defaults if config absent
    require_tile_table,       # raises FileNotFoundError when missing
    require_gaiaxp_reference, # raises FileNotFoundError when missing
)
```

So missing SE++/SWarp configs are produced on first run via
`sourcextractor++ --dump-default-config` / `SWarp -dd`, while missing
survey artefacts (tile table, Gaia XP CSVs) raise a clear error.

Science images must have `OBJECT` (tile id), `OBJCTRA` and `OBJCTDEC`
in the header when building DELVE images from the script.

**Run** (from the repository root, with `sourcextractor++` on `$PATH`):

```bash
cd phot7ds
pip install -e .    # or: export PYTHONPATH=$PWD:$PYTHONPATH

# optional: conda activate your env (e.g. 7ds)
python examples/example_run.py
```

By default the script calls `run_single()` (one `run_photometry` job).
Switch to the batch helper by calling `run_batch()` at the bottom of
the file.

**External requirements for this example** (unlike the smoke tests):

- [`sourcextractor++`](https://sextractor.readthedocs.io/) on `$PATH`
- [`SWarp`](https://www.astromatic.net/software/swarp/) when DELVE
  mosaics need to be built (`FORCE_BUILD_DELVE = True` rebuilds even
  cached coadds)
- Network access for DELVE patch download
- A Gaia XP synphot CSV per tile inside `examples/config/gaiaxp/`

**Typical output** (under `examples/example_run/` by default):

```
catalog : .../examples/example_run/test_zp.fits        # the calibrated catalog
manifest: .../examples/example_run/test_zp_manifest.json
log     : .../examples/example_run/test_zp.log
sources : <N>
```

The raw SE++ catalog is saved alongside as `test_zp_raw.fits`.

Set `standardize_catalog=True` in `run_single()` if you want the
canonical column layout (see
[Canonical output schema](#canonical-output-schema-optional)).