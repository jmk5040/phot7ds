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

The original `RIS_catalog_sepp.py` / `DELVE_DetImage.py` / `Utils_7DT.py`
scripts have been refactored into this reusable Python package; the legacy
files are kept in `../legacy/` for reference (with a `_260520` suffix).

## Layout

```
phot7ds/          # repository root (this package)
├── README.md
├── pyproject.toml
├── examples/example_run.py             # minimal Python example
├── tests/test_smoke.py                 # pytest smoke tests (no SE++/network)
└── phot7ds/
    ├── __init__.py
    ├── _logging.py
    ├── config.py                       # optional PhotometryConfig (frozen dataclass)
    ├── filters.py                      # 7DS filter definitions, DEFAULT_BANDS
    ├── tile_geometry.py                # tile polygon trim
    ├── images.py                       # organise-by-filter, coverage-mask builder
    ├── sepp.py                         # SourceExtractor++ config, command, post-process
    ├── calibration.py                  # Gaia XP loader, NN matching, spatial ZP
    ├── diagnostics.py                  # residual-map plot
    ├── schema.py                       # canonical schema, standardize, load
    ├── pipeline.py                     # run_photometry()  ← main entry
    ├── batch.py                        # batch_run(jobs=[...])
    └── detection/
        └── delve.py                    # DELVE patch download + SWarp coadd
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
    catalog_path="/data/.../catalog/7ds/T01234_20260512_DELVE_phot.zp.fits",
    output_dir="/data/.../results/sepp/T01234",
    sepp_config_file="/data/data1/7DS/RIS/config/7ds_sepp.config",
    # Any other knob can be passed inline:
    detection_label="DELVE",
    detection_threshold=10.0,
    fixed_apertures_arcsec=(5.0, 10.0),
    save_residual_plots=True,
    thread_count=8,
)

print(result.catalog_path)   # /data/.../out/T01234/<run_name>_phot.zp.fits
print(result.manifest_path)  # JSON snapshot of the inputs + config
print(result.log_file)
print(result.n_sources)
```

Required keyword arguments:

1. `science_images` — list of science FITS paths (filter from `FILTER` header).
2. `detection_image` — detection FITS path.
3. `reference_catalog` — Gaia XP synphot CSV path.
4. `sepp_config_file` — SourceExtractor++ `--config-file` path.
5. `output_dir` — working directory (created automatically if missing).
6. Name the final catalog with **either**:
   - `catalog_name` — basename only, written inside `output_dir` (typical):

```python
run_photometry(
    ...,
    output_dir="/data/.../results/sepp/T01234",
    catalog_name=f"{tile}_{date}_DELVE_phot.zp.fits",  # -> output_dir/<that name>
)
```

   - `catalog_path` — full path (alternative; do not pass both).

Alternatively, omit both and use `run_name` + `output_dir` (auto names
`{output_dir}/{run_name}_phot.zp.fits`).

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
    ra_center="12:34:56.7", dec_center="-01:02:03.4",
    imgtype="image",
    output_path="/data/.../detect_imgs/T01234",
    swarp_cfg_path="/data/data1/7DS/RIS/config/7ds.swarp",
)
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

## Tuning SE++

The defaults in `PhotometryConfig` are conservative. For a **DELVE**
detection image, the deeper / sharper imagery calls for aggressive cuts:

```python
PhotometryConfig(
    sepp_config_file=...,
    detection_threshold=10.0,
    auto_kron_min_radius=8.0,
    partition_minimum_contrast=0.0005,
)
```

For a **7DT detection coadd**:

```python
PhotometryConfig(
    sepp_config_file=...,
    detection_threshold=1.0,
    auto_kron_min_radius=3.5,
    partition_minimum_contrast=1e-5,
)
```

## Testing

The included smoke tests cover the import surface, the schema, the config
machinery, the kwarg-vs-config merge, the polygon trim, and the batch loop.
They never invoke `sourcextractor++` or fetch anything from the network.

```bash
cd phot7ds
python -m pytest tests/ -v
```

## Migrating from the legacy scripts

| Legacy entry point                 | New entry point                                                       |
| ---------------------------------- | --------------------------------------------------------------------- |
| `RIS_catalog_sepp.py` main loop     | `phot7ds.batch_run(jobs=[...])` (a list of plain dicts, no CSV required) |
| `DELVE_DetImage.py` main loop       | `phot7ds.detection.build_delve_detection_image(...)`                  |
| `from Utils_7DT import *`           | `from phot7ds import <thing>` (see `phot7ds/__init__.py`)             |
| Hardcoded `path_base`, `server`     | Plain kwargs / optional `PhotometryConfig`                            |
| RSS / `malloc_trim` / `enforce_rss_guard` plumbing | Removed                                                  |
| `Utils_7DT.{sex_config, hotpants, sepp_config, calculate_integrated_magnitude, extcor, synth_phot, get_fast_columns, ...}` | Removed; they weren't used by the calibrated-catalog pipeline |
