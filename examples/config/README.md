# `examples/config/`

The example script `examples/example_run.py` expects to find these files
**inside this directory** (paths resolve relative to the repository, so
the example works regardless of where you cloned the repo).

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

## Auto-generated configs

`phot7ds.ensure_sepp_config` and `phot7ds.ensure_swarp_config` create
the SE++ and SWarp configs on the first run by shelling out to the
respective binaries (`sourcextractor++ --dump-default-config` and
`SWarp -dd`). If the files already exist they are left untouched.

If you don't have `sourcextractor++` / `SWarp` on `PATH`, drop the
files into this folder manually.

## Required artefacts

* `7DT_tiles.ascii` — the survey tile manifest. Must contain at least:
  `tile`, `ra`, `dec`, `ra1`–`ra4`, `dec1`–`dec4`. Missing file raises
  `FileNotFoundError` from `phot7ds.require_tile_table`.

* `gaiaxp/gaiaxp_dr3_synphot_<TILE>.csv` — per-tile Gaia XP synphot
  catalog. Must contain `ra`, `dec` and `mag_<band>` columns for every
  band you'll process. Missing file raises `FileNotFoundError` from
  `phot7ds.require_gaiaxp_reference`.

## Catalog naming conventions

See `column_convention.md` in this directory for the full output
column / FITS header cheat-sheet.
