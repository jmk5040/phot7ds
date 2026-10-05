"""
7DS filter registry: the single place that lists the filters phot7ds knows.

The registry is a packaged ECSV table, ``phot7ds/data/filters_7ds.ecsv``,
one row per filter, in canonical order (catalog band order):

``name``
    Filter name as written in the ``FILTER`` header card (``g``, ``m425w``).
``kind``
    ``broad``, ``medium`` or ``wide``.
``lambda_pivot_nm``, ``fwhm_nm``
    Pivot wavelength and FWHM of the transmission curve [nm].
``key``
    1-3 character token used in FITS header keywords (``ZP05M{key}``,
    ``UL5EM{key}``, ``BRMSM{key}``). Stored per filter, so an irregular name
    only needs an explicit entry here.

When the filter set changes, edit (or regenerate) that one file. To
regenerate it from a directory of transmission curves (``<name>.csv`` with
``lam`` [nm] and ``trans`` columns)::

    python -m phot7ds.filters /path/to/Filter_transmission phot7ds/data/filters_7ds.ecsv

New rows get :func:`default_header_key` (``m425`` -> ``425``,
``m425w`` -> ``42W``, ``g`` -> ``G``). A different registry file can be used
via ``PhotometryConfig.filter_registry`` or ``$PHOT7DS_FILTER_REGISTRY``.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator, Literal, Mapping, Sequence

import numpy as np

REGISTRY_ENV = "PHOT7DS_FILTER_REGISTRY"
PACKAGED_REGISTRY = Path(__file__).parent / "data" / "filters_7ds.ecsv"
FILTER_KINDS = ("broad", "medium", "wide")

_KEY_RE = re.compile(r"^[A-Z0-9]{1,3}$")
_NAME_RE = re.compile(r"^[a-z][a-z0-9]*$")
_DUP_SUFFIX_RE = re.compile(r"^(.+)-(\d+)$")


@dataclass(frozen=True)
class FilterInfo:
    """One registry row."""

    name: str
    kind: str
    lambda_pivot_nm: float
    fwhm_nm: float
    key: str


class FilterRegistry:
    """Ordered, validated collection of :class:`FilterInfo` rows."""

    def __init__(self, filters: Iterable[FilterInfo], source: str = "<memory>"):
        self.filters: tuple[FilterInfo, ...] = tuple(filters)
        self.source = source
        self._by_name = {f.name: f for f in self.filters}
        self._validate()

    def _validate(self) -> None:
        problems: list[str] = []
        if not self.filters:
            problems.append("registry is empty")
        seen_names: dict[str, str] = {}
        seen_keys: dict[str, str] = {}
        for f in self.filters:
            if not _NAME_RE.match(f.name):
                problems.append(f"bad filter name {f.name!r} (lowercase letters/digits)")
            if f.name.lower() in seen_names:
                problems.append(f"duplicate filter name {f.name!r}")
            seen_names[f.name.lower()] = f.name
            if f.kind not in FILTER_KINDS:
                problems.append(f"{f.name}: kind {f.kind!r} not in {FILTER_KINDS}")
            if not _KEY_RE.match(f.key):
                problems.append(
                    f"{f.name}: header key {f.key!r} must be 1-3 of A-Z/0-9 "
                    "(so ZP05M<key> stays within 8 characters)"
                )
            if f.key in seen_keys:
                problems.append(
                    f"header key {f.key!r} shared by {seen_keys[f.key]} and {f.name}"
                )
            seen_keys[f.key] = f.name
        if problems:
            raise ValueError(f"Invalid filter registry {self.source}: " + "; ".join(problems))

    # -- container protocol ---------------------------------------------
    def __len__(self) -> int:
        return len(self.filters)

    def __iter__(self) -> Iterator[FilterInfo]:
        return iter(self.filters)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._by_name

    def __getitem__(self, name: str) -> FilterInfo:
        return self._by_name[name]

    def __repr__(self) -> str:
        return f"FilterRegistry({len(self)} filters from {self.source})"

    # -- lookups ----------------------------------------------------------
    @property
    def names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.filters)

    def names_of_kind(self, *kinds: str) -> tuple[str, ...]:
        return tuple(f.name for f in self.filters if f.kind in kinds)

    def canonical(self, label: str | None) -> str | None:
        """Registry name matching ``label`` (case-insensitive), else ``None``."""
        if label is None:
            return None
        label = str(label).strip()
        if label in self._by_name:
            return label
        lowered = label.lower()
        for name in self._by_name:
            if name.lower() == lowered:
                return name
        return None

    def header_key(self, band: str) -> str:
        """Header-keyword token for ``band``.

        A disambiguated duplicate (``g-1``, see
        :func:`phot7ds.images.extract_band_names_and_saturation`) gets the
        base key plus ``-N``; the resulting keyword is longer than 8
        characters and is written as a HIERARCH card.
        """
        if band in self._by_name:
            return self._by_name[band].key
        m = _DUP_SUFFIX_RE.match(band)
        if m and m.group(1) in self._by_name:
            return f"{self._by_name[m.group(1)].key}-{m.group(2)}"
        raise KeyError(f"Band {band!r} is not in the filter registry ({self.source})")

    def header_keys(self, bands: Sequence[str]) -> dict[str, str]:
        """``{band: key}`` for a run, refusing any two bands sharing a key.

        A shared key would make the second band's header cards silently
        overwrite the first band's.
        """
        keys: dict[str, str] = {}
        owner: dict[str, str] = {}
        for band in bands:
            key = self.header_key(band)
            if key in owner and owner[key] != band:
                raise ValueError(
                    f"Bands {owner[key]!r} and {band!r} map to the same header key "
                    f"{key!r}; fix the 'key' column of {self.source}"
                )
            owner[key] = band
            keys[band] = key
        return keys


def default_header_key(name: str) -> str:
    """Default header key for a new registry row.

    ``g`` -> ``G``; ``m425`` -> ``425``; ``m425w`` -> ``42W`` (the first two
    digits plus ``W``, keeping 3 characters and never colliding with a
    3-digit medium-band key). Irregular names fall back to the first three
    upper-cased characters; check the result for collisions.
    """
    m = re.fullmatch(r"m(\d+)(w?)", name.lower())
    if m:
        digits, wide = m.groups()
        return f"{digits[:2]}W" if wide else digits[:3]
    return re.sub(r"[^A-Z0-9]", "", name.upper())[:3]


def default_kind(name: str) -> str:
    """``broad`` for single letters, ``wide`` for ``m###w``, else ``medium``."""
    if len(name) == 1:
        return "broad"
    return "wide" if re.fullmatch(r"m\d+w", name.lower()) else "medium"


def _read_registry_table(path: Path) -> FilterRegistry:
    from astropy.table import Table

    tbl = Table.read(str(path), format="ascii.ecsv")
    missing = {"name", "kind", "lambda_pivot_nm", "fwhm_nm", "key"} - set(tbl.colnames)
    if missing:
        raise ValueError(f"Filter registry {path} lacks columns {sorted(missing)}")
    return FilterRegistry(
        (
            FilterInfo(
                name=str(row["name"]).strip(),
                kind=str(row["kind"]).strip(),
                lambda_pivot_nm=float(row["lambda_pivot_nm"]),
                fwhm_nm=float(row["fwhm_nm"]),
                key=str(row["key"]).strip(),
            )
            for row in tbl
        ),
        source=str(path),
    )


@lru_cache(maxsize=8)
def _load_cached(path: str) -> FilterRegistry:
    return _read_registry_table(Path(path))


def resolve_registry_path(path: str | os.PathLike | None = None) -> Path:
    """``path`` if given, else ``$PHOT7DS_FILTER_REGISTRY``, else the packaged file."""
    chosen = path or os.environ.get(REGISTRY_ENV) or PACKAGED_REGISTRY
    return Path(chosen).expanduser().resolve()


def load_filter_registry(
    path: str | os.PathLike | FilterRegistry | None = None,
) -> FilterRegistry:
    """Load (and cache) the filter registry. See the module docstring."""
    if isinstance(path, FilterRegistry):
        return path
    resolved = resolve_registry_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Filter registry not found: {resolved}")
    return _load_cached(str(resolved))


def registry_from_transmission_curves(
    curve_dir: str | os.PathLike,
    *,
    keys: Mapping[str, str] | None = None,
    kinds: Mapping[str, str] | None = None,
):
    """Build a registry table from ``<name>.csv`` transmission curves.

    Each CSV has ``lam`` [nm] and ``trans`` columns. Rows are ordered broad
    bands first (by wavelength), then medium and wide bands by their name's
    number (``m425`` before ``m425w``). ``keys`` / ``kinds`` override the
    defaults per filter.
    """
    from astropy.table import Table

    keys = dict(keys or {})
    kinds = dict(kinds or {})
    trapz = getattr(np, "trapezoid", None) or np.trapz
    rows = []
    for csv in sorted(Path(curve_dir).glob("*.csv")):
        name = csv.stem
        data = np.genfromtxt(csv, delimiter=",", names=True)
        lam = np.asarray(data["lam"], dtype=float)
        trans = np.asarray(data["trans"], dtype=float)
        order = np.argsort(lam, kind="stable")
        lam, trans = lam[order], trans[order]
        pivot = float(np.sqrt(trapz(trans * lam, lam) / trapz(trans / lam, lam)))
        half = trans >= 0.5 * trans.max()
        fwhm = float(lam[half].max() - lam[half].min())
        kind = kinds.get(name, default_kind(name))
        rows.append((name, kind, round(pivot, 2), round(fwhm, 2), keys.get(name, default_header_key(name))))

    def _sort_key(row):
        name, kind, pivot = row[0], row[1], row[2]
        if kind == "broad":
            return (0, pivot, name)
        num = re.match(r"m(\d+)", name)
        return (1, int(num.group(1)) if num else pivot, name)

    rows.sort(key=_sort_key)
    tbl = Table(rows=rows, names=("name", "kind", "lambda_pivot_nm", "fwhm_nm", "key"))
    tbl["lambda_pivot_nm"].unit = "nm"
    tbl["fwhm_nm"].unit = "nm"
    tbl.meta["description"] = (
        "phot7ds filter registry: one row per 7DS filter, in catalog band order. "
        "'key' is the FITS header-keyword token (ZP05M<key>, UL5EM<key>, BRMSM<key>), "
        "1-3 of A-Z/0-9 and unique."
    )
    tbl.meta["source"] = str(Path(curve_dir))
    FilterRegistry(
        (FilterInfo(r[0], r[1], r[2], r[3], r[4]) for r in rows), source=str(curve_dir)
    )
    return tbl


def get_filter_definitions(
    unit: Literal["angstrom", "nm"] = "angstrom",
    registry: str | os.PathLike | FilterRegistry | None = None,
):
    """Return 7DS filter definitions from the filter registry.

    Parameters
    ----------
    unit
        Wavelength unit: ``'angstrom'`` (default) or ``'nm'``.
    registry
        Registry to read (default: :func:`load_filter_registry`).

    Returns
    -------
    bands_dict : dict[str, float]
        Filter name -> pivot wavelength, in registry order.
    bands_width : dict[str, float]
        Filter name -> half of the FWHM.
    bands_color : dict[str, str | tuple]
        Filter name -> matplotlib colour (broad bands: named colours;
        medium/wide bands: coolwarm value by wavelength).
    lambda_to_color : dict[float, str | tuple]
        Wavelength -> colour, useful for line plots keyed by lambda.
    lambda_to_band : dict[float, str]
        Wavelength -> filter name.
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm

    if unit not in ("angstrom", "nm"):
        raise ValueError(f"unit must be 'angstrom' or 'nm', got {unit!r}")
    scale = 10.0 if unit == "angstrom" else 1.0
    reg = load_filter_registry(registry)

    bands_dict = {f.name: f.lambda_pivot_nm * scale for f in reg}
    bands_width = {f.name: f.fwhm_nm * scale / 2 for f in reg}

    broad_colors = {
        "u": "violet", "g": "lightgreen", "r": "lightcoral", "i": "coral", "z": "brown",
    }
    narrow = [f for f in reg if f.kind != "broad"]
    lams = [f.lambda_pivot_nm for f in narrow] or [0.0, 1.0]
    norm = plt.Normalize(min(lams), max(lams))
    bands_color = {
        f.name: (broad_colors.get(f.name, "gray") if f.kind == "broad"
                 else cm.coolwarm(norm(f.lambda_pivot_nm)))
        for f in reg
    }
    lambda_to_color = {bands_dict[b]: bands_color[b] for b in bands_dict}
    lambda_to_band = {v: k for k, v in bands_dict.items()}
    return bands_dict, bands_width, bands_color, lambda_to_color, lambda_to_band


#: Canonical band order of the unified catalog schema (all registry filters).
#: Evaluated at import from ``$PHOT7DS_FILTER_REGISTRY`` or the packaged file.
DEFAULT_BANDS: list[str] = list(load_filter_registry().names)


def _main(argv: Sequence[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Build a phot7ds filter registry (ECSV) from transmission-curve CSVs.",
    )
    ap.add_argument("curve_dir", help="Directory of <name>.csv files (lam [nm], trans)")
    ap.add_argument("output", help="Output ECSV path")
    args = ap.parse_args(argv)
    tbl = registry_from_transmission_curves(args.curve_dir)
    tbl.write(args.output, format="ascii.ecsv", overwrite=True)
    print(f"Wrote {len(tbl)} filters to {args.output}")


if __name__ == "__main__":
    _main()


__all__ = [
    "FILTER_KINDS",
    "FilterInfo",
    "FilterRegistry",
    "PACKAGED_REGISTRY",
    "REGISTRY_ENV",
    "DEFAULT_BANDS",
    "default_header_key",
    "default_kind",
    "get_filter_definitions",
    "load_filter_registry",
    "registry_from_transmission_curves",
    "resolve_registry_path",
]
