"""
Software provenance of a run: phot7ds version, git state, SE++ and Python.

Recorded in the run log and the manifest so that a run can be traced even
when no catalog was written (``PHOTVER`` lives only in the catalog header).
"""
from __future__ import annotations

import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

_PACKAGE_DIR = Path(__file__).resolve().parent


def _run(cmd: list[str], cwd: Path | None = None, timeout: float = 10.0) -> str | None:
    try:
        out = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def git_state() -> dict[str, Any] | None:
    """Git commit of the phot7ds checkout in use, or ``None`` if not a git checkout.

    Only reported when the repository top level is the directory that holds
    this package, so an installed copy inside some unrelated repository is not
    attributed that repository's commit.
    """
    if shutil.which("git") is None:
        return None
    top = _run(["git", "rev-parse", "--show-toplevel"], cwd=_PACKAGE_DIR)
    if top is None or Path(top).resolve() != _PACKAGE_DIR.parent:
        return None
    commit = _run(["git", "rev-parse", "HEAD"], cwd=_PACKAGE_DIR)
    if commit is None:
        return None
    status = _run(["git", "status", "--porcelain", "--untracked-files=no", "--", "phot7ds"],
                  cwd=_PACKAGE_DIR.parent)
    return {
        "commit": commit,
        "describe": _run(["git", "describe", "--always", "--tags"], cwd=_PACKAGE_DIR),
        "dirty": bool(status),
    }


def sepp_version(executable: str = "sourcextractor++") -> str | None:
    """``sourcextractor++ --version`` (e.g. ``'SourceXtractorPlusPlus 1.0.3'``)."""
    if shutil.which(executable) is None:
        return None
    out = _run([executable, "--version"], timeout=60.0)
    return out.splitlines()[0] if out else None


def collect_provenance() -> dict[str, Any]:
    try:
        from . import __version__ as version
    except Exception:
        version = "unknown"
    return {
        "phot7ds_version": str(version),
        "git": git_state(),
        "sourcextractor_version": sepp_version(),
        "python_version": platform.python_version(),
    }


def format_provenance(prov: dict[str, Any]) -> str:
    """One-line summary for the run log."""
    git = prov.get("git")
    if git:
        git_txt = f"git {git['commit'][:10]}{' (dirty)' if git['dirty'] else ''}"
    else:
        git_txt = "git n/a"
    return (
        f"phot7ds {prov.get('phot7ds_version')} | {git_txt} | "
        f"{prov.get('sourcextractor_version') or 'SE++ version n/a'} | "
        f"Python {prov.get('python_version')}"
    )


__all__ = ["collect_provenance", "format_provenance", "git_state", "sepp_version"]
