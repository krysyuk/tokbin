"""The environment report behind ``tokbin doctor`` (spec 2.5).

Nothing is imported to build it: optional packages are only looked up, so the report
works (and stays fast) whatever is installed.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from importlib import metadata
from typing import Literal

from tokbin import _deps
from tokbin._boundary import public_api
from tokbin._version import get_version

__all__ = ["DoctorReport", "PackageReport", "doctor"]

PackageState = Literal["ok", "missing", "optional"]


@dataclass(frozen=True, slots=True, kw_only=True)
class PackageReport:
    """One package and what depends on it.

    ``state``: ``ok`` if available, ``missing`` if a mode-defining package is absent,
    ``optional`` if an optional extra is absent.
    """

    name: str
    state: PackageState
    #: Installed version, or a short note such as ``stdlib (Python 3.14)``.
    version: str | None
    #: What is unavailable without it; ``None`` when available.
    unavailable: str | None
    #: How to install it; ``None`` when available.
    install: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "state": self.state,
            "version": self.version,
            "unavailable": self.unavailable,
            "install": self.install,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class DoctorReport:
    version: str
    mode: _deps.Mode
    python: str
    platform: str
    packages: tuple[PackageReport, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "mode": self.mode,
            "python": self.python,
            "platform": self.platform,
            "packages": [p.to_dict() for p in self.packages],
        }


def _numpy() -> PackageReport:
    try:
        version: str | None = metadata.version("numpy")
    except metadata.PackageNotFoundError:
        version = None
    return PackageReport(name="numpy", state="ok", version=version, unavailable=None, install=None)


def _package(name: str, *, label: str, required: bool) -> PackageReport:
    status = _deps.probe(name)
    dep = status.dependency
    if status.installed:
        return PackageReport(
            name=label, state="ok", version=status.version, unavailable=None, install=None
        )
    return PackageReport(
        name=label,
        state="missing" if required else "optional",
        version=None,
        unavailable=dep.feature,
        install=dep.install,
    )


def _zstd() -> PackageReport:
    if _deps.has_stdlib_zstd():
        version = f"stdlib (Python {sys.version_info.major}.{sys.version_info.minor})"
        return PackageReport(
            name="zstd", state="ok", version=version, unavailable=None, install=None
        )
    report = _package("zstandard", label="zstd", required=False)
    if report.state == "ok":
        return report
    return PackageReport(
        name="zstd",
        state="optional",
        version=None,
        unavailable="fast compression in pack (lzma is used instead)",
        install=report.install,
    )


@public_api
def doctor() -> DoctorReport:
    """Version, operating mode and optional packages."""
    return DoctorReport(
        version=get_version(),
        mode=_deps.detect_mode(),
        python=platform.python_version(),
        platform=f"{platform.system()} {platform.machine()}".strip(),
        packages=(
            _numpy(),
            _package("tokenizers", label="tokenizers", required=True),
            _zstd(),
            _package("torch", label="torch", required=False),
            _package("huggingface_hub", label="huggingface-hub", required=False),
        ),
    )
