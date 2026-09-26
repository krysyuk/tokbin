"""Optional dependencies: lazy import and detection of the operating mode.

Optional packages are imported only through :func:`require`, inside functions.
Availability checks (:func:`probe`) do not import the package: ``tokbin doctor``
must not pull in torch just to print a line in its report.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from dataclasses import dataclass
from importlib import metadata
from types import ModuleType
from typing import Literal

from tokbin import codes
from tokbin.errors import DependencyError

__all__ = [
    "KNOWN",
    "Dependency",
    "DependencyStatus",
    "Mode",
    "detect_mode",
    "probe",
    "require",
]

Mode = Literal["full", "core"]


@dataclass(frozen=True, slots=True, kw_only=True)
class Dependency:
    """An optional package that some features depend on."""

    #: Module name to import.
    module: str
    #: Distribution name for ``importlib.metadata``.
    distribution: str
    #: What becomes available with this package.
    feature: str
    #: Install command.
    install: str


KNOWN: dict[str, Dependency] = {
    "tokenizers": Dependency(
        module="tokenizers",
        distribution="tokenizers",
        feature="writing datasets",
        install="pip install tokbin",
    ),
    "zstandard": Dependency(
        module="zstandard",
        distribution="zstandard",
        feature="zstd compression on Python < 3.14",
        install="pip install 'tokbin-core[zstd]'",
    ),
    "torch": Dependency(
        module="torch",
        distribution="torch",
        feature="torch DataLoader adapter",
        install="pip install 'tokbin-core[torch]'",
    ),
    "huggingface_hub": Dependency(
        module="huggingface_hub",
        distribution="huggingface-hub",
        feature="loading a tokenizer by name",
        install="pip install 'tokbin-core[hub]'",
    ),
}


@dataclass(frozen=True, slots=True, kw_only=True)
class DependencyStatus:
    """Result of checking a single package."""

    dependency: Dependency
    installed: bool
    version: str | None


def probe(name: str) -> DependencyStatus:
    """Check whether a package is available without importing it."""
    dep = KNOWN[name]
    installed = importlib.util.find_spec(dep.module) is not None
    version: str | None = None
    if installed:
        try:
            version = metadata.version(dep.distribution)
        except metadata.PackageNotFoundError:
            version = None
    return DependencyStatus(dependency=dep, installed=installed, version=version)


def require(name: str) -> ModuleType:
    """Import an optional package or raise :class:`DependencyError`."""
    dep = KNOWN[name]
    try:
        return importlib.import_module(dep.module)
    except ImportError as exc:
        err = DependencyError(
            codes.DEPENDENCY_MISSING,
            dep.distribution,
            why=f"package `{dep.distribution}` is required for {dep.feature}, "
            f"but it could not be imported ({type(exc).__name__}: {exc})",
            fix=f"install it: {dep.install}",
        )
        err.name = dep.module
        raise err from exc


def detect_mode() -> Mode:
    """``full`` if writing is available (``tokenizers`` is installed), else ``core``."""
    return "full" if probe("tokenizers").installed else "core"


def has_stdlib_zstd() -> bool:
    """Python 3.14+ ships zstd in the standard library (``compression.zstd``)."""
    return sys.version_info >= (3, 14)
