from __future__ import annotations

import importlib
import sys

import pytest

from tokbin import _deps
from tokbin.errors import DependencyError


@pytest.mark.requires_tokenizers
def test_require_imports_available_module() -> None:
    assert _deps.require("tokenizers") is importlib.import_module("tokenizers")


def test_require_missing_module_explains_install(monkeypatch: pytest.MonkeyPatch) -> None:
    # None in sys.modules makes the import fail with ImportError.
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(DependencyError) as info:
        _deps.require("torch")
    err = info.value
    assert isinstance(err, ImportError)
    assert err.name == "torch"
    assert "pip install 'tokbin-core[torch]'" in str(err)
    assert isinstance(err.__cause__, ImportError)


def test_probe_does_not_import(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "zstandard", raising=False)
    _deps.probe("zstandard")
    assert "zstandard" not in sys.modules


@pytest.mark.requires_tokenizers
def test_probe_reports_version_of_installed() -> None:
    status = _deps.probe("tokenizers")
    assert status.installed
    assert status.version


def test_detect_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _deps.detect_mode() == ("full" if _deps.probe("tokenizers").installed else "core")

    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a: None if name == "tokenizers" else real(name, *a),
    )
    assert _deps.detect_mode() == "core"


def test_every_known_dependency_has_install_hint() -> None:
    for dep in _deps.KNOWN.values():
        assert dep.install.startswith("pip install ")
