"""Invariant 9: ``import tokbin`` pulls in nothing but stdlib and numpy (spec 2.4)."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from typing import Any

import pytest

_PROBE = textwrap.dedent(
    """
    import json, logging, sys
    before = set(sys.modules)
    {imports}
    added = sorted(set(sys.modules) - before)
    print(json.dumps({{
        "added": added,
        "stdlib": sorted(sys.stdlib_module_names),
        "tokbin_handlers": len(logging.getLogger("tokbin").handlers),
        "root_handlers": len(logging.getLogger().handlers),
    }}))
    """
)


def _run(imports: str) -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE.format(imports=imports)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stderr == "", f"import printed something to stderr:\n{proc.stderr}"
    result: dict[str, Any] = json.loads(proc.stdout)
    return result


def _third_party(result: dict[str, Any]) -> set[str]:
    stdlib = set(result["stdlib"])
    tops = {name.split(".")[0] for name in result["added"]}
    return tops - stdlib


@pytest.mark.parametrize(
    "imports",
    [
        "import tokbin",
        "import tokbin.errors, tokbin.codes, tokbin._deps, tokbin._fs, tokbin._boundary",
        "import tokbin.cli.main",
        "import tokbin.write.stream_writer, tokbin.tokenizer.resolve, tokbin.dataset",
        "from tokbin import Dataset, StreamWriter, WriterConfig",
    ],
)
def test_import_pulls_only_numpy(imports: str) -> None:
    result = _run(imports)
    assert _third_party(result) <= {"tokbin", "numpy"}


def test_import_configures_no_logging() -> None:
    result = _run("import tokbin")
    assert result["tokbin_handlers"] == 0
    assert result["root_handlers"] == 0


def test_import_prints_nothing() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", "import tokbin"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout == ""
    assert proc.stderr == ""
