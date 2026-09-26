"""Check that the versions of tokbin-core and the tokbin meta-package match.

The meta-package must depend on exactly ``tokbin-core==<same version>``. With
``--tag vX.Y.Z`` the release tag is checked as well.

Usage: ``python scripts/check_versions.py [--tag v0.1.0]``. Requires Python 3.11+ (tomllib).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parent.parent


def _load(path: Path) -> dict[str, object]:
    with path.open("rb") as f:
        project: dict[str, object] = tomllib.load(f)["project"]
    return project


def check(tag: str | None) -> list[str]:
    core = _load(ROOT / "pyproject.toml")
    meta = _load(ROOT / "packages" / "tokbin" / "pyproject.toml")
    problems: list[str] = []

    version = core["version"]
    if meta["version"] != version:
        problems.append(f"versions differ: tokbin-core {version}, tokbin {meta['version']}")

    pin = f"tokbin-core=={version}"
    deps = meta.get("dependencies", [])
    if not isinstance(deps, list) or pin not in deps:
        problems.append(f"the meta-package must depend on `{pin}`, found: {deps}")

    if tag is not None and tag.removeprefix("v") != version:
        problems.append(f"tag {tag} does not match version {version}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="release tag, e.g. v0.1.0")
    args = parser.parse_args()
    problems = check(args.tag)
    for problem in problems:
        print(f"✖ {problem}", file=sys.stderr)
    if not problems:
        print("✔ package versions are consistent")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
