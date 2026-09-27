"""``docs/errors.md`` is generated from ``tokbin.codes`` and must not drift from it."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def test_errors_doc_is_up_to_date() -> None:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "gen_errors_doc.py"), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
