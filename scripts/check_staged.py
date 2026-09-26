"""Guard staged files against text that must never be committed.

Checks the staged content (not the working tree) of every added or modified file:

- no Cyrillic: all committed text is English;
- no personal identity: the global git ``user.name`` / ``user.email`` must not appear
  in files and must not be the identity of the commit. The values are read at run time
  and are never written down anywhere in the repository.

Usage: ``python scripts/check_staged.py`` (called from the pre-commit hook).
"""

from __future__ import annotations

import re
import subprocess
import sys

_CYRILLIC = re.compile(r"[\u0400-\u04FF]")


def _git(*args: str) -> str:
    proc = subprocess.run(["git", *args], capture_output=True, text=True, check=False)  # noqa: S607
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _staged_files() -> list[str]:
    out = _git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z")
    return [name for name in out.split("\0") if name]


def _staged_text(path: str) -> str | None:
    proc = subprocess.run(["git", "show", f":{path}"], capture_output=True, check=False)  # noqa: S607
    if proc.returncode != 0 or b"\0" in proc.stdout:
        return None  # binary or unreadable
    return proc.stdout.decode("utf-8", errors="replace")


def check() -> list[str]:
    problems: list[str] = []
    forbidden = [
        v
        for v in (_git("config", "--global", "user.name"), _git("config", "--global", "user.email"))
        if v
    ]

    commit_name = _git("config", "user.name")
    commit_email = _git("config", "user.email")
    if commit_name in forbidden or commit_email in forbidden:
        problems.append(
            "commit identity is the global git identity; set a repository-local "
            "user.name and user.email"
        )

    for path in _staged_files():
        text = _staged_text(path)
        if text is None:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if _CYRILLIC.search(line):
                problems.append(f"{path}:{lineno}: Cyrillic text (committed text must be English)")
            lowered = line.lower()
            if any(value.lower() in lowered for value in forbidden):
                problems.append(f"{path}:{lineno}: personal identity from the global git config")
    return problems


def main() -> int:
    problems = check()
    for problem in problems:
        print(f"✖ {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
