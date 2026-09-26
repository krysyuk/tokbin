"""Create the frozen dataset fixtures in tests/fixtures/.

Fixtures are generated once and committed; they must never be regenerated for an
existing schema version, because their job is to prove that old datasets stay
readable. Add a new directory for a new schema version instead.

Usage: ``uv run python tests/fixtures/make_fixtures.py schema_v1``
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from tokbin import Dataset, ShardingWarning, WriterConfig  # noqa: E402

from support import make_docs, make_tokenizer  # noqa: E402


def schema_v1(root: Path) -> None:
    ds = Dataset(root)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ShardingWarning)
        ds.write(
            "tiny", make_docs(40, seed=1), make_tokenizer(), config=WriterConfig(shard_bytes=256)
        )
        ds.write(
            "tiny",
            make_docs(5, seed=2),
            make_tokenizer(),
            config=WriterConfig(split="valid", shard_bytes=256),
        )


def main() -> int:
    name = sys.argv[1]
    root = HERE / name
    if root.exists():
        print(f"{root} exists; fixtures are frozen and never regenerated", file=sys.stderr)
        return 1
    {"schema_v1": schema_v1}[name](root)
    print(f"created {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
