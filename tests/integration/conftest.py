from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from support import make_tokenizer


@pytest.fixture
def tokenizer() -> Any:
    return make_tokenizer()


@pytest.fixture
def tokenizer_path(tmp_path: Path, tokenizer: Any) -> Path:
    path = tmp_path / "tokenizers" / "tiny" / "tokenizer.json"
    path.parent.mkdir(parents=True)
    tokenizer.save(str(path))
    return path
