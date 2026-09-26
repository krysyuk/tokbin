from __future__ import annotations

from pathlib import Path

import pytest

from tokbin import codes
from tokbin.errors import FormatError, SchemaVersionError, UnsupportedFeatureError
from tokbin.format import schema


def test_current_schema_is_accepted_silently() -> None:
    schema.check_schema(schema.SCHEMA_VERSION, "meta.json")


def test_newer_schema_asks_to_upgrade() -> None:
    with pytest.raises(SchemaVersionError) as info:
        schema.check_schema(schema.SCHEMA_VERSION + 1, "meta.json")
    assert info.value.code is codes.SCHEMA_TOO_NEW
    assert "pip install -U tokbin" in str(info.value)


def test_unsupported_old_schema() -> None:
    with pytest.raises(FormatError) as info:
        schema.check_schema(0, "meta.json")
    assert info.value.code is codes.SCHEMA_TOO_OLD


def test_older_supported_schema_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(schema, "SCHEMA_VERSION", 2)
    monkeypatch.setattr(schema, "SUPPORTED_SCHEMAS", frozenset({1, 2}))
    with pytest.warns(FutureWarning, match="tokbin migrate"):
        schema.check_schema(1, "meta.json")


def test_source_kind(tmp_path: Path) -> None:
    (tmp_path / "meta.json").write_text("{}")
    assert schema.source_kind(tmp_path) == "single"


def test_source_kind_multimodal_is_recognized_but_unsupported(tmp_path: Path) -> None:
    (tmp_path / "dataset.json").write_text("{}")
    with pytest.raises(UnsupportedFeatureError) as info:
        schema.source_kind(tmp_path)
    assert info.value.code is codes.FEATURE_UNSUPPORTED


def test_source_kind_not_a_source(tmp_path: Path) -> None:
    with pytest.raises(FormatError) as info:
        schema.source_kind(tmp_path)
    assert info.value.code is codes.METADATA_MISSING
