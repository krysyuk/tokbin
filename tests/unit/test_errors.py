from __future__ import annotations

import pickle
import re

import pytest

import tokbin
from tokbin import codes, errors
from tokbin.codes import Code

_CATEGORY_OF = {
    errors.ConfigError: "C",
    errors.ContractError: "K",
    errors.DataError: "D",
    errors.FormatError: "F",
    errors.SchemaVersionError: "F",
    errors.IntegrityError: "I",
    errors.CompatibilityError: "M",
    errors.ResumeError: "R",
    errors.DependencyError: "P",
    errors.UnsupportedFeatureError: "F",
    errors.InternalError: "X",
}


def test_hierarchy_matches_spec() -> None:
    for cls in _CATEGORY_OF:
        assert issubclass(cls, errors.TokbinError)
    assert issubclass(errors.SchemaVersionError, errors.FormatError)
    assert issubclass(errors.DependencyError, ImportError)
    for w in (errors.DataQualityWarning, errors.ShardingWarning, errors.CompatibilityWarning):
        assert issubclass(w, errors.TokbinWarning)
    assert issubclass(errors.TokbinWarning, UserWarning)


def test_every_error_class_is_exported() -> None:
    for cls in _CATEGORY_OF:
        assert getattr(tokbin, cls.__name__) is cls


@pytest.mark.parametrize(("cls", "category"), _CATEGORY_OF.items())
def test_class_accepts_only_its_category(cls: type[errors.TokbinError], category: str) -> None:
    own = Code(f"TB-{category}999", "test")
    err = cls(own, why="w", fix="f")
    assert err.code is own

    foreign_category = "C" if category != "C" else "K"
    with pytest.raises(TypeError):
        cls(Code(f"TB-{foreign_category}999", "test"), why="w", fix="f")


def test_message_has_what_why_fix() -> None:
    err = errors.IntegrityError(
        Code("TB-I999", "Shard is missing"),
        "corpus/web/train-00003.bin",
        why="meta.json lists 13 shards, 12 found on disk",
        fix="download the shard and run `tokbin verify corpus/web`",
        where="tokbin.read.source.read_source",
    )
    assert str(err) == (
        "[TB-I999] Shard is missing: corpus/web/train-00003.bin\n"
        "  where: tokbin.read.source.read_source\n"
        "  cause: meta.json lists 13 shards, 12 found on disk\n"
        "  fix:   download the shard and run `tokbin verify corpus/web`"
    )


def test_message_without_detail_and_where() -> None:
    err = errors.ConfigError(Code("TB-C999", "Invalid parameter"), why="w", fix="f")
    assert str(err).splitlines() == ["[TB-C999] Invalid parameter", "  cause: w", "  fix:   f"]


def test_where_set_later_is_rendered() -> None:
    err = errors.ConfigError(Code("TB-C999", "x"), why="w", fix="f")
    err.where = "tokbin.api"
    assert "where: tokbin.api" in str(err)


@pytest.mark.parametrize("cls", list(_CATEGORY_OF))
def test_errors_survive_pickle(cls: type[errors.TokbinError]) -> None:
    code = Code(f"TB-{_CATEGORY_OF[cls]}999", "test")
    err = cls(code, "detail", why="w", fix="f", where="here")
    restored = pickle.loads(pickle.dumps(err))  # noqa: S301 (our own object)
    assert type(restored) is cls
    assert str(restored) == str(err)
    assert restored.code == code


def test_internal_error_wraps_foreign_exception() -> None:
    err = errors.InternalError.wrap(ValueError("boom"), where="tokbin.x")
    assert err.code is codes.INTERNAL
    assert "ValueError: boom" in str(err)
    assert "issue" in str(err)


def test_dependency_error_is_caught_as_import_error() -> None:
    with pytest.raises(ImportError):
        raise errors.DependencyError(codes.DEPENDENCY_MISSING, "pkg", why="w", fix="f")


def test_code_format() -> None:
    for code in codes.CODES.values():
        assert re.fullmatch(r"TB-[CKDFIMRPSX]\d{3}", code.id)
        assert code.title
        assert code.id not in codes.RETIRED
