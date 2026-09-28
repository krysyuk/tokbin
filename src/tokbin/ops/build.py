"""Building a source from standard inputs without code (spec 12.3).

::

    tokbin build code --from-txt data/code --tokenizer gpt2/tokenizer.json
    tokbin build web  --from-jsonl data/web --field text --tokenizer gpt2/tokenizer.json
    tokbin build web  --resume

The settings of a build (input, tokenizer, writer configuration) are saved as
``build.json`` in the partial directory when the write starts. ``--resume`` without
inputs takes them from there, so an interrupted build continues with exactly the same
settings. The file is removed when the source is published.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from tokbin import codes
from tokbin._boundary import public_api
from tokbin.errors import ConfigError
from tokbin.format import naming
from tokbin.format._fields import Fields
from tokbin.format._json import read_json_bounded, write_json_atomic
from tokbin.inputs import InputDoc, JsonlInput, TxtDirInput
from tokbin.write.config import ErrorPolicy, WriterConfig
from tokbin.write.partial import partial_path
from tokbin.write.result import WriteResult
from tokbin.write.stream_writer import StreamWriter

__all__ = ["BuildRecipe", "InputKind", "build_source", "read_recipe"]

InputKind = Literal["txt", "jsonl"]
RECIPE_FORMAT: Final = 1

#: Called with input bytes read so far, the total, and the file being read.
ProgressCallback = Callable[[int, int, str], None]

_CONFIG_FIELDS: Final = tuple(f.name for f in dataclasses.fields(WriterConfig))


@dataclass(frozen=True, slots=True, kw_only=True)
class BuildRecipe:
    """Everything a build needs besides the target."""

    input_kind: InputKind
    #: Directory (or single file) of the input.
    input_path: Path
    tokenizer: Path
    config: WriterConfig = dataclasses.field(default_factory=WriterConfig)
    #: JSON key with the text (jsonl only).
    field: str = "text"
    #: File pattern; ``None`` means ``*.txt`` / ``*.jsonl``.
    pattern: str | None = None

    def open_input(self) -> TxtDirInput | JsonlInput:
        if self.input_kind == "txt":
            return TxtDirInput(self.input_path, pattern=self.pattern or "*.txt")
        return JsonlInput(self.input_path, field=self.field, pattern=self.pattern or "*.jsonl")

    def to_dict(self) -> dict[str, object]:
        return {
            "format": RECIPE_FORMAT,
            "input": {
                "kind": self.input_kind,
                "path": str(self.input_path.resolve()),
                "field": self.field,
                "pattern": self.pattern,
            },
            "tokenizer": str(self.tokenizer.resolve()),
            "config": {name: getattr(self.config, name) for name in _CONFIG_FIELDS},
        }

    @classmethod
    def from_dict(cls, data: object, where: str) -> BuildRecipe:
        f = Fields(data, where)
        if f.get_int("format", minimum=1) != RECIPE_FORMAT:
            raise f.error("format", f"expected {RECIPE_FORMAT}")
        inp = f.get_object("input")
        kind = inp.get_str("kind")
        if kind not in ("txt", "jsonl"):
            raise inp.error("kind", "expected 'txt' or 'jsonl'")
        cfg = f.get_object("config")
        values: dict[str, object] = {}
        for name in cfg.field_names():
            if name not in _CONFIG_FIELDS:
                raise cfg.error(name, "unknown WriterConfig field")
            values[name] = cfg.get_any(name)
        try:
            config = WriterConfig(**values)  # type: ignore[arg-type]
        except ConfigError as exc:
            raise cfg.error("config", f"invalid settings: {exc.what}") from exc
        return cls(
            input_kind=kind,  # type: ignore[arg-type]
            input_path=Path(inp.get_str("path")),
            field=inp.get_str("field"),
            pattern=inp.get_optional_str("pattern"),
            tokenizer=Path(f.get_str("tokenizer")),
            config=config,
        )


def read_recipe(target: Path) -> BuildRecipe | None:
    """The saved recipe of an unfinished build of ``target``, if there is one."""
    path = partial_path(target) / naming.BUILD_RECIPE
    if not path.is_file():
        return None
    return BuildRecipe.from_dict(read_json_bounded(path), where=str(path))


def _tracked(source: TxtDirInput | JsonlInput, progress: ProgressCallback) -> Iterator[InputDoc]:
    for doc in source:
        progress(source.bytes_read, source.total_bytes, source.current)
        yield doc


@public_api
def build_source(
    target: str | Path,
    recipe: BuildRecipe | None = None,
    *,
    policy: ErrorPolicy | None = None,
    overwrite: bool = False,
    resume: bool = False,
    progress: ProgressCallback | None = None,
) -> WriteResult:
    """Write the source ``target`` from a directory of text files or JSON Lines.

    With ``resume=True`` and no ``recipe``, the settings saved by the interrupted build
    are used.
    """
    target = Path(target)
    naming.check_source_name(target.name)
    if recipe is None:
        saved = read_recipe(target) if resume else None
        if saved is None:
            raise ConfigError(
                codes.BUILD_RECIPE_MISSING,
                str(target),
                why="no input was given, and there is no interrupted build of this source "
                f"with saved settings ({partial_path(target) / naming.BUILD_RECIPE})"
                if resume
                else "no input was given",
                fix="pass the input and the tokenizer (--from-txt / --from-jsonl and --tokenizer)",
            )
        recipe = saved
    if not recipe.tokenizer.is_file():
        raise ConfigError(
            codes.TOKENIZER_FILE_INVALID,
            str(recipe.tokenizer),
            why="the tokenizer file does not exist",
            fix="pass the path to tokenizer.json",
        )
    source = recipe.open_input()
    docs: TxtDirInput | JsonlInput | Iterator[InputDoc] = (
        source if progress is None else _tracked(source, progress)
    )
    with StreamWriter(
        target,
        recipe.tokenizer,
        config=recipe.config,
        policy=policy,
        overwrite=overwrite,
        resume=resume,
    ) as writer:
        partial = writer.partial_dir
        if partial is not None and not (partial / naming.BUILD_RECIPE).is_file():
            write_json_atomic(partial / naming.BUILD_RECIPE, recipe.to_dict())
        writer.write(docs)
    return writer.result
