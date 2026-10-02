"""``tokbin build <target>``: write a source from text files or JSON Lines (spec 12.3)."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from tokbin.cli.context import Context, Outcome, plural
from tokbin.cli.help import error_lines
from tokbin.cli.render import fmt_count, fmt_int
from tokbin.ops.build import BuildRecipe, build_source
from tokbin.write.config import WriterConfig

NAME = "build"
HELP = "Tokenize text files or JSON Lines into a source"

_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgt]?)(i?b)?\s*$", re.IGNORECASE)
_UNITS = {"": 1, "k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}

#: Options that describe a build; with --resume and no input they come from build.json.
_RECIPE_OPTIONS = (
    "tokenizer",
    "field",
    "pattern",
    "split",
    "shard_size",
    "no_eos",
    "bos",
    "eos_token",
    "bos_token",
    "batch_docs",
)


def parse_size(text: str) -> int:
    """``512M``, ``1G``, ``256MiB``, ``1048576``: binary units."""
    match = _SIZE_RE.match(text)
    if not match:
        raise argparse.ArgumentTypeError(f"not a size: {text!r} (examples: 512M, 1G, 1048576)")
    return int(float(match.group(1)) * _UNITS[match.group(2).lower()])


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("target", help="Source directory to write, e.g. corpus/web")
    inputs = parser.add_argument_group("Input")
    source = inputs.add_mutually_exclusive_group()
    source.add_argument("--from-txt", metavar="DIR", help="Every *.txt file is a document")
    source.add_argument(
        "--from-jsonl", metavar="PATH", help="Every line of *.jsonl files is a document"
    )
    inputs.add_argument("--field", metavar="KEY", help="JSON key with the text [default: text]")
    inputs.add_argument("--pattern", metavar="GLOB", help="File pattern [default: *.txt / *.jsonl]")
    tokenizer = parser.add_argument_group("Tokenizer")
    tokenizer.add_argument("--tokenizer", metavar="PATH", help="Path to tokenizer.json")
    tokenizer.add_argument(
        "--no-eos", action="store_true", default=None, help="No EOS after documents"
    )
    tokenizer.add_argument("--bos", action="store_true", default=None, help="BOS before documents")
    tokenizer.add_argument("--eos-token", metavar="TEXT", help="EOS token text [default: detected]")
    tokenizer.add_argument("--bos-token", metavar="TEXT", help="BOS token text [default: detected]")
    tokenizer.add_argument(
        "--batch-docs", metavar="N", type=int, help="Documents per tokenizer batch"
    )
    output = parser.add_argument_group("Output")
    output.add_argument(
        "--split", choices=("train", "valid", "test"), help="Split to write [default: train]"
    )
    output.add_argument(
        "--shard-size", metavar="SIZE", type=parse_size, help="Shard size [default: 512M]"
    )
    output.add_argument("--overwrite", action="store_true", help="Replace an existing split")
    output.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted build; without an input its saved settings are used",
    )


class UsageError(Exception):
    """Wrong combination of options: exit code 2."""


def _recipe(args: argparse.Namespace) -> BuildRecipe | None:
    if args.from_txt is None and args.from_jsonl is None:
        given = [
            f"--{o.replace('_', '-')}" for o in _RECIPE_OPTIONS if getattr(args, o) is not None
        ]
        if not args.resume:
            raise UsageError("pass --from-txt DIR or --from-jsonl PATH (or --resume)")
        if given:
            raise UsageError(
                f"these options need an input: {', '.join(given)}; with --resume alone the "
                "settings of the interrupted build are used"
            )
        return None
    if args.tokenizer is None:
        raise UsageError("--tokenizer PATH is required")
    if args.from_txt is not None and args.field is not None:
        raise UsageError("--field applies to --from-jsonl only")
    defaults = WriterConfig()
    config = WriterConfig(
        split=args.split or defaults.split,
        shard_bytes=args.shard_size or defaults.shard_bytes,
        append_eos=not args.no_eos,
        prepend_bos=bool(args.bos),
        eos_token=args.eos_token,
        bos_token=args.bos_token,
        batch_docs=args.batch_docs or defaults.batch_docs,
    )
    return BuildRecipe(
        input_kind="txt" if args.from_txt is not None else "jsonl",
        input_path=Path(args.from_txt if args.from_txt is not None else args.from_jsonl),
        tokenizer=Path(args.tokenizer),
        config=config,
        field=args.field or "text",
        pattern=args.pattern,
    )


def run(args: argparse.Namespace, ctx: Context) -> Outcome:
    try:
        recipe = _recipe(args)
    except UsageError as exc:
        lines = error_lines(ctx.err_style, str(exc), prog=f"tokbin {NAME}")
        ctx.err.write("\n".join(lines) + "\n")
        return Outcome(exit_code=2, data={"usage_error": str(exc)})
    progress = ctx.progress()

    def report(done: int, total: int, name: str) -> None:
        if progress is not None:
            progress.update(done, total, name)

    try:
        result = build_source(
            args.target, recipe, overwrite=args.overwrite, resume=args.resume, progress=report
        )
    except KeyboardInterrupt:
        if progress is not None:
            progress.close()
        ctx.err.write(
            f"the last checkpoint is kept; continue with: tokbin build {args.target} --resume\n"
        )
        raise
    finally:
        if progress is not None:
            progress.close()

    s = ctx.style
    st = result.stats
    ctx.line(f"{s.mark('ok')} built {result.path} [{result.split}]")
    ctx.line(
        "  "
        + s.join(
            f"{fmt_int(st.n_docs)} docs",
            f"{fmt_count(st.n_items)} tokens",
            plural(st.n_shards, "shard"),
            f"{fmt_int(st.n_skipped)} skipped" if st.n_skipped else "",
        )
    )
    for issue in sorted(result.issues, key=lambda i: i.level != "warning"):
        paint = s.yellow if issue.level == "warning" else s.dim
        count = f" (x{issue.count:,})" if issue.count > 1 else ""
        ctx.line(f"  {s.dim(s.branch)} {paint(f'[{issue.code}] {issue.message}{count}')}")
    warnings_ = [i for i in result.issues if i.level == "warning"]
    ctx.line()
    if warnings_:
        status = s.join("complete", plural(len(warnings_), "warning"))
        ctx.line(f"{s.mark('warn')} {s.bold('Status:')} {status}")
    else:
        ctx.line(f"{s.mark('ok')} {s.bold('Status:')} complete")
    data: dict[str, object] = {
        "path": str(result.path),
        "split": result.split,
        "status": result.status,
        "stats": {
            "n_input": st.n_input,
            "n_docs": st.n_docs,
            "n_skipped": st.n_skipped,
            "n_items": st.n_items,
            "n_shards": st.n_shards,
            "n_split_docs": st.n_split_docs,
        },
        "issues": [
            {"code": i.code, "level": i.level, "message": i.message, "count": i.count}
            for i in result.issues
        ],
    }
    return Outcome(
        exit_code=0,
        data=data,
        n_warnings=len(warnings_),
        shown_codes=frozenset(i.code for i in result.issues),
    )
