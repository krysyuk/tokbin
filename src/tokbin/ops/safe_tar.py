"""Reading and writing ``.tar`` files of packs without trusting their content (spec 9.3).

Nothing is ever extracted with ``TarFile.extract``: members are validated up front and
read one by one through ``extractfile``, and the caller decides where the data goes.
Refused: absolute paths, ``..``, backslashes and drive letters, links, devices, FIFOs,
duplicate names and anything outside the single top-level ``*.tbpack`` directory. The
same rules apply on every Python version; there is no dependence on the availability of
``tarfile`` extraction filters.
"""

from __future__ import annotations

import tarfile
from pathlib import Path, PurePosixPath
from typing import IO

from tokbin import codes
from tokbin.errors import FormatError

__all__ = ["TarPack", "check_relative_path", "write_tar"]

PACK_SUFFIX = ".tbpack"


def _unsafe(where: str, name: str, why: str) -> FormatError:
    return FormatError(
        codes.ARCHIVE_ENTRY_UNSAFE,
        f"{where}: {name!r}",
        why=why,
        fix="the archive is damaged or crafted; do not use it, get the dataset from its "
        "original source",
    )


def check_relative_path(name: str, where: str) -> PurePosixPath:
    """A safe relative POSIX path, or ``FormatError``."""
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or "\0" in name
        or path.is_absolute()
        or (path.parts and ":" in path.parts[0])
        or any(part in ("", ".", "..") for part in name.split("/"))
    ):
        raise _unsafe(where, name, "entries must be plain relative paths inside the pack")
    return path


class TarPack:
    """A pack stored as one ``.tar``: ``<name>.tbpack/`` and the files below it."""

    def __init__(self, path: Path) -> None:
        self.path = path
        where = str(path)
        try:
            self._tar = tarfile.open(path, "r:")  # noqa: SIM115 - closed in close()
        except (tarfile.TarError, OSError) as exc:
            raise FormatError(
                codes.NOT_A_PACK,
                where,
                why=f"it is not an uncompressed tar archive: {exc}",
                fix="pass a .tbpack directory or the .tar made by `tokbin pack --tar`",
            ) from exc
        try:
            self._files = self._index(where)
        except BaseException:
            self._tar.close()
            raise

    def _index(self, where: str) -> dict[str, tarfile.TarInfo]:
        files: dict[str, tarfile.TarInfo] = {}
        top: str | None = None
        seen: set[str] = set()
        for member in self._tar.getmembers():
            path = check_relative_path(member.name.rstrip("/"), where)
            if member.name in seen:
                raise _unsafe(where, member.name, "the name appears twice")
            seen.add(member.name)
            if not (member.isfile() or member.isdir()):
                raise _unsafe(where, member.name, "links, devices and FIFOs are not allowed")
            first = path.parts[0]
            if top is None:
                top = first
            if first != top or not top.endswith(PACK_SUFFIX):
                raise _unsafe(
                    where, member.name, f"everything must be inside one *{PACK_SUFFIX} directory"
                )
            if member.isfile():
                if len(path.parts) < 2:
                    raise _unsafe(where, member.name, f"a file outside the {PACK_SUFFIX} directory")
                files[str(PurePosixPath(*path.parts[1:]))] = member
        if top is None:
            raise FormatError(
                codes.NOT_A_PACK,
                where,
                why="the archive is empty",
                fix="pass the .tar made by `tokbin pack --tar`",
            )
        return files

    def names(self) -> set[str]:
        return set(self._files)

    def size(self, name: str) -> int:
        return self._files[name].size

    def open(self, name: str) -> IO[bytes]:
        member = self._files[name]
        handle = self._tar.extractfile(member)
        if handle is None:  # pragma: no cover - only regular files are indexed
            raise _unsafe(str(self.path), name, "not a regular file")
        return handle

    def close(self) -> None:
        self._tar.close()


def write_tar(directory: Path, out: Path, *, top: str) -> None:
    """Write ``directory`` into an uncompressed, reproducible tar under the name ``top``.

    Entries are sorted; owners, times and permissions are normalized, so the same pack
    always gives the same bytes.
    """

    def normalize(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        info.mtime = 0
        info.mode = 0o755 if info.isdir() else 0o644
        return info

    with tarfile.open(out, "w", format=tarfile.PAX_FORMAT) as tar:
        tar.add(directory, arcname=top, recursive=False, filter=normalize)
        for path in sorted(directory.rglob("*")):
            arcname = f"{top}/{path.relative_to(directory).as_posix()}"
            tar.add(path, arcname=arcname, recursive=False, filter=normalize)
