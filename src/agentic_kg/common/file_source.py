"""Single owner of source-file access.

Every component that reads a source file goes through this module. It resolves
the configured SOURCE_URI into an fsspec filesystem plus a root path, and
exposes listing, opening and existence checks by *relative* name.

The relative-name convention matters: construction plans record files as
"assemblies.csv", not as absolute locations, so a plan built against a local
folder still works when the same files move elsewhere. This module is the one
place that knows the difference.
"""

import codecs
import logging
from pathlib import Path, PureWindowsPath
from typing import Any, BinaryIO, NamedTuple, Optional, Tuple, cast

from fsspec import AbstractFileSystem
from fsspec.core import url_to_fs

from .config import get_settings

logger = logging.getLogger(__name__)

# .../<repo>/src/agentic_kg/common/file_source.py -> .../<repo>
_REPO_ROOT = Path(__file__).resolve().parents[3]


class SourceError(Exception):
    """The source location, or a file in it, is unusable."""


class SourceEncodingError(SourceError):
    """A source file is not text this program can read.

    Raised for UTF-16, binary files and anything that is neither UTF-8 nor
    Windows-1252. The message names the file and says why, so a tool can hand
    it to the agent unchanged.
    """


def _anchor(uri: str) -> str:
    """Absolutise a relative local path against the repository root.

    fsspec resolves relative paths against the process working directory, and
    `adk web` makes no promise about what that is. Anything with a scheme, and
    anything already absolute, passes through untouched.
    """
    if "://" in uri:
        return uri
    path = Path(uri)
    if path.is_absolute():
        return str(path)
    return str((_REPO_ROOT / path).resolve())


def get_source_fs() -> Tuple[AbstractFileSystem, str]:
    """Return the configured filesystem and its root path.

    Raises:
        SourceError: if SOURCE_URI is unset, or names a scheme whose backing
            package is not installed (e.g. s3:// without s3fs).
    """
    uri = get_settings().source_uri
    if not uri:
        raise SourceError(
            "SOURCE_URI is not set. Point it at a folder of source files, "
            "for example SOURCE_URI=./data/bom"
        )
    try:
        fs, root = url_to_fs(_anchor(uri))
    except ImportError as exc:
        raise SourceError(
            f"SOURCE_URI '{uri}' needs a package that is not installed: {exc}"
        ) from exc
    except Exception as exc:  # noqa: BLE001 - fsspec raises a variety of types
        raise SourceError(f"SOURCE_URI '{uri}' could not be resolved: {exc}") from exc
    return fs, root.rstrip("/")


def get_source_root() -> str:
    """Return the resolved root path, for display to a user."""
    _fs, root = get_source_fs()
    return root


def _checked_relative(relative_path: str) -> str:
    """Reject any name that would resolve outside the source root.

    Source names reach here from an LLM-proposed construction plan or from a
    tool argument the model chose, not only from list_source_files(). Joining
    such a name unchecked means "../.env" reads the developer's OpenRouter key
    and Neo4j password on the local filesystem and returns them into the
    model's context.

    Confinement is by name only. A symlink inside the root that points outside
    it is still followed, so a source location must be a directory whose
    contents are trusted.

    Raises:
        SourceError: if the name is absolute, carries a scheme, or contains a
            ".." segment.
    """
    if not isinstance(relative_path, str) or not relative_path:
        raise SourceError(f"Not a usable source file name: {relative_path!r}")
    if "://" in relative_path:
        raise SourceError(
            f"Source file names are relative to the source root, so '{relative_path}' "
            "cannot name a location of its own."
        )
    normalised = relative_path.replace("\\", "/")
    if normalised.startswith("/") or PureWindowsPath(relative_path).is_absolute():
        raise SourceError(
            f"Source file names must be relative to the source root: '{relative_path}'"
        )
    if any(part == ".." for part in normalised.split("/")):
        raise SourceError(
            f"Source file names cannot leave the source root: '{relative_path}'"
        )
    # The backslash form is normalised for the checks above only. A backslash is
    # an ordinary character in a POSIX file name, so returning the rewritten
    # version would make a file that list_source_files() just reported
    # unopenable.
    return relative_path


def _full_path(root: str, relative_path: str) -> str:
    """Join a checked relative name onto an already-resolved root."""
    return f"{root}/{_checked_relative(relative_path)}"


def source_path(relative_path: str) -> str:
    """Return the filesystem-native absolute path for a relative name.

    Raises:
        SourceError: if the name would resolve outside the source root.
    """
    _fs, root = get_source_fs()
    return _full_path(root, relative_path)


def list_source_files() -> list[str]:
    """List every file under the source root, as sorted relative names.

    Raises:
        SourceError: if the root does not exist.
    """
    fs, root = get_source_fs()
    if not fs.exists(root):
        raise SourceError(f"Source location does not exist: {root}")
    # Strip the root and then any separator, rather than slicing at
    # len(root) + 1. The slice is only correct while get_source_fs() returns a
    # root with no trailing separator, so it silently depended on that rstrip:
    # anyone who made the root keep its trailing "/" would have cut a character
    # off every name here instead of getting an error.
    return sorted(found[len(root) :].lstrip("/") for found in fs.find(root))


def source_exists(relative_path: str) -> bool:
    """Whether a file exists at the given relative name."""
    fs, root = get_source_fs()
    return bool(fs.exists(_full_path(root, relative_path)))


# Bytes per read of the encoding scan. A module constant so a test can shrink it
# and put a chunk boundary inside a multi-byte character.
_SCAN_CHUNK = 1 << 20

_UTF16_BOMS = (b"\xff\xfe", b"\xfe\xff")
_UTF8_BOM = b"\xef\xbb\xbf"


class _Decision(NamedTuple):
    """How to read one file: an encoding, or the reason it is refused."""

    encoding: Optional[str]
    refusal: Optional[str] = None


def _refused(reason: str) -> _Decision:
    return _Decision(None, reason)


def _refusal_message(relative_path: str, reason: str) -> str:
    """The one wording of a refusal. The path is the caller's relative name."""
    return (
        f"{relative_path} is not valid UTF-8 or Windows-1252 text, so it was not "
        f"read: {reason}. Re-save it as UTF-8."
    )


def _outcome(
    utf8: codecs.IncrementalDecoder,
    utf8_ok: bool,
    has_utf8_bom: bool,
    cp1252_failure: Optional[Tuple[int, int]],
) -> _Decision:
    """The decision once every chunk has been read. See _scan."""
    if utf8_ok:
        try:
            utf8.decode(b"", final=True)
            return _Decision("utf-8-sig")
        except UnicodeDecodeError:
            pass  # a multi-byte character cut off at the end of the file
    if has_utf8_bom:
        return _refused("it starts with a UTF-8 byte-order mark but is not valid UTF-8")
    if cp1252_failure is None:
        return _Decision("cp1252")
    position, byte = cp1252_failure
    return _refused(
        f"byte 0x{byte:02X} at offset {position} is not valid Windows-1252, "
        "and the file is not valid UTF-8"
    )


def _scan(fs: AbstractFileSystem, full_path: str) -> _Decision:
    """Decide how to read a file, from one pass over its bytes.

    The two decoders are tracked independently and a failure of one is never a
    refusal on its own: valid UTF-8 routinely contains bytes Windows-1252 leaves
    undefined ("Ё" is D0 81), and such a file must keep reading as UTF-8. Only
    a NUL byte, a UTF-16 byte-order mark, or both decoders failing refuses a
    file. A NUL byte is looked for in every chunk, so a file with an invalid byte
    early and a NUL later still gets the "UTF-16 or binary" hint.

    A file starting with a UTF-8 byte-order mark is UTF-8 only: read as
    Windows-1252 the BOM would become "ï»¿" glued onto the first header name.

    Chunked, so a large file is never held whole. The UTF-8 decoder is
    incremental, which carries a multi-byte character split by a chunk
    boundary; `final=True` after the last chunk (in _outcome) catches a
    truncated one.
    """
    utf8 = codecs.getincrementaldecoder("utf-8")()
    utf8_ok = True
    has_utf8_bom = False
    cp1252_failure: Optional[Tuple[int, int]] = None  # (absolute offset, byte)
    offset = 0
    with fs.open(full_path, "rb") as opened:
        # fsspec types read() as str | bytes; a "rb" open yields bytes.
        handle = cast(BinaryIO, opened)
        head = handle.read(3)
        if head[:2] in _UTF16_BOMS:
            return _refused("it looks like UTF-16 or binary (a UTF-16 byte-order mark)")
        has_utf8_bom = head == _UTF8_BOM
        handle.seek(0)
        while chunk := handle.read(_SCAN_CHUNK):
            nul = chunk.find(b"\x00")
            if nul != -1:
                return _refused(
                    f"it looks like UTF-16 or binary (a NUL byte at offset {offset + nul})"
                )
            if utf8_ok:
                try:
                    utf8.decode(chunk)
                except UnicodeDecodeError:
                    utf8_ok = False
            if cp1252_failure is None:
                try:
                    chunk.decode("cp1252")
                except UnicodeDecodeError as exc:
                    cp1252_failure = (offset + exc.start, chunk[exc.start])
            offset += len(chunk)
    return _outcome(utf8, utf8_ok, has_utf8_bom, cp1252_failure)


def open_source(relative_path: str, mode: str = "r", **kwargs: Any) -> Any:
    """Open a source file by relative name.

    Text mode is the default because clevercsv requires an iterable of str.

    Raises:
        FileNotFoundError: if the file does not exist.
        SourceError: if the source location is misconfigured.
    """
    fs, root = get_source_fs()
    full_path = _full_path(root, relative_path)
    if not fs.exists(full_path):
        raise FileNotFoundError(f"No such source file: {relative_path}")
    if "b" not in mode:
        kwargs.setdefault("newline", "")
        # Without an explicit encoding, TextIOWrapper falls back to
        # locale.getpreferredencoding(False). On a non-UTF-8 locale that is
        # silent mojibake, not an exception, and every bundled CSV under
        # data/bom/ contains non-ASCII characters.
        kwargs.setdefault("encoding", "utf-8")
    return fs.open(full_path, mode, **kwargs)
