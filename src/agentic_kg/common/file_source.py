"""Single owner of source-file access.

Every component that reads a source file goes through this module. It resolves
the configured SOURCE_URI into an fsspec filesystem plus a root path, and
exposes listing, opening and existence checks by *relative* name.

The relative-name convention matters: construction plans record files as
"assemblies.csv", not as absolute locations, so a plan built against a local
folder still works when the same files move elsewhere. This module is the one
place that knows the difference.
"""
import logging
from pathlib import Path
from typing import Any, Tuple

from fsspec import AbstractFileSystem
from fsspec.core import url_to_fs

from .config import get_settings

logger = logging.getLogger(__name__)

# .../<repo>/src/agentic_kg/common/file_source.py -> .../<repo>
_REPO_ROOT = Path(__file__).resolve().parents[3]


class SourceError(Exception):
    """The configured source location is unusable."""


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


def source_path(relative_path: str) -> str:
    """Return the filesystem-native absolute path for a relative name."""
    _fs, root = get_source_fs()
    return f"{root}/{relative_path}"


def list_source_files() -> list[str]:
    """List every file under the source root, as sorted relative names.

    Raises:
        SourceError: if the root does not exist.
    """
    fs, root = get_source_fs()
    if not fs.exists(root):
        raise SourceError(f"Source location does not exist: {root}")
    prefix_length = len(root) + 1
    return sorted(found[prefix_length:] for found in fs.find(root))


def source_exists(relative_path: str) -> bool:
    """Whether a file exists at the given relative name."""
    fs, _root = get_source_fs()
    return bool(fs.exists(source_path(relative_path)))


def open_source(relative_path: str, mode: str = "r", **kwargs: Any):
    """Open a source file by relative name.

    Text mode is the default because clevercsv requires an iterable of str.

    Raises:
        FileNotFoundError: if the file does not exist.
        SourceError: if the source location is misconfigured.
    """
    fs, _root = get_source_fs()
    full_path = source_path(relative_path)
    if not fs.exists(full_path):
        raise FileNotFoundError(f"No such source file: {relative_path}")
    if "b" not in mode:
        kwargs.setdefault("newline", "")
    return fs.open(full_path, mode, **kwargs)
