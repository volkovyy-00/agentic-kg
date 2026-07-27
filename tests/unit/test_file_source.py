import io

import fsspec
import pytest

from agentic_kg.common.config import reset_settings
from agentic_kg.common import file_source


@pytest.fixture
def memory_source(monkeypatch):
    """A memory:// source populated with two files, one in a subdirectory.

    The fsspec memory filesystem is a process-global singleton, so the store
    must be cleared between tests or files leak across cases.
    """
    fs = fsspec.filesystem("memory")
    fs.store.clear()
    fs.pseudo_dirs.clear()
    with fs.open("/src/top.csv", "w") as handle:
        handle.write("a,b\n1,2\n")
    with fs.open("/src/nested/deep.md", "w") as handle:
        handle.write("# hello\n")
    monkeypatch.setenv("SOURCE_URI", "memory://src")
    reset_settings()
    yield fs
    fs.store.clear()
    fs.pseudo_dirs.clear()


def test_lists_relative_names_recursively(memory_source):
    assert file_source.list_source_files() == ["nested/deep.md", "top.csv"]


def test_opens_by_relative_name(memory_source):
    with file_source.open_source("top.csv") as handle:
        assert handle.read() == "a,b\n1,2\n"


def test_opens_file_in_subdirectory(memory_source):
    with file_source.open_source("nested/deep.md") as handle:
        assert handle.read() == "# hello\n"


def test_source_exists(memory_source):
    assert file_source.source_exists("top.csv") is True
    assert file_source.source_exists("absent.csv") is False


def test_opening_a_missing_file_raises(memory_source):
    with pytest.raises(FileNotFoundError):
        file_source.open_source("absent.csv")


def test_source_path_is_native_not_relative(memory_source):
    assert file_source.source_path("top.csv") == "/src/top.csv"


def test_unset_source_uri_raises_source_error(monkeypatch):
    monkeypatch.delenv("SOURCE_URI", raising=False)
    reset_settings()
    with pytest.raises(file_source.SourceError, match="SOURCE_URI"):
        file_source.get_source_fs()


def test_relative_path_anchors_to_repo_root_not_cwd(monkeypatch, tmp_path):
    monkeypatch.setenv("SOURCE_URI", "./data/bom")
    reset_settings()
    monkeypatch.chdir(tmp_path)
    _fs, root = file_source.get_source_fs()
    assert root.endswith("/data/bom")
    assert str(tmp_path) not in root


def test_open_source_reads_non_ascii_content_as_utf8(memory_source, monkeypatch):
    """open_source's text mode must not fall back to
    locale.getpreferredencoding(): every bundled CSV under data/bom/ contains
    non-ASCII (Swedish) characters, so on a non-UTF-8 locale that fallback is
    silent mojibake in the constructed graph, not an exception. Faking the
    process locale is not reliably observable by TextIOWrapper (its default
    encoding is resolved once, not looked up live), so this pins the actual
    encoding kwarg that reaches fsspec's TextIOWrapper instead.
    """
    payload = "Björk café".encode("utf-8")
    with memory_source.open("/src/nonascii.csv", "wb") as handle:
        handle.write(payload)

    captured_kwargs = {}
    real_text_io_wrapper = io.TextIOWrapper

    class _SpyTextIOWrapper(real_text_io_wrapper):
        def __init__(self, buffer, *args, **kwargs):
            captured_kwargs.update(kwargs)
            super().__init__(buffer, *args, **kwargs)

    monkeypatch.setattr(io, "TextIOWrapper", _SpyTextIOWrapper)

    with file_source.open_source("nonascii.csv") as handle:
        assert handle.read() == "Björk café"

    assert captured_kwargs.get("encoding") == "utf-8"


def test_uninstalled_scheme_raises_source_error(monkeypatch):
    monkeypatch.setenv("SOURCE_URI", "s3://some-bucket/prefix")
    reset_settings()
    with pytest.raises(file_source.SourceError, match="s3"):
        file_source.get_source_fs()
