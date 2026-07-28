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


# Source names outside the root

def test_parent_traversal_is_rejected(memory_source):
    """A construction plan is LLM-produced, so a name like "../.env" would
    otherwise read the developer's OpenRouter key and Neo4j password straight
    back into the model's context."""
    with pytest.raises(file_source.SourceError, match="leave the source root"):
        file_source.source_path("../secret.env")


def test_traversal_in_the_middle_of_a_name_is_rejected(memory_source):
    with pytest.raises(file_source.SourceError, match="leave the source root"):
        file_source.source_path("nested/../../secret.env")


def test_windows_style_traversal_is_rejected(memory_source):
    """Backslashes are normalised before the checks precisely so these are
    caught. Only the returned name keeps its original form."""
    for name in ("..\\secret.env", "nested\\..\\..\\secret.env"):
        with pytest.raises(file_source.SourceError, match="leave the source root"):
            file_source.source_path(name)


def test_absolute_names_are_rejected(memory_source):
    with pytest.raises(file_source.SourceError, match="relative to the source root"):
        file_source.source_path("/etc/passwd")


def test_names_with_a_scheme_are_rejected(memory_source):
    with pytest.raises(file_source.SourceError, match="cannot name a location"):
        file_source.source_path("file:///etc/passwd")


def test_a_dot_in_a_name_is_still_allowed(memory_source):
    """Only a whole ".." segment escapes. Names that merely contain dots,
    including a doubled one, are ordinary file names."""
    assert file_source.source_path("a..b.csv") == "/src/a..b.csv"
    assert file_source.source_path("..hidden.csv") == "/src/..hidden.csv"
    assert file_source.source_path("nested/deep.md") == "/src/nested/deep.md"


def test_source_exists_reports_traversal_as_an_error_not_a_hit(memory_source):
    with pytest.raises(file_source.SourceError):
        file_source.source_exists("../top.csv")


def test_traversal_is_still_rejected_through_open_source(memory_source):
    """The single-resolution path must keep the confinement check."""
    with pytest.raises(file_source.SourceError, match="leave the source root"):
        file_source.open_source("../top.csv")


def test_a_backslash_in_a_name_survives_the_round_trip(monkeypatch, tmp_path):
    """A backslash is an ordinary character in a POSIX file name. Normalising
    it into the returned path made a file that list_source_files() had just
    reported impossible to open."""
    (tmp_path / "a\\b.csv").write_text("a,b\n1,2\n")
    monkeypatch.setenv("SOURCE_URI", str(tmp_path))
    reset_settings()
    listed = file_source.list_source_files()
    assert listed == ["a\\b.csv"]
    assert file_source.source_exists(listed[0]) is True
    with file_source.open_source(listed[0]) as handle:
        assert handle.read() == "a,b\n1,2\n"
