import io

import fsspec
import pytest

from agentic_kg.common import file_source
from agentic_kg.common.config import reset_settings


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


def test_listing_does_not_depend_on_the_root_lacking_a_separator(
    memory_source, monkeypatch
):
    """The relative names must come out the same whether or not the resolved
    root keeps a trailing separator. Slicing at len(root) + 1 was correct only
    for the stripped form, so a change to that stripping would have silently
    cut a character off every name rather than failing."""
    baseline = file_source.list_source_files()
    assert baseline == ["nested/deep.md", "top.csv"]

    real = file_source.get_source_fs
    monkeypatch.setattr(
        file_source, "get_source_fs", lambda: (real()[0], real()[1] + "/")
    )
    assert file_source.list_source_files() == baseline


# --- encoding detection (KG-47) ------------------------------------------------

_NOT_TEXT = "is not valid UTF-8 or Windows-1252 text, so it was not read"


def _put(fs, name, data):
    with fs.open(f"/src/{name}", "wb") as handle:
        handle.write(data)


def _scan(name):
    fs, root = file_source.get_source_fs()
    return file_source._scan(fs, f"{root}/{name}")


def test_a_utf8_file_is_read_as_utf8_sig(memory_source):
    _put(memory_source, "a.csv", "id,name\n1,Luleå\n".encode("utf-8"))
    assert _scan("a.csv") == file_source._Decision("utf-8-sig")


def test_a_windows_1252_file_is_read_as_cp1252(memory_source):
    _put(
        memory_source,
        "a.csv",
        "id,name\n1,Luleå\n2,Rössle Sauerkraut\n".encode("cp1252"),
    )
    assert _scan("a.csv") == file_source._Decision("cp1252")


def test_an_empty_file_and_a_bom_only_file_are_utf8(memory_source):
    _put(memory_source, "empty.csv", b"")
    _put(memory_source, "bom.csv", b"\xef\xbb\xbf")
    assert _scan("empty.csv") == file_source._Decision("utf-8-sig")
    assert _scan("bom.csv") == file_source._Decision("utf-8-sig")


def test_valid_utf8_that_cp1252_rejects_is_still_utf8(memory_source):
    """Review focus 1. 'Ё' is D0 81 and 0x81 is undefined in cp1252; a Devanagari
    letter can carry 0x8D. A cp1252 failure alone must never refuse a file."""
    _put(memory_source, "a.csv", "Ёлка,नमस्ते\n".encode("utf-8"))
    assert _scan("a.csv") == file_source._Decision("utf-8-sig")


def test_a_truncated_lead_byte_at_the_end_reads_as_cp1252(memory_source):
    """The lone 0xC3 is a valid UTF-8 prefix, so UTF-8 fails only at final=True."""
    _put(memory_source, "a.csv", b"cafe \xc3")
    assert _scan("a.csv") == file_source._Decision("cp1252")


@pytest.mark.parametrize(
    ("text", "chunk"),
    [("aé", 2), ("xxxé", 4), ("ééé", 1), ("\ufeffid\n", 1)],
    ids=["split-after-lead", "lead-ends-chunk", "byte-per-chunk", "bom-byte-per-chunk"],
)
def test_a_character_split_across_chunks_still_decodes(
    memory_source, monkeypatch, text, chunk
):
    monkeypatch.setattr(file_source, "_SCAN_CHUNK", chunk)
    _put(memory_source, "a.csv", text.encode("utf-8"))
    assert _scan("a.csv") == file_source._Decision("utf-8-sig")


def test_cp1252_is_decided_with_one_byte_chunks(memory_source, monkeypatch):
    monkeypatch.setattr(file_source, "_SCAN_CHUNK", 1)
    _put(memory_source, "a.csv", "Luleå".encode("cp1252"))
    assert _scan("a.csv") == file_source._Decision("cp1252")


def test_a_file_valid_in_neither_encoding_is_refused(memory_source):
    _put(memory_source, "a.csv", b"id,name\n1,Bj\x81rk\n")
    decision = _scan("a.csv")
    assert decision.encoding is None
    message = file_source._refusal_message("a.csv", decision.refusal)
    assert message == (
        f"a.csv {_NOT_TEXT}: byte 0x81 at offset 12 is not valid Windows-1252, "
        "and the file is not valid UTF-8. Re-save it as UTF-8."
    )


def test_the_offset_is_absolute_across_chunks(memory_source, monkeypatch):
    monkeypatch.setattr(file_source, "_SCAN_CHUNK", 4)
    _put(memory_source, "a.csv", b"abcdefg\x81")
    assert "offset 7 " in _scan("a.csv").refusal


def test_a_nul_byte_is_refused_as_utf16_or_binary(memory_source):
    _put(memory_source, "a.csv", b"id,na\x00me\n")
    assert _scan("a.csv").refusal == (
        "it looks like UTF-16 or binary (a NUL byte at offset 5)"
    )


def test_a_nul_byte_in_a_later_chunk_reports_its_true_offset(
    memory_source, monkeypatch
):
    monkeypatch.setattr(file_source, "_SCAN_CHUNK", 4)
    _put(memory_source, "a.csv", b"abcdefgh\x00")
    assert "NUL byte at offset 8" in _scan("a.csv").refusal


def test_a_nul_after_an_invalid_byte_still_gets_the_binary_hint(
    memory_source, monkeypatch
):
    """Both decoders failing does not end the scan: a NUL later in the file is
    still what SC2 promises to report."""
    monkeypatch.setattr(file_source, "_SCAN_CHUNK", 4)
    _put(memory_source, "a.csv", b"\x81bcdefgh\x00")
    assert "UTF-16 or binary" in _scan("a.csv").refusal


def test_utf16_with_a_bom_is_refused(memory_source):
    _put(memory_source, "a.csv", "id,name\n".encode("utf-16"))
    assert _scan("a.csv").refusal == (
        "it looks like UTF-16 or binary (a UTF-16 byte-order mark)"
    )


def test_utf16_without_a_bom_is_refused_by_its_nul_bytes(memory_source):
    _put(memory_source, "a.csv", "id,name\n".encode("utf-16-le"))
    assert "NUL byte at offset 1" in _scan("a.csv").refusal


def test_bad_utf8_after_a_utf8_bom_is_refused_not_read_as_cp1252(memory_source):
    """0xE9 is a fine cp1252 byte. Reading the file as cp1252 would glue 'ï»¿'
    onto the first header name, so a BOM commits the file to UTF-8."""
    _put(memory_source, "a.csv", b"\xef\xbb\xbfid,n\xe9\n")
    assert _scan("a.csv").refusal == (
        "it starts with a UTF-8 byte-order mark but is not valid UTF-8"
    )


def test_a_file_larger_than_one_real_chunk_is_scanned_in_pieces(memory_source):
    """Review focus 4, at the real chunk size: 2.5 chunks of ASCII, then one
    cp1252 byte at the very end, so the offset crosses two chunk boundaries."""
    size = int(file_source._SCAN_CHUNK * 2.5)
    _put(memory_source, "big.csv", b"a" * size + b"\xe9")
    assert _scan("big.csv") == file_source._Decision("cp1252")
    _put(memory_source, "bigbad.csv", b"a" * size + b"\x81")
    assert f"offset {size} " in _scan("bigbad.csv").refusal


def test_a_refusal_is_a_source_error_with_a_self_contained_message():
    assert issubclass(file_source.SourceEncodingError, file_source.SourceError)
    message = file_source._refusal_message("x.csv", "it looks like UTF-16 or binary")
    assert message == (
        f"x.csv {_NOT_TEXT}: it looks like UTF-16 or binary. Re-save it as UTF-8."
    )
    assert "SourceEncodingError" not in message
