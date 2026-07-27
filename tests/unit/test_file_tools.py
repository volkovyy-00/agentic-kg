import fsspec
import pytest

from agentic_kg.common.config import reset_settings
from agentic_kg.tools import file_tools


class FakeToolContext:
    """Minimal stand-in for ADK's ToolContext — the file tools only use .state."""

    def __init__(self):
        self.state = {}


@pytest.fixture
def memory_source(monkeypatch):
    fs = fsspec.filesystem("memory")
    fs.store.clear()
    fs.pseudo_dirs.clear()
    with fs.open("/src/people.csv", "w") as handle:
        handle.write("id,name\n1,Ada\n2,Grace\n")
    with fs.open("/src/notes/readme.md", "w") as handle:
        handle.write("# Title\nAda appears here.\n")
    monkeypatch.setenv("SOURCE_URI", "memory://src")
    reset_settings()
    yield fs
    fs.store.clear()
    fs.pseudo_dirs.clear()


def test_list_import_files_returns_relative_names(memory_source):
    context = FakeToolContext()
    result = file_tools.list_import_files(context)
    assert result["status"] == "success"
    assert result["all_available_files"] == ["notes/readme.md", "people.csv"]


def test_sample_file_reads_content(memory_source):
    context = FakeToolContext()
    result = file_tools.sample_file("people.csv", context)
    assert result["status"] == "success"
    assert "Ada" in result["sample"]["content"]


def test_sample_file_missing_returns_error(memory_source):
    context = FakeToolContext()
    result = file_tools.sample_file("nope.csv", context)
    assert result["status"] == "error"


def test_search_file_finds_matching_line(memory_source):
    result = file_tools.search_file("notes/readme.md", "ada")
    assert result["status"] == "success"
    assert result["search_results"]["metadata"]["lines_found"] == 1


def test_approve_suggested_files_returns_a_result(memory_source):
    context = FakeToolContext()
    file_tools.set_suggested_files(["people.csv"], context)
    result = file_tools.approve_suggested_files(context)
    assert result is not None
    assert result["status"] == "success"
    assert context.state["approved_file_list"] == ["people.csv"]


def test_approve_without_suggestions_returns_error(memory_source):
    context = FakeToolContext()
    result = file_tools.approve_suggested_files(context)
    assert result["status"] == "error"


def test_unset_source_uri_surfaces_as_tool_error(monkeypatch):
    monkeypatch.delenv("SOURCE_URI", raising=False)
    reset_settings()
    result = file_tools.list_import_files(FakeToolContext())
    assert result["status"] == "error"
    assert "SOURCE_URI" in result["error_message"]


@pytest.fixture
def join_source(memory_source):
    """Two extra CSVs: parts reference groups, only some of which exist."""
    fs = memory_source
    with fs.open("/src/groups.csv", "w") as handle:
        handle.write("group_name,row_id\nAlpha,1\nAlpha,2\nBeta,3\n")
    with fs.open("/src/parts.csv", "w") as handle:
        handle.write("part_id,group_name\np1,Alpha\np2,Beta\np3,Gamma\n")
    return fs


def test_column_stats_reports_a_unique_identifier(join_source):
    result = file_tools.column_stats("parts.csv", "part_id", FakeToolContext())
    assert result["status"] == "success"
    stats = result["column_stats"]
    assert stats["row_count"] == 3
    assert stats["distinct_count"] == 3
    assert stats["is_unique"] is True


def test_column_stats_reports_duplicates(join_source):
    result = file_tools.column_stats("groups.csv", "group_name", FakeToolContext())
    stats = result["column_stats"]
    assert stats["row_count"] == 3
    assert stats["distinct_count"] == 2
    assert stats["is_unique"] is False


def test_column_stats_missing_file_returns_error(memory_source):
    result = file_tools.column_stats("nope.csv", "id", FakeToolContext())
    assert result["status"] == "error"


def test_column_stats_missing_column_returns_error(join_source):
    result = file_tools.column_stats("parts.csv", "not_a_column", FakeToolContext())
    assert result["status"] == "error"
    assert "not_a_column" in result["error_message"]


def test_join_preview_full_coverage(join_source):
    result = file_tools.join_preview(
        "groups.csv", "group_name", "parts.csv", "group_name", FakeToolContext())
    assert result["status"] == "success"
    preview = result["join_preview"]
    assert preview["file_a_total"] == 2
    assert preview["file_a_matched"] == 2
    assert preview["file_a_match_fraction"] == 1.0


def test_join_preview_partial_coverage(join_source):
    """parts.csv references a group that groups.csv does not contain."""
    result = file_tools.join_preview(
        "parts.csv", "group_name", "groups.csv", "group_name", FakeToolContext())
    preview = result["join_preview"]
    assert preview["file_a_total"] == 3
    assert preview["file_a_matched"] == 2
    assert preview["file_a_match_fraction"] < 1.0
    assert preview["file_b_match_fraction"] == 1.0


def test_join_preview_missing_file_returns_error(join_source):
    result = file_tools.join_preview(
        "parts.csv", "group_name", "nope.csv", "group_name", FakeToolContext())
    assert result["status"] == "error"


def test_join_preview_missing_column_returns_error(join_source):
    result = file_tools.join_preview(
        "parts.csv", "nope", "groups.csv", "group_name", FakeToolContext())
    assert result["status"] == "error"
