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


@pytest.fixture
def collapse_source(memory_source):
    """One row per line-item: 'part_name' is the real node key and repeats,
    'assembly_id' is unique per row, 'category' is constant per part."""
    fs = memory_source
    with fs.open("/src/line_items.csv", "w") as handle:
        handle.write(
            "part_name,assembly_id,category\n"
            "Bolt,a1,fastener\n"
            "Bolt,a2,fastener\n"
            "Nut,a3,fastener\n"
        )
    return fs


def test_collapse_check_reports_a_column_that_survives(collapse_source):
    """'category' has one distinct value per part, so it survives the MERGE."""
    result = file_tools.collapse_check(
        "line_items.csv", "part_name", "category", FakeToolContext())
    assert result["status"] == "success"
    check = result["collapse_check"]
    assert check["row_count"] == 3
    assert check["group_count"] == 2
    assert check["groups_with_conflicts"] == 0
    assert check["survives_collapse"] is True
    assert check["example_conflicts"] == []


def test_collapse_check_flags_a_per_row_id_that_collapses(collapse_source):
    """'assembly_id' is unique per row -- exactly what column_stats calls a
    great key -- but two of its values collide onto the 'Bolt' node."""
    result = file_tools.collapse_check(
        "line_items.csv", "part_name", "assembly_id", FakeToolContext())
    check = result["collapse_check"]
    assert check["group_count"] == 2
    assert check["groups_with_conflicts"] == 1
    assert check["survives_collapse"] is False
    assert check["example_conflicts"] == [{"node_key": "Bolt", "values": ["a1", "a2"]}]


def test_collapse_check_of_the_node_key_itself_survives(collapse_source):
    result = file_tools.collapse_check(
        "line_items.csv", "part_name", "part_name", FakeToolContext())
    assert result["collapse_check"]["survives_collapse"] is True


def test_collapse_check_missing_file_returns_error(memory_source):
    result = file_tools.collapse_check("nope.csv", "id", "name", FakeToolContext())
    assert result["status"] == "error"


def test_collapse_check_missing_column_returns_error(collapse_source):
    result = file_tools.collapse_check(
        "line_items.csv", "part_name", "not_a_column", FakeToolContext())
    assert result["status"] == "error"
    assert "not_a_column" in result["error_message"]


# Suggested files must exist at the source

def test_suggesting_a_file_that_is_not_there_is_refused(memory_source):
    """The list is LLM-chosen, so a plausible name that does not exist would
    otherwise be stored, approved, and only fail an agent hop later."""
    context = FakeToolContext()
    result = file_tools.set_suggested_files(["people.csv", "ghost.csv"], context)
    assert result["status"] == "error"
    assert "ghost.csv" in result["error_message"]
    assert file_tools.SUGGESTED_FILES not in context.state


def test_suggesting_no_files_is_refused(memory_source):
    context = FakeToolContext()
    result = file_tools.set_suggested_files([], context)
    assert result["status"] == "error"
    assert file_tools.SUGGESTED_FILES not in context.state


def test_suggesting_real_files_still_works(memory_source):
    context = FakeToolContext()
    result = file_tools.set_suggested_files(["people.csv", "notes/readme.md"], context)
    assert result["status"] == "success"
    assert context.state[file_tools.SUGGESTED_FILES] == ["people.csv", "notes/readme.md"]


def test_a_leading_dot_slash_is_accepted_and_normalised(memory_source):
    """source_exists() treats "./people.csv" and "people.csv" as the same file,
    so refusing it here would make two stages disagree. What is stored must
    match what list_import_files returns."""
    context = FakeToolContext()
    result = file_tools.set_suggested_files(["./people.csv"], context)
    assert result["status"] == "success"
    assert context.state[file_tools.SUGGESTED_FILES] == ["people.csv"]


def test_a_non_string_entry_is_reported_not_raised(memory_source):
    """An ADK tool must return a tool_error rather than raise into the runner."""
    context = FakeToolContext()
    result = file_tools.set_suggested_files([None], context)
    assert result["status"] == "error"
    assert file_tools.SUGGESTED_FILES not in context.state


def test_a_bare_string_is_not_treated_as_a_list_of_characters(memory_source):
    context = FakeToolContext()
    result = file_tools.set_suggested_files("people.csv", context)
    assert result["status"] == "error"
    assert file_tools.SUGGESTED_FILES not in context.state
