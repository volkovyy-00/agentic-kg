import fsspec
import pytest

from agentic_kg.common.config import reset_settings
from agentic_kg.common.value_types import classify
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
        "groups.csv", "group_name", "parts.csv", "group_name", FakeToolContext()
    )
    assert result["status"] == "success"
    preview = result["join_preview"]
    assert preview["file_a_total"] == 2
    assert preview["file_a_matched"] == 2
    assert preview["file_a_match_fraction"] == 1.0


def test_join_preview_partial_coverage(join_source):
    """parts.csv references a group that groups.csv does not contain."""
    result = file_tools.join_preview(
        "parts.csv", "group_name", "groups.csv", "group_name", FakeToolContext()
    )
    preview = result["join_preview"]
    assert preview["file_a_total"] == 3
    assert preview["file_a_matched"] == 2
    assert preview["file_a_match_fraction"] < 1.0
    assert preview["file_b_match_fraction"] == 1.0


def test_join_preview_missing_file_returns_error(join_source):
    result = file_tools.join_preview(
        "parts.csv", "group_name", "nope.csv", "group_name", FakeToolContext()
    )
    assert result["status"] == "error"


def test_join_preview_missing_column_returns_error(join_source):
    result = file_tools.join_preview(
        "parts.csv", "nope", "groups.csv", "group_name", FakeToolContext()
    )
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
        "line_items.csv", "part_name", "category", FakeToolContext()
    )
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
        "line_items.csv", "part_name", "assembly_id", FakeToolContext()
    )
    check = result["collapse_check"]
    assert check["group_count"] == 2
    assert check["groups_with_conflicts"] == 1
    assert check["survives_collapse"] is False
    assert check["example_conflicts"] == [{"node_key": "Bolt", "values": ["a1", "a2"]}]


def test_collapse_check_of_the_node_key_itself_survives(collapse_source):
    result = file_tools.collapse_check(
        "line_items.csv", "part_name", "part_name", FakeToolContext()
    )
    assert result["collapse_check"]["survives_collapse"] is True


def test_collapse_check_missing_file_returns_error(memory_source):
    result = file_tools.collapse_check("nope.csv", "id", "name", FakeToolContext())
    assert result["status"] == "error"


def test_collapse_check_missing_column_returns_error(collapse_source):
    result = file_tools.collapse_check(
        "line_items.csv", "part_name", "not_a_column", FakeToolContext()
    )
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
    assert context.state[file_tools.SUGGESTED_FILES] == [
        "people.csv",
        "notes/readme.md",
    ]


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


# --- column type hints ------------------------------------------------------


@pytest.fixture
def bom_source(monkeypatch):
    """Point the source root at the bundled example data.

    tests/conftest.py already defaults SOURCE_URI to ./data/bom, but other
    fixtures in this file override it, so set it explicitly and reset the cached
    settings both ways.
    """
    monkeypatch.setenv("SOURCE_URI", "./data/bom")
    reset_settings()
    yield
    reset_settings()


def test_column_type_hint_reports_a_plain_count_as_integer(bom_source):
    """lead_time_days is bare digits; suggesting float for it would store 8.0
    where the source says 8."""
    context = FakeToolContext()
    result = file_tools.column_type_hint(
        "part_supplier_mapping.csv", "lead_time_days", context
    )

    assert result["status"] == "success"
    hint = result["column_type_hint"]
    assert hint["shape"] == "bare_numeric"
    assert hint["suggested_type"] == "integer"
    assert hint["unconvertible_count"] == 0


def test_suggested_type_ignores_a_single_dirty_value_in_a_bare_numeric_column():
    """Catches _suggested_type downgrading a clean integer column to float
    because ONE non-blank value fails INTEGER coercion, even when that value
    is not a number at all (e.g. 'N/A'). A value that also fails FLOAT
    coercion is dirty data, not evidence of fractional-ness, and must not
    influence integer-vs-float -- otherwise 400 integers plus one 'N/A'
    would be suggested float and store 8.0 where the source says 8."""
    values = [str(n) for n in range(400)] + ["N/A"]
    assert file_tools._suggested_type(file_tools.BARE_NUMERIC, values) == "integer"


def test_column_type_hint_reports_currency_as_float_by_shape(bom_source):
    """Every price in products.csv is a whole dollar amount, so a derivation
    keyed off 'are all values whole' would answer integer -- and then refuse the
    first $99.99 the data ever gains. The shape assertion is the real guard:
    it fails even if a broken derivation happens to return the right label."""
    context = FakeToolContext()
    result = file_tools.column_type_hint("products.csv", "price", context)

    hint = result["column_type_hint"]
    assert hint["shape"] == "numeric_after_cleaning"
    assert hint["suggested_type"] == "float"


def test_column_type_hint_reports_a_yes_no_column_as_boolean(bom_source):
    context = FakeToolContext()
    result = file_tools.column_type_hint(
        "part_supplier_mapping.csv", "preferred_supplier", context
    )

    hint = result["column_type_hint"]
    assert hint["shape"] == "boolean_like"
    assert hint["suggested_type"] == "boolean"


def test_column_type_hint_suggests_nothing_for_text(bom_source):
    """A suggestion for a name column would invite the model to type it, and
    every value would then fail to convert."""
    context = FakeToolContext()
    result = file_tools.column_type_hint("suppliers.csv", "name", context)

    hint = result["column_type_hint"]
    assert hint["shape"] == "text"
    assert hint["suggested_type"] is None


def test_column_type_hint_counts_come_from_the_loader_converter(monkeypatch):
    """The counts must be what the loader will actually do, not a second
    implementation that can disagree with it."""
    fs = fsspec.filesystem("memory")
    fs.store.clear()
    fs.pseudo_dirs.clear()
    try:
        with fs.open("/hints/costs.csv", "w") as handle:
            handle.write('cost,note\n"$1,234.50",a\n42,b\nN/A,c\n,d\n99\n')
        monkeypatch.setenv("SOURCE_URI", "memory://hints")
        reset_settings()

        hint = file_tools.column_type_hint("costs.csv", "cost", FakeToolContext())[
            "column_type_hint"
        ]

        assert hint["convertible_count"] == 3
        assert hint["blank_count"] == 1
        assert hint["unconvertible_count"] == 1
        assert hint["example_unconvertible"] == ["N/A"]

        # The last row stops before 'note'. The loader skips an absent key and
        # leaves any earlier value alone, so counting it as blank would claim
        # the build erases something it does not touch.
        note = file_tools.column_type_hint("costs.csv", "note", FakeToolContext())[
            "column_type_hint"
        ]

        assert note["missing_count"] == 1
        assert note["blank_count"] == 0
    finally:
        fs.store.clear()
        fs.pseudo_dirs.clear()


def test_column_type_hints_returns_one_entry_per_column(bom_source):
    context = FakeToolContext()
    result = file_tools.column_type_hints(
        "part_supplier_mapping.csv", ["lead_time_days", "preferred_supplier"], context
    )

    assert result["status"] == "success"
    hints = result["column_type_hints"]
    assert [hint["column"] for hint in hints] == [
        "lead_time_days",
        "preferred_supplier",
    ]
    assert [hint["suggested_type"] for hint in hints] == ["integer", "boolean"]


def test_suggested_type_does_not_downgrade_an_overflowing_integer_to_float():
    """coerce refuses a value for two different reasons. Reading "integer
    failed, float succeeded" as evidence of a fraction types the column float on
    the strength of a whole number too big for Neo4j's INTEGER -- and float
    stores 9223372036854775809 as 9.223372036854776e+18, a wrong number reported
    as a clean conversion. Integer keeps the outlier on the counted-and-cleared
    path instead, where the hint's own unconvertible_count shows it."""
    values = ["1", "2", "9223372036854775809"]
    assert file_tools._suggested_type(classify(values), values) == "integer"


def test_suggested_type_still_sees_a_genuinely_fractional_column():
    """The overflow fix must not cost the fractional detection it sits next to."""
    assert (
        file_tools._suggested_type(classify(["1.5", "2.5"]), ["1.5", "2.5"]) == "float"
    )


def test_column_type_hints_reads_the_source_once_for_all_columns(
    bom_source, monkeypatch
):
    """Source files are read through fsspec and may be remote, so one read per
    requested column turns a hint request for N properties into N downloads and
    N parses of the same file. The schema prompt directs the agent to inspect
    every property, so N is not small."""
    reads = []
    real_read_csv_batches = file_tools.read_csv_batches

    def counting_read(path, *args, **kwargs):
        reads.append(path)
        return real_read_csv_batches(path, *args, **kwargs)

    monkeypatch.setattr(file_tools, "read_csv_batches", counting_read)

    context = FakeToolContext()
    result = file_tools.column_type_hints(
        "part_supplier_mapping.csv",
        ["lead_time_days", "preferred_supplier", "unit_cost"],
        context,
    )

    assert result["status"] == "success"
    assert len(result["column_type_hints"]) == 3
    assert len(reads) == 1


def test_column_type_hint_reads_a_header_only_file(monkeypatch):
    """read_csv_batches yields nothing when a file has a header and no data
    rows, because it only yields once it has collected a batch. Inferring "no
    header" from "no batches" rejected a valid header-only file -- an empty
    export, or a table whose rows have not landed yet -- with an error about a
    header sitting right there in the file, and blocked analysis of columns that
    are present."""
    fs = fsspec.filesystem("memory")
    fs.store.clear()
    fs.pseudo_dirs.clear()
    try:
        with fs.open("/empty/parts.csv", "w") as handle:
            handle.write("part_id,quantity\n")
        monkeypatch.setenv("SOURCE_URI", "memory://empty")
        reset_settings()

        result = file_tools.column_type_hint("parts.csv", "quantity", FakeToolContext())
        assert result["status"] == "success"
        hint = result["column_type_hint"]
        assert hint["convertible_count"] == 0
        assert hint["blank_count"] == 0
        assert hint["missing_count"] == 0

        batched = file_tools.column_type_hints(
            "parts.csv", ["part_id", "quantity"], FakeToolContext()
        )
        assert batched["status"] == "success"
        assert len(batched["column_type_hints"]) == 2

        # A misspelling in a header-only file must still be named, not silently
        # answered with zero counts.
        missing = file_tools.column_type_hint("parts.csv", "qty", FakeToolContext())
        assert missing["status"] == "error"
        assert "qty" in missing["error_message"]
    finally:
        fs.store.clear()
        fs.pseudo_dirs.clear()


def test_column_type_hints_rejects_a_bare_string_instead_of_spelling_it(bom_source):
    """A model sending "price" where a list belongs would otherwise have it
    iterated into ['p','r','i','c','e'] and get "Column 'p' is not in
    products.csv" -- an answer about the data for what is a call-shape mistake,
    sending the model to inspect a file that is perfectly fine. The error must
    name the real problem and the single-column tool to use instead."""
    context = FakeToolContext()
    result = file_tools.column_type_hints("products.csv", "price", context)

    assert result["status"] == "error"
    assert "list" in result["error_message"]
    assert "column_type_hint" in result["error_message"]
    assert "'p'" not in result["error_message"]


def test_column_type_hints_names_a_missing_column_before_reading_rows(bom_source):
    """Batched reading must not cost the caller the error message the
    per-column path gave: the failure still names the column to correct."""
    context = FakeToolContext()
    result = file_tools.column_type_hints(
        "part_supplier_mapping.csv", ["lead_time_days", "leed_time"], context
    )

    assert result["status"] == "error"
    assert "leed_time" in result["error_message"]


def test_column_type_hint_errors_on_a_missing_column(bom_source):
    """A misspelled column must not come back as 'text, no suggestion' -- that
    reads as an answer about the data rather than a typo."""
    context = FakeToolContext()
    result = file_tools.column_type_hint("products.csv", "prcie", context)

    assert result["status"] == "error"
    assert "prcie" in result["error_message"]


def test_the_hint_docstring_documents_every_key_it_returns(bom_source):
    """This docstring is the tool description ADK sends to the model, so a key
    the payload carries but the description never names is a number the model
    has to guess the meaning of.

    Cannot catch a description that is merely wrong -- 'blank_count' was
    documented as always cleared when that is true only for a typed property --
    but it does catch the half a test can check: a count added to the payload
    and never explained."""
    hint = file_tools.column_type_hint("products.csv", "price", FakeToolContext())[
        "column_type_hint"
    ]

    for key in hint:
        assert f"'{key}'" in file_tools.column_type_hint.__doc__, key


def test_group_values_by_key_returns_every_group_not_only_conflicts():
    """Catches an extraction that returns the filtered conflict list: collapse_check
    needs group_count, which is len(groups) BEFORE filtering."""
    groups = file_tools.group_values_by_key([("k1", "a"), ("k1", "b"), ("k2", "c")])
    assert set(groups) == {"k1", "k2"}
    assert groups["k1"] == {"a", "b"}
    assert groups["k2"] == {"c"}


def test_group_values_by_key_normalises_none_to_empty_string():
    """Catches an extraction that drops collapse_check's None handling, which would
    make a ragged row's absent key crash on set membership."""
    groups = file_tools.group_values_by_key([(None, None)])
    assert groups == {"": {""}}


def test_the_column_readers_are_importable_under_their_public_names(memory_source):
    """Catches a promotion that renamed only the definition and left call sites (or
    vice versa) — reference_reachability imports these by their public names."""
    values, error = file_tools.collect_column_values("people.csv", "name")
    assert error is None and values == ["Ada", "Grace"]
    pairs, error = file_tools.collect_column_pairs("people.csv", "id", "name")
    assert error is None and pairs == [("1", "Ada"), ("2", "Grace")]
