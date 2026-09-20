import logging
from bisect import insort
from itertools import chain, islice
from typing import Any, Dict, List, NamedTuple, Optional

from google.adk.tools import ToolContext

from agentic_kg.common.csv_reader import (
    make_csv_reader,
    read_csv_batches,
    read_csv_header,
)
from agentic_kg.common.file_source import (
    SourceError,
    get_source_root,
    list_source_files,
    open_source,
    source_exists,
)
from agentic_kg.common.tool_result import tool_error, tool_success
from agentic_kg.common.value_types import (
    BARE_NUMERIC,
    BLANK,
    BOOLEAN,
    BOOLEAN_LIKE,
    CONVERTED,
    FLOAT,
    INTEGER,
    NUMERIC_AFTER_CLEANING,
    classify,
    coerce,
    has_fractional_part,
    is_blank,
)

logger = logging.getLogger(__name__)

ALL_AVAILABLE_FILES = "all_available_files"
SUGGESTED_FILES = "suggested_file_list"
APPROVED_FILES = "approved_file_list"


def list_import_files(tool_context: ToolContext) -> dict:
    """Lists files available for knowledge graph construction.

    All names are relative to the configured source location.

    Returns:
        dict: 'status' of 'success' or 'error'. On success, an
              'all_available_files' key with a list of relative file names.
    """
    try:
        file_names = list_source_files()
    except SourceError as exc:
        return tool_error(str(exc))

    tool_context.state[ALL_AVAILABLE_FILES] = file_names
    return tool_success(ALL_AVAILABLE_FILES, file_names)


def set_suggested_files(
    suggest_files: List[str], tool_context: ToolContext
) -> Dict[str, Any]:
    """Set the files to be used for data import.

    Args:
        suggest_files: a list of file names, exactly as 'list_import_files' returned them

    Returns:
        dict: 'status' of 'success' or 'error'. On success, a 'suggested_file_list'
              key with the names that were set. On error, an 'error_message' key.
    """
    # The list is chosen by a model, so a plausible name the source does not
    # hold would otherwise be stored, approved, and fail only when the schema
    # agent tried to read it -- an agent hop away from the mistake.
    if not isinstance(suggest_files, list) or not suggest_files:
        return tool_error(
            "No files were suggested. Call 'list_import_files' and pass a list of "
            "the names it returns."
        )

    if not all(isinstance(name, str) for name in suggest_files):
        return tool_error(
            "Every suggested file must be a name returned by 'list_import_files'."
        )

    try:
        available = set(list_source_files())
    except SourceError as exc:
        return tool_error(str(exc))

    # "./name.csv" identifies the same file as "name.csv" everywhere else in
    # the codebase, so accept it here too rather than refusing a name the rest
    # of the tools would have read.
    suggest_files = [name.removeprefix("./") for name in suggest_files]
    unknown = [name for name in suggest_files if name not in available]
    if unknown:
        return tool_error(
            f"These are not files at the source location: {unknown}. "
            "Call 'list_import_files' and choose from the names it returns."
        )

    tool_context.state[SUGGESTED_FILES] = suggest_files
    return tool_success(SUGGESTED_FILES, suggest_files)


def get_suggested_files(tool_context: ToolContext) -> Dict[str, Any]:
    """Get the suggested files to be used for import.

    Returns:
        dict: A dictionary containing success or failure information.
              Includes a 'status' key ('success' or 'error').
              If 'success', includes a 'suggested_files' key with list of files.
              If 'error', includes an 'error_message' key.

    """
    if SUGGESTED_FILES not in tool_context.state:
        return tool_error(
            "Suggested files have not been set. Take no action other than to inform user."
        )
    return tool_success(SUGGESTED_FILES, tool_context.state[SUGGESTED_FILES])


def get_source_location(tool_context: ToolContext) -> Dict[str, Any]:
    """Reports where the system is reading source files from."""
    try:
        return tool_success("source_location", get_source_root())
    except SourceError as exc:
        return tool_error(str(exc))


def approve_suggested_files(tool_context: ToolContext) -> Dict[str, Any]:
    """Approves the suggested files for further processing."""
    if SUGGESTED_FILES not in tool_context.state:
        return tool_error(
            "Current files have not been set. Take no action other than to inform user."
        )

    tool_context.state[APPROVED_FILES] = tool_context.state[SUGGESTED_FILES]
    return tool_success(APPROVED_FILES, tool_context.state[APPROVED_FILES])


def get_approved_files(tool_context: ToolContext) -> Dict[str, Any]:
    """Get the files that have been approved for importing into a knowledge graph."""

    if APPROVED_FILES not in tool_context.state:
        return tool_error("Approved files have not been set.")

    return tool_success(APPROVED_FILES, tool_context.state[APPROVED_FILES])


def sample_file(file_path: str, tool_context: ToolContext) -> dict:
    """Samples a file by reading up to 100 lines as text.

    Args:
      file_path: file to sample, relative to the source location
      tool_context: ToolContext object

    Returns:
        dict: 'status' of 'success' or 'error'. On success, a 'sample' key with
              metadata and content.
    """
    suffix = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
    mimetype = {"csv": "text/csv", "md": "text/markdown"}.get(suffix, "text/plain")

    result = {
        "metadata": {"path": file_path, "mimetype": mimetype},
        "annotations": [],
    }

    try:
        with open_source(file_path, "r") as handle:
            result["content"] = "".join(islice(handle, 100))
    except SourceError as exc:
        return tool_error(str(exc))
    except FileNotFoundError:
        return tool_error(f"Path does not exist: {file_path}")
    except Exception as exc:  # noqa: BLE001 - report decoding failures to the agent
        return tool_error(f"Error reading or processing file {file_path}: {exc}")

    return tool_success("sample", result)


def search_csv_file(
    file_path: str, query: str, tool_context: ToolContext, case_sensitive: bool = False
) -> dict:
    """
    Searches a CSV file for rows containing the given query string in any of its fields.

    Args:
      file_path: Path to the CSV file, relative to the source location.
      query: The string to search for.
      tool_context: The ToolContext object.
      case_sensitive: Whether the search should be case-sensitive (default: False).

    Returns:
        dict: A dictionary with 'status' ('success' or 'error').
              If 'success', includes 'search_results' containing 'matching_rows'
              (a list of rows, where each row is a list of strings)
              and 'metadata' (path, mimetype, query, case_sensitive, rows_found).
              If 'error', includes an 'error_message'.
    """
    try:
        if not source_exists(file_path):
            return tool_error(f"CSV file does not exist: {file_path}")
    except SourceError as exc:
        return tool_error(str(exc))

    matching_rows = []
    search_query = query if case_sensitive else query.lower()
    header_row = []

    try:
        # Handle empty query - return no results
        if not query:
            with open_source(file_path, "r") as csvfile:
                reader = make_csv_reader(csvfile, file_path)
                header_row = next(reader, [])
                # Empty query returns no matches, but we still read the header
        else:
            with open_source(file_path, "r") as csvfile:
                reader = make_csv_reader(csvfile, file_path)

                header_row = next(
                    reader, []
                )  # Store header, or empty list if file is empty

                for row in reader:
                    for field in row:
                        field_to_check = (
                            str(field) if case_sensitive else str(field).lower()
                        )
                        if search_query in field_to_check:
                            matching_rows.append(row)
                            break  # Move to next row once a match is found
    except Exception as e:
        return tool_error(f"Error reading or searching CSV file {file_path}: {e}")

    result_data = {
        "metadata": {
            "path": file_path,
            "mimetype": "text/csv",
            "query": query,
            "case_sensitive": case_sensitive,
            "header": header_row,
            "rows_found": len(matching_rows),
        },
        "matching_rows": matching_rows,
    }
    return tool_success("search_results", result_data)


def _missing_column_error(file_path: str, column: str, header: List[str]) -> dict:
    """One wording for a misspelled column, wherever it is noticed."""
    return tool_error(
        f"Column '{column}' is not in {file_path}. Available columns: {header}"
    )


def _column_rows(file_path: str, columns: List[str]):
    """Stream one tuple per data row, holding only the named columns.

    Returns:
        (rows, error). On success `rows` yields a tuple per data row carrying
        each column's raw value: None when the row was too short to reach that
        column, "" for a present-but-empty cell. Folding those together is the
        caller's decision, not this reader's -- the loader treats them
        differently and the hint tools count them apart.

    Column names are validated ONCE, against the first batch's header or, when
    there is no batch at all, against a header read on its own. That second read
    is what a header-only file costs, and it is the loader's `_batches_and_header`
    rule: read_csv_batches yields nothing for a valid empty export, so a check
    living inside the batch loop never runs for one.

    A failure part way through the stream raises out of the returned iterator.
    Consume it inside a try -- `summarize_column` and `summarize_key_groups` do,
    which is why they, and not their callers, own the whole read.
    """
    try:
        if not source_exists(file_path):
            return iter(()), tool_error(f"CSV file does not exist: {file_path}")
    except SourceError as exc:
        return iter(()), tool_error(str(exc))

    try:
        batches = read_csv_batches(file_path)
        first = next(batches, None)
        if first is None:
            header = read_csv_header(file_path)
            batches = iter(())
        else:
            header = first[0]
            batches = chain([first], batches)
    except Exception as exc:  # noqa: BLE001 - report read failures to the agent
        return iter(()), tool_error(f"Error reading CSV file {file_path}: {exc}")

    if not header:
        return iter(()), tool_error(f"CSV file has no header row: {file_path}")
    missing = [column for column in columns if column not in header]
    if missing:
        if len(columns) == 1:
            return iter(()), _missing_column_error(file_path, columns[0], header)
        return iter(()), tool_error(
            f"Column(s) {missing} are not in {file_path}. Available columns: {header}"
        )

    def rows():
        for _batch_header, batch in batches:
            for row in batch:
                yield tuple(row.get(column) for column in columns)

    return rows(), None


EXAMPLE_VALUE_LIMIT = 10
"""collapse_check reports sorted(values)[:10], so only the ten smallest are ever
read. Holding the full set instead is what makes a per-row-unique candidate under
a repeating key cost O(rows) -- the one shape the tool exists to catch."""


class ColumnSummary(NamedTuple):
    """What every caller needs to know about one column's values."""

    row_count: int
    empty_count: int
    distinct: set[str]

    @property
    def is_unique(self) -> bool:
        """THE definition of a per-row-unique column, for every caller.

        Both column_stats and reference_reachability ask this question, and they
        used to answer it in two places. Note that zero rows is unique here (no
        empties, and zero distinct equals zero rows); _home_files adds its own
        at-least-one-row condition on top, because a zero-row file is not an
        identifier's home even though its column is vacuously unique.
        """
        return self.empty_count == 0 and len(self.distinct) == self.row_count


class KeyGroupSummary(NamedTuple):
    """What every caller needs to know about values grouped under a node key."""

    row_count: int
    group_count: int
    conflict_count: int
    examples: list[dict]
    values_on_one_key: bool | None
    """None when track_value_owners was False -- never guess it from the rest."""


def summarize_column(file_path: str, column: str):
    """Read one column, keeping only its distinct non-blank values.

    Returns (summary, error). This owns the whole read: a failure part way
    through comes back as an error result, never as a raise. Callers depend on
    that -- find_plan_problems does not catch, so a raise there makes a plan
    unshowable.

    The distinct SET, not just a count, because two callers need the values
    themselves: join_preview intersects two sides, and _home_files returns a
    per-file value set that later comparisons read instead of filtering again.
    """
    rows, error = _column_rows(file_path, [column])
    if error is not None:
        return None, error

    row_count = 0
    empty_count = 0
    distinct: set[str] = set()
    try:
        for (value,) in rows:
            row_count += 1
            if is_blank(value):
                empty_count += 1
            else:
                distinct.add(str(value))
    except Exception as exc:  # noqa: BLE001 - report read failures to the agent
        return None, tool_error(f"Error reading CSV file {file_path}: {exc}")

    return ColumnSummary(row_count, empty_count, distinct), None


class _KeyState:
    """One key's running state: enough to answer, never the rows themselves."""

    __slots__ = ("first_value", "rank", "conflicted", "examples")

    def __init__(self, first_value: str, rank: int):
        self.first_value = first_value
        self.rank = rank
        self.conflicted = False
        self.examples: list[str] | None = None
        """A sorted list of the ten smallest distinct values, or None when this
        key is not one of the reported few (or never was, or was displaced)."""


def _keep_smallest(values: list[str], value: str) -> None:
    """Insert into a sorted list capped at the ten smallest distinct values."""
    if value in values:
        return
    if len(values) < EXAMPLE_VALUE_LIMIT:
        insort(values, value)
    elif value < values[-1]:
        insort(values, value)
        values.pop()


class _KeyGroupAccumulator:
    """The running state of one pass over a key/value column pair.

    A class rather than one long loop body because the pass answers two
    independent questions at once -- which keys collapse, and whether any value
    belongs to more than one key -- and they share nothing but the row. Folding
    each row is still a single pass; only the reading of it is split.
    """

    def __init__(self, keep_examples: int, track_value_owners: bool):
        self.keep_examples = keep_examples
        self.states: dict[str, _KeyState] = {}
        self.held: list[str] = []  # keys currently holding example values
        self.owners: dict[str, str] = {}  # value -> its one key, while that holds
        self.row_count = 0
        self.conflict_count = 0
        self.values_on_one_key = True if track_value_owners else None

    def add(self, key, value) -> None:
        """Fold one row in, holding nothing that scales with the rows seen."""
        self.row_count += 1
        key_text = "" if key is None else str(key)
        value_text = "" if value is None else str(value)
        self._group(key_text, value_text)
        if self.values_on_one_key and not is_blank(value):
            self._own(key_text, value_text)

    def _group(self, key_text: str, value_text: str) -> None:
        """Track this key's first value, its rank, and whether it conflicts."""
        state = self.states.get(key_text)
        if state is None:
            self.states[key_text] = _KeyState(value_text, len(self.states))
        elif not state.conflicted and value_text != state.first_value:
            state.conflicted = True
            self.conflict_count += 1
            self._admit(key_text, state, value_text)
        elif state.examples is not None:
            # Not a fresh conflict: the key is already conflicted and still
            # held, or this value just repeats its first_value.
            _keep_smallest(state.examples, value_text)

    def _own(self, key_text: str, value_text: str) -> None:
        """Note this value's owning key, until a second key claims it."""
        owner = self.owners.setdefault(value_text, key_text)
        if owner != key_text:
            self.values_on_one_key = False
            self.owners.clear()  # the answer cannot change back

    def _admit(self, key: str, state: _KeyState, value: str) -> None:
        """Give this newly-conflicting key an example list, if it earns one."""
        if self.keep_examples <= 0:
            return
        if len(self.held) >= self.keep_examples:
            latest = max(self.held, key=lambda other: self.states[other].rank)
            if self.states[latest].rank < state.rank:
                return  # every held key appears earlier: the newcomer loses
            self.states[latest].examples = None
            self.held.remove(latest)
        state.examples = []
        _keep_smallest(state.examples, state.first_value)
        _keep_smallest(state.examples, value)
        self.held.append(key)

    def examples(self) -> List[dict]:
        """The held keys, ordered by where they first appear in the file."""
        reported = []
        for key in sorted(self.held, key=lambda other: self.states[other].rank):
            kept = self.states[key].examples
            # A held key always has an example list -- `held` is only ever
            # appended to alongside setting it. Assert rather than `or []`: a
            # None here means that invariant broke, and an empty conflict list
            # would hide it.
            assert kept is not None
            reported.append({"node_key": key, "values": list(kept)})
        return reported


def summarize_key_groups(
    file_path: str,
    key_column: str,
    value_column: str,
    *,
    keep_examples: int = 0,
    track_value_owners: bool = False,
):
    """Group a column's values under a node key, as MERGE would collapse them.

    Returns (summary, error), and owns the whole read for the same reason
    summarize_column does.

    Memory is one small state per DISTINCT KEY -- a first value, a rank and a
    flag -- plus the ten smallest values for at most `keep_examples` keys. It is
    deliberately not one state per distinct value: the shape this tool exists to
    catch is a candidate unique per row under a repeating key, where the two are
    the same number as the row count.

    WHICH keys are reported is not "the first few that conflict". collapse_check
    reports the conflicting keys that appear EARLIEST IN THE FILE, which is a
    different set: a key seen on row 1 may only start disagreeing on the last
    row. So a newly-conflicting key COMPETES with the ones already held, and the
    earliest-appearing survive; a key that loses can never return, because the
    highest rank among the held only ever decreases.

    `track_value_owners` is opt-in because answering it costs one entry per
    distinct VALUE, which is exactly the cost collapse_check must not pay.
    _property_failure needs it; collapse_check does not.
    """
    rows, error = _column_rows(file_path, [key_column, value_column])
    if error is not None:
        return None, error

    accumulator = _KeyGroupAccumulator(keep_examples, track_value_owners)
    try:
        for key, value in rows:
            accumulator.add(key, value)
    except Exception as exc:  # noqa: BLE001 - report read failures to the agent
        return None, tool_error(f"Error reading CSV file {file_path}: {exc}")

    return (
        KeyGroupSummary(
            row_count=accumulator.row_count,
            group_count=len(accumulator.states),
            conflict_count=accumulator.conflict_count,
            examples=accumulator.examples(),
            values_on_one_key=accumulator.values_on_one_key,
        ),
        None,
    )


def collect_column_values(file_path: str, column: str):
    """Read every value of one column from a source CSV.

    Returns:
        (values, error) where values holds one entry per data row and error is a
        tool_error dict when the file or column cannot be read.

    read_csv_batches omits the key entirely for a row shorter than the header, so
    a ragged row contributes None here while a present-but-empty cell contributes
    "". The two are the same absence to column_stats, whose row_count and
    empty_count treat both as empty -- but not to the loader, which skips an
    absent key and clears a blank one, so the hint tool counts them apart.

    Public because `column_type_hint` reads one column through it. The hint tools
    count an absent key apart from a blank cell, so they need the per-row list
    this returns rather than either summariser's reduction; giving them the
    streaming reader is their own ticket.
    """
    try:
        if not source_exists(file_path):
            return None, tool_error(f"CSV file does not exist: {file_path}")
    except SourceError as exc:
        return None, tool_error(str(exc))

    values: List[Optional[str]] = []
    header: List[str] = []
    saw_header = False
    try:
        for batch_header, rows in read_csv_batches(file_path):
            if not saw_header:
                header = batch_header
                saw_header = True
                if column not in header:
                    return None, _missing_column_error(file_path, column, header)
            for row in rows:
                values.append(row.get(column))
        if not saw_header:
            # No batches at all: either a header-only file or an empty one.
            header = read_csv_header(file_path)
            if not header:
                return None, tool_error(f"CSV file has no header row: {file_path}")
            if column not in header:
                return None, _missing_column_error(file_path, column, header)
    except Exception as exc:  # noqa: BLE001 - report read failures to the agent
        return None, tool_error(f"Error reading CSV file {file_path}: {exc}")

    return values, None


def column_stats(file_path: str, column: str, tool_context: ToolContext) -> dict:
    """Reports how unique the values of one CSV column are.

    Use this to decide whether a column can serve as a node's unique
    identifier, and to detect per-row columns that would be collapsed (and
    silently overwritten) when rows are merged into a single node.

    Empty values are counted as rows but are not treated as usable identifier
    values: 'is_unique' is only true when every row has a non-empty value and
    all of those values are distinct.

    Args:
      file_path: Path to the CSV file, relative to the source location.
      column: The column to analyze.
      tool_context: The ToolContext object.

    Returns:
        dict: 'status' of 'success' or 'error'. On success, a 'column_stats'
              key with 'path', 'column', 'row_count', 'distinct_count',
              'empty_count' and 'is_unique'.
    """
    summary, error = summarize_column(file_path, column)
    if error is not None:
        return error
    assert summary is not None

    return tool_success(
        "column_stats",
        {
            "path": file_path,
            "column": column,
            "row_count": summary.row_count,
            "distinct_count": len(summary.distinct),
            "empty_count": summary.empty_count,
            "is_unique": summary.is_unique,
        },
    )


def _suggested_type(shape: str, values) -> str | None:
    """Map a column's shape to the type to suggest for it.

    The shape decides, and for a column needing cleaning it decides alone: every
    price in the bundled products.csv is a round dollar amount, so a
    whole-number test would suggest integer for a currency column and then
    refuse the first fractional price the data ever gains. Needing a currency
    symbol or thousands separator stripped is itself the evidence it is money.

    Wholeness is consulted only to split integer from float WITHIN the
    bare_numeric shape, where there is nothing else to go on -- and only for
    values that are numbers at all, so one "N/A" in a column of 400 integers
    cannot make it look fractional.

    It asks has_fractional_part rather than inferring the fraction from a failed
    integer coercion. coerce refuses a value for two different reasons, and
    treating them alike types a column float on the strength of a whole number
    too large for Neo4j's INTEGER -- which then loads as a rounded, wrong number
    and reports a clean conversion. An overflowing value leaves the suggestion
    at integer, so the loader counts it, reports it as an example, and clears
    it; one unstorable outlier does not cost the whole column its type, the same
    tolerance classify() already applies.
    """
    if shape == BOOLEAN_LIKE:
        return BOOLEAN
    if shape == NUMERIC_AFTER_CLEANING:
        return FLOAT
    if shape == BARE_NUMERIC:
        for value in values:
            if has_fractional_part(value):
                return FLOAT
        return INTEGER
    return None


def _hint_from_values(file_path: str, column: str, values: List[Optional[str]]) -> dict:
    """Build one column_type_hint payload from values already read.

    Split out from column_type_hint so column_type_hints can reuse it after a
    single file pass, rather than re-reading the source once per column.
    """
    shape = classify(values)
    suggested = _suggested_type(shape, values)

    convertible_count = 0
    blank_count = 0
    missing_count = 0
    unconvertible_count = 0
    examples: List[str] = []

    for value in values:
        # A row too short to reach this column is not the same as a row with an
        # empty cell. An absent key is skipped whatever the property's type,
        # leaving an earlier row's value alone; an empty cell is a value the row
        # actually carried, and the loader either clears the property (typed) or
        # stores "" (text). Reported together, blank_count would overstate what
        # the build will erase -- and this tool's counts exist precisely to say
        # what the build will do.
        if value is None:
            missing_count += 1
            continue
        if suggested is None:
            if is_blank(value):
                blank_count += 1
            continue
        _converted, outcome = coerce(value, suggested)
        if outcome == CONVERTED:
            convertible_count += 1
        elif outcome == BLANK:
            blank_count += 1
        else:
            unconvertible_count += 1
            if len(examples) < 3:
                examples.append(value)

    return {
        "path": file_path,
        "column": column,
        "shape": shape,
        "suggested_type": suggested,
        "convertible_count": convertible_count,
        "blank_count": blank_count,
        "missing_count": missing_count,
        "unconvertible_count": unconvertible_count,
        "example_unconvertible": examples,
    }


def _collect_columns_values(file_path: str, columns: List[str]):
    """Read every value of several columns in ONE pass over a source CSV.

    Returns:
        (values_by_column, error) where values_by_column maps each requested
        column to one entry per data row, and error is a tool_error dict when
        the file or any requested column cannot be read.

    Source files are read through fsspec and may be remote, so reading once per
    requested column turns a hint request for N properties into N downloads and
    N parses of the same file. Columns are validated against the header before
    any row is collected, so an unreadable column still fails on the first
    batch rather than after a full scan. A ragged row contributes None and a
    present-but-empty cell contributes "", the distinction
    collect_column_values documents and _hint_from_values counts apart.
    """
    try:
        if not source_exists(file_path):
            return None, tool_error(f"CSV file does not exist: {file_path}")
    except SourceError as exc:
        return None, tool_error(str(exc))

    values_by_column: Dict[str, List[Optional[str]]] = {
        column: [] for column in columns
    }
    saw_header = False
    try:
        for batch_header, rows in read_csv_batches(file_path):
            if not saw_header:
                saw_header = True
                for column in columns:
                    if column not in batch_header:
                        return None, _missing_column_error(
                            file_path, column, batch_header
                        )
            for row in rows:
                for column in columns:
                    values_by_column[column].append(row.get(column))
        if not saw_header:
            # No batches at all: either a header-only file or an empty one.
            header = read_csv_header(file_path)
            if not header:
                return None, tool_error(f"CSV file has no header row: {file_path}")
            for column in columns:
                if column not in header:
                    return None, _missing_column_error(file_path, column, header)
    except Exception as exc:  # noqa: BLE001 - report read failures to the agent
        return None, tool_error(f"Error reading CSV file {file_path}: {exc}")

    return values_by_column, None


def column_type_hint(file_path: str, column: str, tool_context: ToolContext) -> dict:
    """Reports what type one CSV column's values can actually be stored as.

    Use this before declaring a property's type in a construction plan. It
    answers one question only -- what the data supports -- and deliberately says
    nothing about whether a column is a good identifier ('column_stats'), or
    whether it survives being collapsed into a node ('collapse_check'). A
    suggestion is evidence, not a decision: a column of bare digits can be a
    product code, and only the column name and the user goal can tell.

    The counts come from the same converter the loader runs, so they are exactly
    what would happen at build time.

    Args:
      file_path: Path to the CSV file, relative to the source location.
      column: The column to analyze.
      tool_context: The ToolContext object.

    Returns:
        dict: 'status' of 'success' or 'error'. On success, a 'column_type_hint'
              key with 'path', 'column', 'shape' (one of 'bare_numeric',
              'numeric_after_cleaning', 'boolean_like', 'text'), 'suggested_type'
              ('integer', 'float', 'boolean', or null when the column is text),
              'convertible_count', 'blank_count' (empty cells: cleared if you
              declare a type for the property, stored as an empty string if you
              leave it text), 'missing_count' (rows too short to reach this
              column, which the build leaves untouched either way),
              'unconvertible_count' and up to three 'example_unconvertible'
              values.
    """
    values, error = collect_column_values(file_path, column)
    if error is not None:
        return error
    assert values is not None

    return tool_success(
        "column_type_hint", _hint_from_values(file_path, column, values)
    )


def column_type_hints(
    file_path: str, columns: List[str], tool_context: ToolContext
) -> dict:
    """Reports 'column_type_hint' for several columns of one file in a single call.

    Each column is analyzed with the same rules as 'column_type_hint'. Analysis
    stops at the first column that cannot be read, so the error names the column
    to correct.

    Args:
      file_path: Path to the CSV file, relative to the source location.
      columns: The columns to analyze.
      tool_context: The ToolContext object.

    Returns:
        dict: 'status' of 'success' or 'error'. On success, a 'column_type_hints'
              key holding one 'column_type_hint' payload per requested column, in
              the order requested.
    """
    # The argument is produced by a model, and a bare "price" would otherwise be
    # iterated into ['p', 'r', 'i', 'c', 'e'] and reported as "Column 'p' is not
    # in products.csv" -- an error about the data for what is a call-shape
    # mistake, sending the model to inspect a file that is fine. Same boundary
    # set_suggested_files guards, for the same reason.
    if columns is not None and not isinstance(columns, list):
        return tool_error(
            f"'columns' must be a list of column names, not {type(columns).__name__}. "
            f"Call 'column_type_hint' for a single column."
        )
    requested = list(columns or [])
    if not all(isinstance(column, str) for column in requested):
        return tool_error("Every entry of 'columns' must be a column name.")
    if not requested:
        return tool_success("column_type_hints", [])

    values_by_column, error = _collect_columns_values(file_path, requested)
    if error is not None:
        return error
    assert values_by_column is not None

    return tool_success(
        "column_type_hints",
        [
            _hint_from_values(file_path, column, values_by_column[column])
            for column in requested
        ],
    )


def collapse_check(
    file_path: str,
    node_key_column: str,
    candidate_column: str,
    tool_context: ToolContext,
) -> dict:
    """Checks whether a column survives collapsing rows onto a node key.

    This is the only tool that answers the post-MERGE question. Node loading
    MERGEs one node per distinct 'node_key_column' value and then overwrites
    the other properties from every row, so whichever row loads last wins. If
    the rows sharing a node key disagree about 'candidate_column', only one
    arbitrary value survives on the node, and any relationship joining on that
    column will silently match almost nothing.

    A candidate column is safe to use as a relationship join key only when
    every group has exactly one distinct value for it, i.e.
    'groups_with_conflicts' is 0 (which is trivially true when the candidate
    column *is* the node key).

    Note that 'column_stats' cannot answer this: a per-row ID is reported as
    perfectly unique, which is exactly the column class that does *not*
    survive collapsing. 'join_preview' cannot answer it either, because it
    compares raw CSV values before any collapsing happens.

    A file holding a header and no data rows is read as zero rows, and answers
    with zero counts and 'survives_collapse' True. That True is vacuous: nothing
    collapsed because there was nothing to collapse. Check 'row_count' before
    treating it as evidence the column is safe.

    Args:
      file_path: Path to the node file's CSV, relative to the source location.
      node_key_column: The column the nodes will be MERGEd on.
      candidate_column: The column being considered as a join key or property.
      tool_context: The ToolContext object.

    Returns:
        dict: 'status' of 'success' or 'error'. On success, a 'collapse_check'
              key with 'path', 'node_key_column', 'candidate_column',
              'row_count', 'group_count' (distinct node keys),
              'groups_with_conflicts' (groups holding more than one distinct
              candidate value), 'survives_collapse' (True when there are no
              conflicts) and 'example_conflicts' (up to 5 entries of
              {'node_key', 'values'}).
    """
    summary, error = summarize_key_groups(
        file_path, node_key_column, candidate_column, keep_examples=5
    )
    if error is not None:
        return error
    assert summary is not None

    return tool_success(
        "collapse_check",
        {
            "path": file_path,
            "node_key_column": node_key_column,
            "candidate_column": candidate_column,
            "row_count": summary.row_count,
            "group_count": summary.group_count,
            "groups_with_conflicts": summary.conflict_count,
            "survives_collapse": summary.conflict_count == 0,
            "example_conflicts": summary.examples,
        },
    )


def join_preview(
    file_a: str, column_a: str, file_b: str, column_b: str, tool_context: ToolContext
) -> dict:
    """Estimates how well a join between two CSV columns would match.

    Compares the distinct values of file_a's column against those of file_b's
    column, so a relationship construction can be checked for coverage before
    it is proposed. Empty values are ignored on both sides.

    Args:
      file_a: Path to the first CSV file, relative to the source location.
      column_a: Column in file_a to join on.
      file_b: Path to the second CSV file, relative to the source location.
      column_b: Column in file_b to join on.
      tool_context: The ToolContext object.

    Returns:
        dict: 'status' of 'success' or 'error'. On success, a 'join_preview'
              key with, for each side, the number of distinct values, how many
              of them have a match on the other side, and the matched fraction
              (0.0 when a side has no usable values).
    """
    summary_a, error = summarize_column(file_a, column_a)
    if error is not None:
        return error
    assert summary_a is not None
    summary_b, error = summarize_column(file_b, column_b)
    if error is not None:
        return error
    assert summary_b is not None

    overlap = summary_a.distinct & summary_b.distinct

    def fraction(matched: int, total: int) -> float:
        return round(matched / total, 4) if total else 0.0

    return tool_success(
        "join_preview",
        {
            "file_a": file_a,
            "column_a": column_a,
            "file_b": file_b,
            "column_b": column_b,
            "file_a_total": len(summary_a.distinct),
            "file_a_matched": len(overlap),
            "file_a_match_fraction": fraction(len(overlap), len(summary_a.distinct)),
            "file_b_total": len(summary_b.distinct),
            "file_b_matched": len(overlap),
            "file_b_match_fraction": fraction(len(overlap), len(summary_b.distinct)),
        },
    )


SEARCH_RESULTS = "search_results"


def search_file(file_path: str, query: str) -> dict:
    """Searches any text file for lines containing the query string, case-insensitively.

    Args:
      file_path: path relative to the source location
      query: the string to search for

    Returns:
        dict: 'status' of 'success' or 'error'. On success, a 'search_results'
              key with 'matching_lines' and metadata.
    """
    try:
        if not source_exists(file_path):
            return tool_error(f"File does not exist: {file_path}")
    except SourceError as exc:
        return tool_error(str(exc))

    if not query:
        return tool_success(
            SEARCH_RESULTS,
            {
                "metadata": {"path": file_path, "query": query, "lines_found": 0},
                "matching_lines": [],
            },
        )

    matching_lines = []
    search_query = query.lower()
    try:
        with open_source(file_path, "r") as handle:
            for line_number, line in enumerate(handle, 1):
                if search_query in line.lower():
                    matching_lines.append(
                        {
                            "line_number": line_number,
                            "content": line.strip(),
                        }
                    )
    except Exception as exc:  # noqa: BLE001
        return tool_error(f"Error reading or searching file {file_path}: {exc}")

    return tool_success(
        SEARCH_RESULTS,
        {
            "metadata": {
                "path": file_path,
                "query": query,
                "lines_found": len(matching_lines),
            },
            "matching_lines": matching_lines,
        },
    )
