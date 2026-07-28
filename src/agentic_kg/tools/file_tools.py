import logging

from itertools import islice

from google.adk.tools import ToolContext
from typing import Dict, Any, List

from agentic_kg.common.csv_reader import make_csv_reader, read_csv_batches
from agentic_kg.common.tool_result import tool_success, tool_error
from agentic_kg.common.file_source import (
    SourceError,
    get_source_root,
    list_source_files,
    open_source,
    source_exists,
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


def set_suggested_files(suggest_files:List[str], tool_context:ToolContext) -> Dict[str, Any]:
    """Set the files to be used for data import.
    """
    tool_context.state[SUGGESTED_FILES] = suggest_files
    return tool_success(SUGGESTED_FILES, suggest_files)

def get_suggested_files(tool_context:ToolContext) -> Dict[str, Any]:
    """Get the suggested files to be used for import.

    Returns:
        dict: A dictionary containing success or failure information.
              Includes a 'status' key ('success' or 'error').
              If 'success', includes a 'suggested_files' key with list of files.
              If 'error', includes an 'error_message' key.

    """
    if SUGGESTED_FILES not in tool_context.state:
        return tool_error("Suggested files have not been set. Take no action other than to inform user.")
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
        return tool_error("Current files have not been set. Take no action other than to inform user.")

    tool_context.state[APPROVED_FILES] = tool_context.state[SUGGESTED_FILES]
    return tool_success(APPROVED_FILES, tool_context.state[APPROVED_FILES])


def get_approved_files(tool_context:ToolContext) -> Dict[str, Any]:
    f"""Get the files that have been approved for importing into a knowledge graph."""
    
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


def search_csv_file(file_path: str, query: str, tool_context: ToolContext, case_sensitive: bool = False) -> dict:
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

                header_row = next(reader, []) # Store header, or empty list if file is empty

                for row in reader:
                    for field in row:
                        field_to_check = str(field) if case_sensitive else str(field).lower()
                        if search_query in field_to_check:
                            matching_rows.append(row)
                            break # Move to next row once a match is found
    except Exception as e:
        return tool_error(f"Error reading or searching CSV file {file_path}: {e}")

    result_data = {
        "metadata": {
            "path": file_path,
            "mimetype": "text/csv",
            "query": query,
            "case_sensitive": case_sensitive,
            "header": header_row,
            "rows_found": len(matching_rows)
        },
        "matching_rows": matching_rows
    }
    return tool_success("search_results", result_data)

def _collect_column_values(file_path: str, column: str):
    """Read every value of one column from a source CSV.

    Returns:
        (values, error) where values holds one entry per data row and error is a
        tool_error dict when the file or column cannot be read.

    read_csv_batches omits the key entirely for a row shorter than the header, so
    a ragged row contributes "" here rather than being skipped. That keeps one
    value per row, which is what column_stats' row_count and empty_count report.
    """
    try:
        if not source_exists(file_path):
            return None, tool_error(f"CSV file does not exist: {file_path}")
    except SourceError as exc:
        return None, tool_error(str(exc))

    values: List[str] = []
    header: List[str] = []
    saw_header = False
    try:
        for batch_header, rows in read_csv_batches(file_path):
            if not saw_header:
                header = batch_header
                saw_header = True
                if column not in header:
                    return None, tool_error(
                        f"Column '{column}' is not in {file_path}. Available columns: {header}"
                    )
            for row in rows:
                values.append(row.get(column, ""))
    except Exception as exc:  # noqa: BLE001 - report read failures to the agent
        return None, tool_error(f"Error reading CSV file {file_path}: {exc}")

    if not saw_header:
        return None, tool_error(f"CSV file has no header row: {file_path}")

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
    values, error = _collect_column_values(file_path, column)
    if error is not None:
        return error

    empty_count = sum(1 for value in values if value is None or str(value).strip() == "")
    non_empty = [value for value in values if value is not None and str(value).strip() != ""]
    distinct_count = len(set(non_empty))

    return tool_success("column_stats", {
        "path": file_path,
        "column": column,
        "row_count": len(values),
        "distinct_count": distinct_count,
        "empty_count": empty_count,
        "is_unique": empty_count == 0 and distinct_count == len(values),
    })


def _collect_column_pairs(file_path: str, column_a: str, column_b: str):
    """Read two columns of one source CSV, row by row.

    Returns:
        (pairs, error) where pairs is a list of (value_a, value_b) tuples and
        error is a tool_error dict when the file or either column cannot be
        read.
    """
    try:
        if not source_exists(file_path):
            return None, tool_error(f"CSV file does not exist: {file_path}")
    except SourceError as exc:
        return None, tool_error(str(exc))

    pairs = []
    saw_header = False
    try:
        for batch_header, rows in read_csv_batches(file_path):
            if not saw_header:
                saw_header = True
                missing = [c for c in (column_a, column_b) if c not in batch_header]
                if missing:
                    return None, tool_error(
                        f"Column(s) {missing} are not in {file_path}. "
                        f"Available columns: {batch_header}"
                    )
            for row in rows:
                pairs.append((row.get(column_a, ""), row.get(column_b, "")))
    except Exception as exc:  # noqa: BLE001 - report read failures to the agent
        return None, tool_error(f"Error reading CSV file {file_path}: {exc}")

    if not saw_header:
        return None, tool_error(f"CSV file has no header row: {file_path}")

    return pairs, None


def collapse_check(file_path: str, node_key_column: str, candidate_column: str,
                   tool_context: ToolContext) -> dict:
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
    pairs, error = _collect_column_pairs(file_path, node_key_column, candidate_column)
    if error is not None:
        return error

    groups: Dict[str, set] = {}
    for key, value in pairs:
        key_text = "" if key is None else str(key)
        value_text = "" if value is None else str(value)
        groups.setdefault(key_text, set()).add(value_text)

    conflicts = [(key, values) for key, values in groups.items() if len(values) > 1]
    example_conflicts = [
        {"node_key": key, "values": sorted(values)[:10]}
        for key, values in conflicts[:5]
    ]

    return tool_success("collapse_check", {
        "path": file_path,
        "node_key_column": node_key_column,
        "candidate_column": candidate_column,
        "row_count": len(pairs),
        "group_count": len(groups),
        "groups_with_conflicts": len(conflicts),
        "survives_collapse": len(conflicts) == 0,
        "example_conflicts": example_conflicts,
    })


def join_preview(file_a: str, column_a: str, file_b: str, column_b: str,
                 tool_context: ToolContext) -> dict:
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
    values_a, error = _collect_column_values(file_a, column_a)
    if error is not None:
        return error
    values_b, error = _collect_column_values(file_b, column_b)
    if error is not None:
        return error

    distinct_a = {str(v) for v in values_a if v is not None and str(v).strip() != ""}
    distinct_b = {str(v) for v in values_b if v is not None and str(v).strip() != ""}
    overlap = distinct_a & distinct_b

    def fraction(matched: int, total: int) -> float:
        return round(matched / total, 4) if total else 0.0

    return tool_success("join_preview", {
        "file_a": file_a,
        "column_a": column_a,
        "file_b": file_b,
        "column_b": column_b,
        "file_a_total": len(distinct_a),
        "file_a_matched": len(overlap),
        "file_a_match_fraction": fraction(len(overlap), len(distinct_a)),
        "file_b_total": len(distinct_b),
        "file_b_matched": len(overlap),
        "file_b_match_fraction": fraction(len(overlap), len(distinct_b)),
    })


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
        return tool_success(SEARCH_RESULTS, {
            "metadata": {"path": file_path, "query": query, "lines_found": 0},
            "matching_lines": [],
        })

    matching_lines = []
    search_query = query.lower()
    try:
        with open_source(file_path, "r") as handle:
            for line_number, line in enumerate(handle, 1):
                if search_query in line.lower():
                    matching_lines.append({
                        "line_number": line_number,
                        "content": line.strip(),
                    })
    except Exception as exc:  # noqa: BLE001
        return tool_error(f"Error reading or searching file {file_path}: {exc}")

    return tool_success(SEARCH_RESULTS, {
        "metadata": {
            "path": file_path,
            "query": query,
            "lines_found": len(matching_lines),
        },
        "matching_lines": matching_lines,
    })
