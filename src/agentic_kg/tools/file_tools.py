import logging

import clevercsv
from itertools import islice

from google.adk.tools import ToolContext
from typing import Dict, Any, List

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
                try:
                    # Just read enough to get the header
                    dialect = clevercsv.Sniffer().sniff(csvfile.read(2048))
                    csvfile.seek(0)
                    reader = clevercsv.reader(csvfile, dialect)
                except clevercsv.Error:
                    csvfile.seek(0)
                    reader = clevercsv.reader(csvfile)
                header_row = next(reader, [])
                # Empty query returns no matches, but we still read the header
        else:
            with open_source(file_path, "r") as csvfile:
                try:
                    # Read a chunk to sniff dialect, then rewind
                    dialect = clevercsv.Sniffer().sniff(csvfile.read(2048))
                    csvfile.seek(0)
                    reader = clevercsv.reader(csvfile, dialect)
                except clevercsv.Error:
                    # Fallback if sniffing fails (e.g., empty or very small file, or not CSV)
                    csvfile.seek(0)
                    reader = clevercsv.reader(csvfile) # Use default dialect
                    logger.warning(f"Could not sniff CSV dialect for {file_path}. Using default dialect.")
                
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
