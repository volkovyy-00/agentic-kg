"""Does a node's key leave another approved file's reference column unreachable?

A column that identifies rows in one approved file and also appears, under the
same name, in another is how the second file points at the first. If the plan
keys its node by something else and does not preserve that column, the pointer
has nothing to point at: no relationship joining the two files can be built at
all, and the only approvable plan is one with that relationship missing. Nothing
errors -- the relationship is simply never proposed.

This module answers that one question mechanically, because the prose rule that
used to answer it resolved the same file two different ways on two runs.

It deliberately does NOT decide how a file should be modelled. It reports that a
choice of key made another file unreachable; keying the node differently and
adding a second node construction both resolve it, and the caller says so.

Nothing here raises. Every read failure becomes a note, and a note never becomes
a refusal -- see the evidence rule in check_reference_columns_are_reachable.
"""

from typing import Dict, List, Tuple

from agentic_kg.common.csv_reader import read_csv_header

from .file_tools import collect_column_values


def _columns_by_file(
    approved_files: List[str],
) -> Tuple[Dict[str, List[str]], List[str], List[str]]:
    """Stage 1: which approved files carry each column name.

    Headers only. Matching is literal string equality -- two files naming the
    same concept differently are not connected, which is a harder problem this
    check does not attempt.

    read_csv_header raises rather than returning an error result, so the failure
    is caught here and reported as a note.
    """
    columns: Dict[str, List[str]] = {}
    unreadable: List[str] = []
    notes: List[str] = []
    for path in approved_files or []:
        try:
            header = read_csv_header(path)
        except Exception as exc:  # noqa: BLE001 - a bad source must not raise here
            unreadable.append(path)
            notes.append(f"the header of '{path}' could not be read ({exc})")
            continue
        for column in header:
            columns.setdefault(column, []).append(path)
    return columns, unreadable, notes


def _home_files(column: str, files: List[str]) -> Tuple[List[str], bool, List[str]]:
    """Stage 2: the files in which this column identifies rows.

    Per-row unique means no empty values and every value distinct -- the same
    condition column_stats reports as 'is_unique'. A column unique nowhere is not
    an identifier and gets no verdict at all.

    Returns evidence_complete=False when any file's values could not be read, so
    a later refusal can be downgraded rather than built on missing evidence.
    """
    homes: List[str] = []
    evidence_complete = True
    notes: List[str] = []
    for path in files:
        values, error = collect_column_values(path, column)
        if error is not None:
            evidence_complete = False
            notes.append(
                f"'{column}' could not be read in '{path}' ({error['error_message']})"
            )
            continue
        non_empty = [v for v in values if v is not None and str(v).strip() != ""]
        if values and len(non_empty) == len(values) == len(set(non_empty)):
            homes.append(path)
    return homes, evidence_complete, notes
