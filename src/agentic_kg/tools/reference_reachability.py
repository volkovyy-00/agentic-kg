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

from .file_tools import collect_column_pairs, collect_column_values, group_values_by_key


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
    if not isinstance(approved_files, (list, tuple)):
        return columns, unreadable, notes

    seen_paths: set = set()
    for path in approved_files:
        # One file cannot reference itself. A path repeated in the approved list,
        # or a column name repeated inside one header, would otherwise make a
        # single file look like the two files the whole check keys on.
        if path in seen_paths:
            continue
        seen_paths.add(path)
        try:
            header = read_csv_header(path)
        except Exception as exc:  # noqa: BLE001 - a bad source must not raise here
            unreadable.append(path)
            notes.append(f"the header of '{path}' could not be read ({exc})")
            continue
        for column in dict.fromkeys(header):
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


def _node_rules(construction_plan: dict) -> List[dict]:
    """Every well-formed node rule in the plan, in plan order."""
    if not isinstance(construction_plan, dict):
        return []
    return [
        rule
        for rule in construction_plan.values()
        if isinstance(rule, dict) and rule.get("construction_type") == "node"
    ]


def _survives_collapse(rule: dict, column: str) -> Tuple[bool, str | None]:
    """Stage 4 for one node rule: does the column keep one value per node?

    Returns (survives, error_message). An unreadable source returns
    (False, message) -- the caller must treat that as missing evidence, never as
    proof the column fails to survive.
    """
    pairs, error = collect_column_pairs(
        rule.get("source_file"), rule.get("unique_column_name"), column
    )
    if error is not None:
        return False, error["error_message"]
    groups = group_values_by_key(pairs)
    return all(len(values) == 1 for values in groups.values()), None


def _rule_properties(rule: dict) -> List[str]:
    """A rule's 'properties', defensively.

    A malformed rule can carry anything here -- an int, a string, another
    dict. Only a genuine list is a property list; anything else is treated as
    empty rather than risking a crash (non-iterable) or silent substring
    matching (a string 'in' check) against a value the rule never declared.
    """
    properties = rule.get("properties")
    return properties if isinstance(properties, list) else []


def _rule_unique_column_name(rule: dict) -> str | None:
    """A rule's 'unique_column_name', defensively.

    A malformed rule can carry a non-string here (e.g. a list), which is not
    a column name and must not be treated as one -- putting it straight into
    a set literal alongside properties raises 'unhashable type' instead.
    """
    name = rule.get("unique_column_name")
    return name if isinstance(name, str) else None


def _quoted_list(paths: List[str]) -> str:
    """Comma-separated, single-quoted file names for a message.

    One spelling, because these lists are read side by side in the same refusal:
    two ways of quoting the same kind of value diverge the moment a path contains
    a quote or a backslash.
    """
    return ", ".join(f"'{path}'" for path in paths)


def _report(column: str, homes: List[str], referencing: List[str], detail: str) -> str:
    """The refusal. It must offer BOTH routes out, every time.

    Re-keying and adding a second node construction both resolve this, and the
    check has no opinion on which is the better model. A message naming only one
    would smuggle in the modelling verdict this check deliberately does not make.

    It says the relationship cannot be built AT ALL, never that coverage is low:
    the standing rules tell the model to keep a partially-covered relationship and
    report the fraction, so a refusal that reads as a coverage complaint gets a
    percentage reported and moved past. And it never suggests dropping the
    relationship, which is the failure this whole check exists to prevent.
    """
    home_list = _quoted_list(homes)
    if referencing:
        other_list = _quoted_list(referencing)
        appears_clause = f" and also appears in {other_list}"
        join_clause = f"joining {other_list} to {home_list}"
    else:
        # Every file sharing the column identifies rows by it -- there is no
        # "other" file left to name, but the column is still stranded: no home
        # file's node carries it reachably, so any relationship between those
        # home files still has nothing to join on.
        appears_clause = ""
        join_clause = f"among {home_list}"
    return (
        f"'{column}' identifies rows in {home_list}{appears_clause}, but no node "
        f"in the plan carries it reachably: {detail} Any relationship {join_clause} "
        f"therefore has no column to join on and cannot be built at all. Fix it "
        f"either by keying a node built from {home_list} by '{column}', or by "
        f"adding a node construction from {home_list} keyed by '{column}' "
        f"alongside the existing one."
    )


def check_reference_columns_are_reachable(
    construction_plan: dict, approved_files: List[str]
) -> Tuple[List[str], List[str]]:
    """Report reference columns the plan leaves with nothing to point at.

    Returns (problems, unverified). Both are always lists and this never raises:
    it runs on every plan presentation, so a bad source must not make a plan
    unshowable.

    THE EVIDENCE RULE, which is one flag per candidate and NOT a per-stage check:
    a 'reachable' verdict may short-circuit freely, because a failed read only
    ever withholds evidence and never manufactures it. A refusal is emitted only
    when no read relevant to that column failed; otherwise it downgrades to
    unverified. Checking this per stage instead looks correct and is not -- a
    column unique in a readable file AND in an unreadable one the node is built
    from would short-circuit on the readable file, never credit the unreadable one
    as a home file, and refuse a correct plan on the strength of a failed read.
    """
    columns, unreadable, problems_notes = _columns_by_file(approved_files)
    rules = _node_rules(construction_plan)
    problems: List[str] = []
    unverified: List[str] = list(problems_notes)

    for column, files in sorted(columns.items()):
        if len(files) < 2:
            continue

        homes, evidence_complete, notes = _home_files(column, files)
        if not homes and evidence_complete:
            continue  # shared, but identifies rows nowhere: not a reference column

        # An unreadable file cannot be shown to lack this column, so a node rule
        # built from it may have supplied the home file that makes this reachable.
        # Each blocker is named in this column's OWN notes: the stage 1 note says
        # a file was unreadable, but only this knows which candidate that cost.
        for rule in rules:
            if rule.get("source_file") in unreadable and column in {
                _rule_unique_column_name(rule),
                *_rule_properties(rule),
            }:
                evidence_complete = False
                notes.append(
                    f"'{rule.get('source_file')}' could not be read, and "
                    f"'{rule.get('label')}' is built from it"
                )

        home_rules = [rule for rule in rules if rule.get("source_file") in homes]
        if any(rule.get("unique_column_name") == column for rule in home_rules):
            continue  # stage 3: keyed by it, on a file that owns it

        detail = (
            f"no node built from {_quoted_list(homes)} is keyed by "
            f"'{column}', and none retains it as a property."
        )
        for rule in home_rules:
            if column not in _rule_properties(rule):
                continue
            survives, error_message = _survives_collapse(rule, column)
            if survives:
                detail = None
                break
            if error_message is not None:
                evidence_complete = False
                notes.append(
                    f"'{column}' could not be checked against "
                    f"'{rule.get('label')}' ({error_message})"
                )
            else:
                detail = (
                    f"'{rule.get('label')}' (built from '{rule.get('source_file')}', "
                    f"keyed by '{rule.get('unique_column_name')}') retains "
                    f"'{column}' as a property, but it does not survive collapsing: "
                    f"nodes sharing a key disagree about it, so each keeps one "
                    f"arbitrary value and a relationship joining on it would match "
                    f"almost nothing, with no error at build time."
                )
        if detail is None:
            continue  # stage 4: a home-file node preserves it

        if evidence_complete:
            referencing = [path for path in files if path not in homes]
            problems.append(_report(column, homes, referencing, detail))
        else:
            unverified.append(
                f"reachability of '{column}' was not verified: " + "; ".join(notes)
            )

    return problems, unverified
