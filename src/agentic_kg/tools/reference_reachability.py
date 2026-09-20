"""Does the plan leave an approved file's reference column with nothing to point at?

A column that identifies rows in one approved file (its "home" file: unique per
row) and also appears, under the same name, in another is how the second file
points at the first. If no node in the plan carries every value the home file
holds for that column, the pointer has nothing to point at: no relationship
joining the two files can be built for the values left out, and the only
approvable plan is one with that relationship missing. Nothing errors -- the
relationship is simply never proposed.

This module answers that one question mechanically, because the prose rule that
used to answer it resolved the same file two different ways on two runs.

What counts is the values a node carries, not which file built it. A node
carries the column when it is keyed by it, or keeps it as a property with one
value per node and one node per value, and its values are those of its own
source file -- which may be a file where the column merely repeats. Some home
file needs one node that carries all of its values. Other files need not be
covered, whether the column repeats there or is unique there too: a value they
point at that no node has is a data-quality matter, which the join preview
reports and the schema agent keeps and discloses. Requiring it would refuse any
two files that merely share a unique column name.

It deliberately does NOT decide how a file should be modelled. It reports that a
choice of key left no home file's values with a node; keying a node by the
column from a home file and adding a second node construction both resolve it,
and the caller says so.

Nothing here raises. Every read failure becomes a note, and a note never becomes
a refusal -- see the evidence rule in check_reference_columns_are_reachable.
"""

from typing import Dict, List, NamedTuple, Set, Tuple

from agentic_kg.common.csv_reader import read_csv_header

from .file_tools import summarize_column, summarize_key_groups


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
        # A path is a string. Anything else names no file, and an unhashable one
        # (a list, a dict) would raise on the dedupe below rather than reaching
        # the read guarded above -- a raise inside a plan presentation.
        if not isinstance(path, str):
            continue
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


def _home_files(
    column: str, files: List[str]
) -> Tuple[List[str], bool, List[str], Dict[str, Set[str]]]:
    """Stage 2: the files in which this column identifies rows, and every value read.

    Per-row unique means no empty values and every value distinct --
    ColumnSummary.is_unique decides that, and column_stats reports the same
    property from the same place. The at-least-one-row condition is this
    function's own: a zero-row column is vacuously unique but is nobody's
    identifier home.

    A column unique nowhere is not an identifier and gets no verdict at all.

    Returns evidence_complete=False when any file's values could not be read, so
    a later refusal can be downgraded rather than built on missing evidence.

    The fourth element maps every file that WAS read -- home or not -- to its
    distinct non-blank values. A node built from a file where the column repeats
    can still carry every value a home file holds, so that file's values are
    needed too. The blank filter here is the only one: later comparisons read
    these sets rather than filtering again.
    """
    homes: List[str] = []
    evidence_complete = True
    notes: List[str] = []
    value_sets: Dict[str, Set[str]] = {}
    for path in files:
        summary, error = summarize_column(path, column)
        if error is not None:
            evidence_complete = False
            notes.append(
                f"'{column}' could not be read in '{path}' ({error['error_message']})"
            )
            continue
        assert summary is not None  # summarize_column: error is None => summary set
        value_sets[path] = summary.distinct
        if summary.row_count and summary.is_unique:
            homes.append(path)
    return homes, evidence_complete, notes, value_sets


def _node_rules(construction_plan: dict) -> List[dict]:
    """Every well-formed node rule in the plan, in plan order."""
    if not isinstance(construction_plan, dict):
        return []
    return [
        rule
        for rule in construction_plan.values()
        if isinstance(rule, dict) and rule.get("construction_type") == "node"
    ]


def _property_failure(rule: dict, column: str) -> Tuple[str | None, str | None]:
    """Why a retained property cannot be joined on, or (None, None) if it can.

    A retained column is a join target only when each node keeps one value of it
    (it survives collapsing) AND each value sits on one node. Surviving alone is
    not enough: a node keyed per row of a file where the column repeats keeps its
    one value, but several nodes share it, so a relationship joining on it lands
    on all of them instead of on the one entity the value names.

    Returns (failure, error_message). An unreadable source returns (None, message)
    -- the caller must treat that as missing evidence, never as proof the property
    fails. So does a malformed rule that names no source file or no key: it
    withholds evidence, it never supplies it.
    """
    source_file = rule.get("source_file")
    if not isinstance(source_file, str):
        return None, "the rule has no usable 'source_file'"
    key = _rule_unique_column_name(rule)
    if not key:
        return None, "the rule has no usable 'unique_column_name'"
    summary, error = summarize_key_groups(
        source_file, key, column, track_value_owners=True
    )
    if error is not None:
        return None, error["error_message"]
    assert summary is not None  # summarize_key_groups: error is None => summary set
    if summary.conflict_count:
        return _collapse_detail(column, rule), None
    if not summary.values_on_one_key:
        return _shared_detail(column, rule), None
    return None, None


def _rule_properties(rule: dict) -> List[str]:
    """A rule's 'properties', defensively -- only the strings in it.

    A malformed rule can carry anything here -- an int, a string, another
    dict. Only a genuine list is a property list; anything else is treated as
    empty rather than risking a crash (non-iterable) or silent substring
    matching (a string 'in' check) against a value the rule never declared.

    The elements need the same guard as the container: a property name is a
    column name, so a non-string is not one, and dropping it here cannot
    change a match. Passing one through can still crash -- an unhashable
    element (a dict, a list) raises the moment a caller splats this into a
    set, which is exactly what stage 1's unreadable-source scan does.
    """
    properties = rule.get("properties")
    if not isinstance(properties, list):
        return []
    return [name for name in properties if isinstance(name, str)]


def _rule_unique_column_name(rule: dict) -> str | None:
    """A rule's 'unique_column_name', defensively.

    A malformed rule can carry a non-string here (e.g. a list), which is not
    a column name and must not be treated as one -- putting it straight into
    a set literal alongside properties raises 'unhashable type' instead.
    """
    name = rule.get("unique_column_name")
    return name if isinstance(name, str) else None


class _Carrier(NamedTuple):
    """A node rule that carries some of the column's values, and which ones."""

    position: int
    rule: dict
    domain: Set[str]


def _carrying_rules(
    rules: List[dict], column: str, value_sets: Dict[str, Set[str]]
) -> Tuple[List[_Carrier], List[Tuple[int, dict]]]:
    """Node rules that carry the column, split by how they carry it.

    A rule keyed by the column carries it outright: MERGE makes one node per
    distinct key, so the node keys ARE the file's values, one node each. A rule
    that merely retains the column as a property is returned unjudged, because it
    carries the column only if each node keeps one value and each value sits on
    one node, and finding that out costs a read.

    Only a rule built from a file whose values were read qualifies. A source_file
    that is not a string cannot be looked up (a list is unhashable), so it
    qualifies as nothing rather than raising.
    """
    keyed: List[_Carrier] = []
    retaining: List[Tuple[int, dict]] = []
    for position, rule in enumerate(rules):
        source_file = rule.get("source_file")
        if not isinstance(source_file, str) or source_file not in value_sets:
            continue
        if _rule_unique_column_name(rule) == column:
            keyed.append(_Carrier(position, rule, value_sets[source_file]))
        elif column in _rule_properties(rule):
            retaining.append((position, rule))
    return keyed, retaining


def _covers_a_home(
    homes: List[str], value_sets: Dict[str, Set[str]], carriers: List[_Carrier]
) -> bool:
    """Whether some single carrier holds every value of some home file.

    One home file is enough. A second file that also identifies rows by the
    column, listing values the first lacks, is treated like a repeating file
    pointing at an id no node has: a data-quality matter, not a stranded column.
    Requiring it too would refuse any two files that merely share a unique column
    name.

    `any` over single carriers, deliberately not a union of their domains: a
    relationship joins to one label, so two nodes that each hold half of a home
    file's values leave it uncovered.
    """
    return any(
        value_sets[home] <= carrier.domain for home in homes for carrier in carriers
    )


def _node_description(rule: dict) -> str:
    return (
        f"'{rule.get('label')}' (built from '{rule.get('source_file')}', "
        f"keyed by '{rule.get('unique_column_name')}')"
    )


def _collapse_detail(column: str, rule: dict) -> str:
    """Why a retained property does not carry the column: its groups disagree."""
    return (
        f"{_node_description(rule)} retains '{column}' as a property, but it does "
        f"not survive collapsing: nodes sharing a key disagree about it, so each "
        f"keeps one arbitrary value and a relationship joining on it would match "
        f"almost nothing, with no error at build time."
    )


def _shared_detail(column: str, rule: dict) -> str:
    """Why a retained property does not carry the column: its values repeat."""
    return (
        f"{_node_description(rule)} retains '{column}' as a property, but more "
        f"than one node holds the same value, so a relationship joining on it "
        f"would attach to all of them instead of to the one row the value "
        f"identifies."
    )


_EXAMPLE_LIMIT = 3
_VALUE_LIMIT = 40


def _quoted_list(names: List[str]) -> str:
    """Comma-separated, single-quoted file names for a message.

    One spelling, because these lists are read side by side in the same refusal:
    two ways of quoting the same kind of value diverge the moment a name contains
    a quote or a backslash.
    """
    return ", ".join(f"'{name}'" for name in names)


def _quoted_value(value: str) -> str:
    """One cell value for a message, which the model reads one problem per line.

    repr() so that a quote or a newline in the cell cannot break that layout, and
    cut short so that one enormous cell cannot swamp the message.
    """
    if len(value) > _VALUE_LIMIT:
        value = value[:_VALUE_LIMIT] + "..."
    return repr(value)


def _report(
    column: str,
    homes: List[str],
    repeating: List[str],
    detail: str,
    consequence: str | None = None,
) -> str:
    """The refusal. It must offer BOTH routes out, every time.

    Re-keying and adding a second node construction both resolve this, and the
    check has no opinion on which is the better model. A message naming only one
    would smuggle in the modelling verdict this check deliberately does not make.
    The new node needs a label of its own: proposing it under an existing label
    replaces that node instead of adding one.

    It says the relationship cannot be built AT ALL, never that coverage is low:
    the standing rules tell the model to keep a partially-covered relationship and
    report the fraction, so a refusal that reads as a coverage complaint gets a
    percentage reported and moved past. That is why a shortfall is stated as counts
    of values that have no node, never as a fraction, and why the message never
    points at the join preview. And it never suggests dropping the relationship,
    which is the failure this whole check exists to prevent.

    "Cannot be built at all" is scoped to the values no node carries, and it stays
    true because both routes out build from a home file, which by definition holds
    every one of its own values: keying a node by the column from that file
    supplies all of them. That is why a refusal can never dead-end.
    """
    home_list = _quoted_list(homes)
    holder = "that file holds" if len(homes) == 1 else "any one of those files holds"
    if repeating:
        other_list = _quoted_list(repeating)
        appears_clause = f" and also appears in {other_list}"
        join_clause = f"joining {other_list} to {home_list}"
    else:
        # Every file sharing the column identifies rows by it -- there is no
        # "other" file left to name, but the column is still stranded: no node
        # carries a home file's values, so any relationship between those home
        # files still has nothing to join on.
        appears_clause = ""
        join_clause = f"among {home_list}"
    if consequence is None:
        consequence = (
            f"Any relationship {join_clause} therefore has no column to join on "
            f"and cannot be built at all."
        )
    return (
        f"'{column}' identifies rows in {home_list}{appears_clause}, but no node "
        f"in the plan carries every value {holder}: {detail} {consequence} Fix it "
        f"either by keying a node built from {home_list} by '{column}', or by "
        f"adding a node construction from {home_list} keyed by '{column}', under a "
        f"label of its own, alongside the existing one."
    )


def _shortfall_clause(
    column: str,
    home: str,
    carriers: List[_Carrier],
    value_sets: Dict[str, Set[str]],
) -> str:
    """How far the closest carrier falls short of one home file's values.

    Closest is the carrier holding the most of them. `max` keeps the first of
    equals, so a tie goes to the earlier rule in the plan.

    A keyed carrier carries its values outright. One that only retains the column
    is here because its file cannot cover the home file, so it was never read and
    whether the property survives is unknown: all that is known is an upper bound.
    """
    wanted = value_sets[home]
    closest = max(carriers, key=lambda carrier: len(wanted & carrier.domain))
    carried = len(wanted & closest.domain)
    count = f"{carried} of the" if carried else "none of the"
    if _rule_unique_column_name(closest.rule) == column:
        share = f"carries {count}"
    elif carried:
        share = f"retains '{column}', so could carry at most {count}"
    else:
        share = f"retains '{column}' but could carry {count}"
    examples = sorted(wanted - closest.domain)[:_EXAMPLE_LIMIT]
    missing = ", ".join(_quoted_value(value) for value in examples)
    return (
        f"{_node_description(closest.rule)} {share} {len(wanted)} "
        f"'{column}' values '{home}' holds (missing e.g. {missing})"
    )


def _refusal(
    column: str,
    homes: List[str],
    repeating: List[str],
    carriers: List[_Carrier],
    failures: List[str],
    value_sets: Dict[str, Set[str]],
) -> str:
    """Choose the wording that is true of this shortfall, and report it.

    Every reason is stated: a retained property that fails, and how far the
    closest carrier falls short of each home file, since covering any one of them
    would do. Only with neither does the message say no node carries the column,
    and then it names the files it looked at, because a rule built from any other
    path carries nothing the check can see.
    """
    details = list(failures)
    consequence = None
    if carriers:
        ordered = sorted(carriers, key=lambda carrier: carrier.position)
        clauses = [_shortfall_clause(column, h, ordered, value_sets) for h in homes]
        details.append("; ".join(clauses) + ".")
        consequence = (
            "The values no node carries have nothing to join to, so any "
            "relationship reaching them cannot be built at all."
        )
    elif not failures:
        details.append(
            f"no node built from {_quoted_list(list(value_sets))} is keyed by "
            f"'{column}', and none retains it as a property."
        )
    return _report(column, homes, repeating, " ".join(details), consequence)


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

        homes, evidence_complete, notes, value_sets = _home_files(column, files)
        if not homes and evidence_complete:
            continue  # shared, but identifies rows nowhere: not a reference column

        # An unreadable file cannot be shown to lack this column, so a node rule
        # built from it may be the node that carries a home file's values.
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

        carriers, retaining = _carrying_rules(rules, column, value_sets)
        covered = _covers_a_home(homes, value_sets, carriers)
        failures: List[str] = []
        if not covered:
            for position, rule in retaining:
                domain = value_sets[rule["source_file"]]
                if not any(value_sets[home] <= domain for home in homes):
                    # Nothing a read could show would let this rule cover a home
                    # file, so it is not read: a failed read here would withhold
                    # a refusal the values already prove.
                    carriers.append(_Carrier(position, rule, domain))
                    continue
                failure, error_message = _property_failure(rule, column)
                if error_message is not None:
                    evidence_complete = False
                    notes.append(
                        f"'{column}' could not be checked against "
                        f"'{rule.get('label')}' ({error_message})"
                    )
                elif failure is not None:
                    failures.append(failure)
                else:
                    covered = True
                    break
        if covered:
            continue  # some node carries every value a home file holds

        if evidence_complete:
            repeating = [path for path in files if path not in homes]
            problems.append(
                _refusal(column, homes, repeating, carriers, failures, value_sets)
            )
        else:
            unverified.append(
                f"reachability of '{column}' was not verified: " + "; ".join(notes)
            )

    return problems, unverified
