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
carries the column when it is keyed by it, or keeps it as a property that
survives collapsing, and its values are those of its own source file -- which
may be a file where the column merely repeats. Each home file needs one node
that carries all of its values. Files where the column repeats are not required
to be covered: a value they point at that no home file lists is a data-quality
matter, which the join preview reports and the schema agent keeps and discloses.
No choice of key could fix it, so refusing would leave no way out.

It deliberately does NOT decide how a file should be modelled. It reports that a
choice of key left a home file's values without a node; keying a node by the
column from that file and adding a second node construction both resolve it, and
the caller says so.

Nothing here raises. Every read failure becomes a note, and a note never becomes
a refusal -- see the evidence rule in check_reference_columns_are_reachable.
"""

from typing import Dict, List, NamedTuple, Set, Tuple

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

    Per-row unique means no empty values and every value distinct -- the same
    condition column_stats reports as 'is_unique'. A column unique nowhere is not
    an identifier and gets no verdict at all.

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
        values, error = collect_column_values(path, column)
        if error is not None:
            evidence_complete = False
            notes.append(
                f"'{column}' could not be read in '{path}' ({error['error_message']})"
            )
            continue
        assert values is not None  # collect_column_values: error is None => values set
        non_empty = [v for v in values if v is not None and str(v).strip() != ""]
        value_sets[path] = {str(v) for v in non_empty}
        if values and len(non_empty) == len(values) == len(set(non_empty)):
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


def _survives_collapse(rule: dict, column: str) -> Tuple[bool, str | None]:
    """Stage 4 for one node rule: does the column keep one value per node?

    Returns (survives, error_message). An unreadable source returns
    (False, message) -- the caller must treat that as missing evidence, never as
    proof the column fails to survive. So does a malformed rule that names no
    source file or no key: it withholds evidence, it never supplies it.
    """
    source_file = rule.get("source_file")
    if not isinstance(source_file, str):
        return False, "the rule has no usable 'source_file'"
    key = _rule_unique_column_name(rule)
    if not key:
        return False, "the rule has no usable 'unique_column_name'"
    pairs, error = collect_column_pairs(source_file, key, column)
    if error is not None:
        return False, error["error_message"]
    groups = group_values_by_key(pairs)
    return all(len(values) == 1 for values in groups.values()), None


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


class _Witness(NamedTuple):
    """A node rule that carries the column, and the values it carries."""

    position: int
    rule: dict
    domain: Set[str]


def _carrying_rules(
    rules: List[dict], column: str, value_sets: Dict[str, Set[str]]
) -> Tuple[List[_Witness], List[Tuple[int, dict]]]:
    """Node rules that carry the column, split by how they carry it.

    A rule keyed by the column is a witness outright: MERGE makes one node per
    distinct key, so the node keys ARE the file's values. A rule that merely
    retains the column as a property is returned unjudged, because it is a
    witness only if the property survives collapsing and that costs a read.

    Only a rule built from a file whose values were read qualifies. A source_file
    that is not a string cannot be looked up (a list is unhashable), so it
    qualifies as nothing rather than raising.
    """
    keyed: List[_Witness] = []
    retaining: List[Tuple[int, dict]] = []
    for position, rule in enumerate(rules):
        source_file = rule.get("source_file")
        if not isinstance(source_file, str) or source_file not in value_sets:
            continue
        if _rule_unique_column_name(rule) == column:
            keyed.append(_Witness(position, rule, value_sets[source_file]))
        elif column in _rule_properties(rule):
            retaining.append((position, rule))
    return keyed, retaining


def _uncovered_homes(
    homes: List[str], value_sets: Dict[str, Set[str]], witnesses: List[_Witness]
) -> List[str]:
    """The home files whose values no single witness carries in full.

    `any` over the witnesses, deliberately not a union of their domains: a
    relationship joins to one label, so two nodes that each hold half of a home
    file's values leave that file short.
    """
    return [
        home
        for home in homes
        if not any(value_sets[home] <= witness.domain for witness in witnesses)
    ]


def _collapse_detail(column: str, rule: dict) -> str:
    """Why a retained property does not carry the column: its groups disagree."""
    return (
        f"'{rule.get('label')}' (built from '{rule.get('source_file')}', "
        f"keyed by '{rule.get('unique_column_name')}') retains "
        f"'{column}' as a property, but it does not survive collapsing: "
        f"nodes sharing a key disagree about it, so each keeps one "
        f"arbitrary value and a relationship joining on it would match "
        f"almost nothing, with no error at build time."
    )


_EXAMPLE_LIMIT = 3


def _quoted_list(names: List[str]) -> str:
    """Comma-separated, single-quoted file names or values for a message.

    One spelling, because these lists are read side by side in the same refusal:
    two ways of quoting the same kind of value diverge the moment a name contains
    a quote or a backslash.
    """
    return ", ".join(f"'{name}'" for name in names)


def _report(
    column: str,
    homes: List[str],
    repeating: List[str],
    detail: str,
    short_homes: List[str],
    consequence: str | None = None,
) -> str:
    """The refusal. It must offer BOTH routes out, every time.

    Re-keying and adding a second node construction both resolve this, and the
    check has no opinion on which is the better model. A message naming only one
    would smuggle in the modelling verdict this check deliberately does not make.
    The opening fact names every home file; the routes out name only the short
    ones, because naming a file that is already covered reads as "any of these
    would work".

    It says the relationship cannot be built AT ALL, never that coverage is low:
    the standing rules tell the model to keep a partially-covered relationship and
    report the fraction, so a refusal that reads as a coverage complaint gets a
    percentage reported and moved past. That is why a shortfall is stated as counts
    of values that have no node, never as a fraction, and why the message never
    points at the join preview. And it never suggests dropping the relationship,
    which is the failure this whole check exists to prevent.

    "Cannot be built at all" is scoped to the values no node carries, and it stays
    true because both routes out build from a home file, which by definition holds
    every one of its own values: whatever a home file is short of, keying a node by
    the column from that file supplies. That is why a refusal can never dead-end.
    """
    home_list = _quoted_list(homes)
    fix_list = _quoted_list(short_homes)
    if repeating:
        other_list = _quoted_list(repeating)
        appears_clause = f" and also appears in {other_list}"
        join_clause = f"joining {other_list} to {home_list}"
    else:
        # Every file sharing the column identifies rows by it -- there is no
        # "other" file left to name, but the column is still stranded: no home
        # file's node carries it reachably, so any relationship between those
        # home files still has nothing to join on.
        appears_clause = ""
        join_clause = f"among {home_list}"
    if consequence is None:
        consequence = (
            f"Any relationship {join_clause} therefore has no column to join on "
            f"and cannot be built at all."
        )
    return (
        f"'{column}' identifies rows in {home_list}{appears_clause}, but no node "
        f"in the plan carries it reachably: {detail} {consequence} Fix it "
        f"either by keying a node built from {fix_list} by '{column}', or by "
        f"adding a node construction from {fix_list} keyed by '{column}' "
        f"alongside the existing one."
    )


def _shortfall_clause(
    column: str,
    home: str,
    witnesses: List[_Witness],
    value_sets: Dict[str, Set[str]],
) -> str:
    """How far the closest witness falls short of one home file's values.

    Closest is the witness holding the most of them. `max` keeps the first of
    equals, so a tie goes to the earlier rule in the plan.
    """
    wanted = value_sets[home]
    closest = max(witnesses, key=lambda witness: len(wanted & witness.domain))
    carried = len(wanted & closest.domain)
    share = f"{carried} of the" if carried else "none of the"
    missing = _quoted_list(sorted(wanted - closest.domain)[:_EXAMPLE_LIMIT])
    rule = closest.rule
    return (
        f"'{rule.get('label')}' (built from '{rule.get('source_file')}', keyed by "
        f"'{rule.get('unique_column_name')}') carries {share} {len(wanted)} "
        f"'{column}' values '{home}' holds (missing e.g. {missing})"
    )


def _refusal(
    column: str,
    homes: List[str],
    repeating: List[str],
    short: List[str],
    witnesses: List[_Witness],
    not_surviving: List[dict],
    value_sets: Dict[str, Set[str]],
) -> str:
    """Choose the wording that is true of this shortfall, and report it.

    A witness that exists but falls short is described by counts. With no witness
    at all every home file is short, so the fix clause names them all.
    """
    if witnesses:
        ordered = sorted(witnesses, key=lambda witness: witness.position)
        clauses = [_shortfall_clause(column, h, ordered, value_sets) for h in short]
        consequence = (
            "The values no node carries have nothing to join to, so any "
            "relationship reaching them cannot be built at all."
        )
        if len(short) > 1:
            consequence += (
                f" Each of {_quoted_list(short)} needs a node carrying all its values."
            )
        return _report(
            column, homes, repeating, "; ".join(clauses) + ".", short, consequence
        )
    if not_surviving:
        detail = _collapse_detail(column, not_surviving[-1])
    else:
        detail = f"no node is keyed by '{column}', and none retains it as a property."
    return _report(column, homes, repeating, detail, homes)


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
        # built from it may be the node that carries every home file's values.
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

        witnesses, retaining = _carrying_rules(rules, column, value_sets)
        short = _uncovered_homes(homes, value_sets, witnesses)
        not_surviving: List[dict] = []
        if short:
            for position, rule in retaining:
                survives, error_message = _survives_collapse(rule, column)
                if survives:
                    domain = value_sets[rule["source_file"]]
                    witnesses.append(_Witness(position, rule, domain))
                    short = _uncovered_homes(homes, value_sets, witnesses)
                    if not short:
                        break
                elif error_message is not None:
                    evidence_complete = False
                    notes.append(
                        f"'{column}' could not be checked against "
                        f"'{rule.get('label')}' ({error_message})"
                    )
                else:
                    not_surviving.append(rule)
        # With no home file confirmed, nothing is short only vacuously: the file
        # that failed to read may be the home file, so fall through to the note.
        if not short and homes:
            continue  # every home file's values are carried by some node

        if evidence_complete:
            repeating = [path for path in files if path not in homes]
            problems.append(
                _refusal(
                    column,
                    homes,
                    repeating,
                    short,
                    witnesses,
                    not_surviving,
                    value_sets,
                )
            )
        else:
            unverified.append(
                f"reachability of '{column}' was not verified: " + "; ".join(notes)
            )

    return problems, unverified
