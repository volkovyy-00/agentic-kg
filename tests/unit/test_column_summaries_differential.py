"""Differential test: the streaming summarisers against today's list logic.

The oracle below is a VERBATIM copy of file_tools.group_values_by_key,
collapse_check's reduction, and _property_failure's reverse grouping (the
source of _oracle_values_on_one_key) as they stood at fcc0661, before this
change deleted them. It is duplicated on purpose: its whole job is to outlive
the original, so the old semantics remain executable and a regression is a
diff rather than a judgement call. Do not "simplify" it to call the new code.
"""

import random

import pytest

from agentic_kg.tools import file_tools


def _oracle_groups(pairs):
    """Verbatim copy of group_values_by_key at fcc0661."""
    groups = {}
    for key, value in pairs:
        key_text = "" if key is None else str(key)
        value_text = "" if value is None else str(value)
        groups.setdefault(key_text, set()).add(value_text)
    return groups


def _oracle_examples(pairs):
    """Verbatim copy of collapse_check's reduction at fcc0661."""
    groups = _oracle_groups(pairs)
    conflicts = [(key, values) for key, values in groups.items() if len(values) > 1]
    return {
        "group_count": len(groups),
        "groups_with_conflicts": len(conflicts),
        "example_conflicts": [
            {"node_key": key, "values": sorted(values)[:10]}
            for key, values in conflicts[:5]
        ],
    }


def _oracle_values_on_one_key(pairs):
    """Verbatim copy of _property_failure's reverse grouping at fcc0661."""
    by_value = _oracle_groups(
        (value, key)
        for key, value in pairs
        if value is not None and str(value).strip() != ""
    )
    return all(len(keys) == 1 for keys in by_value.values())


def _random_pairs(rng):
    """A pair list shaped to hit the cases that distinguish the two algorithms."""
    key_count = rng.randint(1, 9)
    # The whitespace entries are load-bearing: " " and "   " are blank for the
    # emptiness test but distinct values for identity, and " v00"/"v00 " sort
    # around "v00". Without them, three mutants that strip before comparing
    # survive every test in this repo.
    value_pool = ["", " ", "   ", " v00", "v00 ", None] + [
        f"v{i:02d}" for i in range(rng.randint(1, 15))
    ]
    keys = [None, ""] + [f"k{i}" for i in range(key_count)]
    return [
        (rng.choice(keys), rng.choice(value_pool)) for _ in range(rng.randint(0, 60))
    ]


def test_the_oracle_reproduces_a_known_collapse_check_answer():
    """Guards the oracle itself: if this drifts from the shipped tool, every
    other assertion in this file is comparing two wrongs."""
    pairs = [("Bolt", "a1"), ("Bolt", "a2"), ("Nut", "a3")]
    assert _oracle_examples(pairs) == {
        "group_count": 2,
        "groups_with_conflicts": 1,
        "example_conflicts": [{"node_key": "Bolt", "values": ["a1", "a2"]}],
    }


def _write(fs, pairs):
    """Write a pair list as a CSV, escaping nothing -- the values are generated."""
    lines = ["key,value"]
    for key, value in pairs:
        if key is None:
            lines.append("")  # a short row: no cell reaches 'value' either
        else:
            lines.append(f"{key},{'' if value is None else value}")
    with fs.open("/src/pairs.csv", "w") as handle:
        handle.write("\n".join(lines) + "\n")


@pytest.fixture
def pair_source(monkeypatch):
    import fsspec

    from agentic_kg.common.config import reset_settings

    fs = fsspec.filesystem("memory")
    fs.store.clear()
    fs.pseudo_dirs.clear()
    monkeypatch.setenv("SOURCE_URI", "memory://src")
    reset_settings()
    yield fs
    fs.store.clear()
    fs.pseudo_dirs.clear()


@pytest.mark.parametrize("seed", range(200))
def test_the_streaming_grouping_matches_the_old_list_logic(pair_source, seed):
    """Seeded rather than random, so a failure is reproducible from its id alone.
    hypothesis is not a dependency; this covers the same ground for this shape."""
    rng = random.Random(seed)
    pairs = _random_pairs(rng)
    _write(pair_source, pairs)

    summary, error = file_tools.summarize_key_groups(
        "pairs.csv", "key", "value", keep_examples=5, track_value_owners=True
    )
    assert error is None
    assert summary is not None

    # The oracle reads what the CSV really produced, not the generated pairs:
    # a written short row comes back with absent keys, which is the point.
    rows, read_error = file_tools._column_rows("pairs.csv", ["key", "value"])
    assert read_error is None
    written = list(rows)

    expected = _oracle_examples(written)
    assert summary.group_count == expected["group_count"]
    assert summary.conflict_count == expected["groups_with_conflicts"]
    assert summary.examples == expected["example_conflicts"]
    assert summary.values_on_one_key == _oracle_values_on_one_key(written)
