import csv
import io

import fsspec
import pytest

from agentic_kg.common.config import reset_settings
from agentic_kg.common.csv_reader import read_csv_batches, read_csv_header


@pytest.fixture
def csv_source(monkeypatch):
    fs = fsspec.filesystem("memory")
    fs.store.clear()
    fs.pseudo_dirs.clear()
    with fs.open("/csv/people.csv", "w") as handle:
        handle.write("id,name\n1,Ada\n2,Grace\n3,Alan\n")
    with fs.open("/csv/semicolons.csv", "w") as handle:
        handle.write("id;name\n1;Ada\n")
    with fs.open("/csv/ragged.csv", "w") as handle:
        handle.write("id,name,note\n1,Ada\n")
    with fs.open("/csv/headeronly.csv", "w") as handle:
        handle.write("id,name\n")
    monkeypatch.setenv("SOURCE_URI", "memory://csv")
    reset_settings()
    yield fs
    fs.store.clear()
    fs.pseudo_dirs.clear()


def test_yields_header_and_rows_as_dicts(csv_source):
    batches = list(read_csv_batches("people.csv"))
    assert len(batches) == 1
    header, rows = batches[0]
    assert header == ["id", "name"]
    assert rows == [
        {"id": "1", "name": "Ada"},
        {"id": "2", "name": "Grace"},
        {"id": "3", "name": "Alan"},
    ]


def test_splits_into_batches(csv_source):
    batches = list(read_csv_batches("people.csv", batch_size=2))
    assert [len(rows) for _header, rows in batches] == [2, 1]


def test_detects_non_comma_separator(csv_source):
    _header, rows = next(iter(read_csv_batches("semicolons.csv")))
    assert rows == [{"id": "1", "name": "Ada"}]


def test_short_row_omits_missing_column(csv_source):
    _header, rows = next(iter(read_csv_batches("ragged.csv")))
    assert rows == [{"id": "1", "name": "Ada"}]
    assert "note" not in rows[0]


def test_header_only_file_yields_nothing(csv_source):
    assert list(read_csv_batches("headeronly.csv")) == []


def test_values_stay_strings(csv_source):
    _header, rows = next(iter(read_csv_batches("people.csv")))
    assert all(isinstance(value, str) for value in rows[0].values())


def test_a_windows_1252_csv_with_a_semicolon_dialect_splits_and_decodes(csv_source):
    """make_csv_reader reads 2048 characters to sniff the dialect, then seek(0)s.
    That must still work on a cp1252 handle."""
    text = "id;city;price\n1;Luleå;€5\n2;Köln;€7\n"
    with csv_source.open("/csv/win.csv", "wb") as handle:
        handle.write(text.encode("cp1252"))
    header, rows = next(iter(read_csv_batches("win.csv")))
    assert header == ["id", "city", "price"]
    assert rows == [
        {"id": "1", "city": "Luleå", "price": "€5"},
        {"id": "2", "city": "Köln", "price": "€7"},
    ]


def test_a_utf8_bom_leaves_a_clean_first_column_name(csv_source):
    with csv_source.open("/csv/bom.csv", "wb") as handle:
        handle.write(b"\xef\xbb\xbfid,name\n1,Ada\n")
    header, rows = next(iter(read_csv_batches("bom.csv")))
    assert header == ["id", "name"]
    assert rows == [{"id": "1", "name": "Ada"}]


ODD_HEADERS = [
    "Order ID",
    "customer-id",
    "Straße",
    "we`ird",
    "line\nbreak",
    "   ",
    "C:\\users",
    "a\\u0041b",
    "x\\u0060: 1}) SET n.pwned = true //",
]


@pytest.mark.parametrize("name", ODD_HEADERS, ids=lambda name: repr(name)[:24])
def test_any_header_text_is_read_back_exactly(csv_source, name):
    """KG-51: a header the key and join checks now accept must reach them
    unchanged, a quoted line break and whitespace-only text included."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow([name, "note"])
    writer.writerows([["k1", "a"], ["k2", "b"]])
    with csv_source.open("/csv/odd.csv", "w") as handle:
        handle.write(buffer.getvalue())

    assert read_csv_header("odd.csv") == [name, "note"]
    header, rows = next(iter(read_csv_batches("odd.csv")))
    assert header == [name, "note"]
    assert rows == [{name: "k1", "note": "a"}, {name: "k2", "note": "b"}]
