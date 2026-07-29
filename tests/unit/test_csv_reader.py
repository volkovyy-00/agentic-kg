import fsspec
import pytest

from agentic_kg.common.config import reset_settings
from agentic_kg.common.csv_reader import read_csv_batches


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
