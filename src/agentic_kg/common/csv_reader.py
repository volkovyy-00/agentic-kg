"""Read source CSVs into batches of row dictionaries.

Kept separate from anything that talks to Neo4j so that batching can be tested
without a database. Values stay strings, exactly as LOAD CSV produced them —
typed fields are deliberately out of scope.
"""
import logging
from typing import Iterator, List, Tuple

import clevercsv

from .file_source import open_source

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 1000
_SNIFF_BYTES = 2048


def make_csv_reader(handle, relative_path: str):
    """Build a clevercsv reader, sniffing the dialect where possible."""
    sample = handle.read(_SNIFF_BYTES)
    handle.seek(0)
    dialect = None
    try:
        dialect = clevercsv.Sniffer().sniff(sample)
    except clevercsv.Error:
        logger.warning("Could not sniff CSV dialect for %s; using default", relative_path)
    # sniff() returns a degenerate SimpleDialect('', '', '') for empty or
    # trivial samples rather than raising, so the except clause above does not
    # fire for those. Check the delimiter explicitly.
    if dialect is None or not getattr(dialect, "delimiter", ""):
        logger.warning(
            "Could not determine a CSV delimiter for %s (degenerate dialect); using default",
            relative_path,
        )
        return clevercsv.reader(handle)
    return clevercsv.reader(handle, dialect)


def read_csv_batches(
    relative_path: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Iterator[Tuple[List[str], List[dict]]]:
    """Yield (header, batch) pairs for a source-relative CSV file.

    Rows shorter than the header omit the missing keys rather than padding with
    empty strings, which keeps the resulting graph free of meaningless blanks.

    Args:
        relative_path: file name relative to the source root
        batch_size: rows per batch

    Yields:
        (header, rows) where rows is a list of dicts of column name to string
    """
    with open_source(relative_path, "r") as handle:
        reader = make_csv_reader(handle, relative_path)
        header = next(reader, [])
        if not header:
            return
        batch: List[dict] = []
        for row in reader:
            batch.append({key: value for key, value in zip(header, row)})
            if len(batch) >= batch_size:
                yield header, batch
                batch = []
        if batch:
            yield header, batch
