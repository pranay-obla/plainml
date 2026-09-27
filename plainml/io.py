"""Reading and writing tabular data in the formats plainml supports."""

from __future__ import annotations

import csv
import hashlib
import sqlite3
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from plainml.errors import PlainMLError, did_you_mean, is_installed, require

READERS = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".tab": "tsv",
    ".txt": "text",
    ".xlsx": "excel",
    ".xlsm": "excel",
    ".xls": "excel",
    ".parquet": "parquet",
    ".pq": "parquet",
    ".json": "json",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".feather": "feather",
}

SQL_SCHEMES = ("sqlite", "postgresql", "postgres", "mysql", "mariadb", "mssql", "oracle", "duckdb")

# Files bigger than this are read with polars when it's installed (it's several times faster).
POLARS_THRESHOLD_BYTES = 200 * 1024 * 1024

SUPPORTED_HELP = "CSV, TSV, Excel, Parquet, JSON/JSONL, Feather, an http(s) URL, or a SQL URL"


def is_sql_url(source: str) -> bool:
    scheme = urlparse(source).scheme.split("+")[0].lower()
    return scheme in SQL_SCHEMES


def _is_url(source: str) -> bool:
    return urlparse(source).scheme in ("http", "https", "s3", "gs")


def _format_for(source: str) -> str:
    path = urlparse(source).path if _is_url(source) else source
    suffixes = [s.lower() for s in Path(path).suffixes]
    if suffixes and suffixes[-1] in (".gz", ".zip", ".bz2", ".xz") and len(suffixes) > 1:
        suffix = suffixes[-2]
    else:
        suffix = suffixes[-1] if suffixes else ""
    if suffix in READERS:
        return READERS[suffix]
    if _is_url(source) and not suffix:
        return "csv"
    raise PlainMLError(
        f"Don't know how to read '{source}' (unrecognised extension '{suffix or 'none'}').",
        hint=f"Supported: {SUPPORTED_HELP}.",
    )


# UTF-8 with or without a byte-order mark (Excel's "CSV UTF-8"), then the Windows and
# Latin-1 encodings that older Excel exports use. Latin-1 accepts any byte, so it's last.
ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")


def _read_delimited(source: str, **kwargs: Any) -> pd.DataFrame:
    error: UnicodeDecodeError | None = None
    for encoding in ENCODINGS:
        try:
            return pd.read_csv(source, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            error = exc
    raise error  # type: ignore[misc]  # pragma: no cover - latin-1 decodes everything


def _read_excel(source: str, sheet: str | int | None) -> pd.DataFrame:
    """Read one sheet. Without --sheet, a workbook's largest sheet is used (and named)."""
    from plainml.console import esc, warn

    workbook = pd.ExcelFile(source)
    names = workbook.sheet_names
    if sheet is not None:
        if isinstance(sheet, str) and sheet not in names:
            raise PlainMLError(
                f"There's no sheet called '{sheet}'.{did_you_mean(sheet, names)}",
                hint="Sheets: " + ", ".join(map(str, names)),
            )
        return workbook.parse(sheet)
    if len(names) == 1:
        return workbook.parse(names[0])
    sheets = {name: workbook.parse(name) for name in names}
    filled = {name: frame for name, frame in sheets.items() if not frame.empty}
    if not filled:
        return sheets[names[0]]
    chosen = max(filled, key=lambda name: filled[name].size)
    warn(
        esc(
            f"This workbook has {len(names)} sheets ({', '.join(map(str, names))}); "
            f"using '{chosen}', the largest. Pick another with --sheet NAME."
        )
    )
    return filled[chosen]


def _sniff_delimiter(path: str) -> str:
    try:
        with open(path, newline="", encoding="utf-8", errors="replace") as handle:
            sample = handle.read(64 * 1024)
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except (csv.Error, OSError):
        return ","


def _read_sql(source: str, query: str | None, table: str | None) -> pd.DataFrame:
    if not query and not table:
        raise PlainMLError(
            "Reading from a database needs a query or a table name.",
            hint='Add --query "SELECT * FROM customers" or --table customers.',
        )
    sql = query or f'SELECT * FROM "{table}"'
    parsed = urlparse(source)
    if parsed.scheme == "sqlite" and not is_installed("sqlalchemy"):
        path = source[len("sqlite:///") :] if source.startswith("sqlite:///") else parsed.path
        if not Path(path).exists():
            raise PlainMLError(f"SQLite database '{path}' does not exist.")
        with sqlite3.connect(path) as connection:
            return pd.read_sql_query(sql, connection)
    sqlalchemy = require("sqlalchemy", "Reading from this database")
    engine = sqlalchemy.create_engine(source)
    try:
        with engine.connect() as connection:
            return pd.read_sql_query(sqlalchemy.text(sql), connection)
    finally:
        engine.dispose()


def _read_file(source: str, fmt: str, sheet: str | int | None, engine: str) -> pd.DataFrame:
    local = not _is_url(source)
    use_polars = engine == "polars" or (
        engine == "auto"
        and local
        and fmt in ("csv", "tsv", "parquet")
        and Path(source).stat().st_size > POLARS_THRESHOLD_BYTES
        and is_installed("polars")
        and is_installed("pyarrow")
    )
    if use_polars:
        pl = require("polars", "The polars engine")
        require("pyarrow", "The polars engine")
        if fmt == "parquet":
            return pl.read_parquet(source).to_pandas()
        separator = "\t" if fmt == "tsv" else ","
        return pl.read_csv(source, separator=separator, infer_schema_length=10000).to_pandas()

    if fmt == "csv":
        df = _read_delimited(source)
        if df.shape[1] == 1 and local:
            # Semicolon- or pipe-separated files saved with a .csv extension (common in Europe).
            delimiter = _sniff_delimiter(source)
            if delimiter != ",":
                df = _read_delimited(source, sep=delimiter)
        return df
    if fmt == "tsv":
        return _read_delimited(source, sep="\t")
    if fmt == "text":
        return _read_delimited(source, sep=_sniff_delimiter(source) if local else ",")
    if fmt == "excel":
        return _read_excel(source, sheet)
    if fmt == "parquet":
        if not (is_installed("pyarrow") or is_installed("fastparquet")):
            require("pyarrow", "Reading Parquet files")
        return pd.read_parquet(source)
    if fmt == "feather":
        require("pyarrow", "Reading Feather files")
        return pd.read_feather(source)
    if fmt == "json":
        try:
            return pd.read_json(source)
        except ValueError:
            return pd.read_json(source, lines=True)
    if fmt == "jsonl":
        return pd.read_json(source, lines=True)
    raise AssertionError(fmt)  # pragma: no cover


def load_data(
    source: str | Path | pd.DataFrame,
    *,
    sheet: str | int | None = None,
    query: str | None = None,
    table: str | None = None,
    sample: int | float | None = None,
    engine: str = "auto",
    seed: int = 42,
) -> pd.DataFrame:
    """Load a dataset from a file path, URL, SQL URL, or pass a DataFrame through.

    ``sample`` keeps a random subset: an int is a row count, a float in (0, 1) a fraction.
    ``engine`` is "auto", "pandas" or "polars".
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        source = str(source)
        if is_sql_url(source):
            df = _read_sql(source, query, table)
        else:
            if not _is_url(source):
                path = Path(source).expanduser()
                if not path.exists():
                    raise PlainMLError(
                        f"File '{source}' does not exist.",
                        hint=_nearby_files_hint(path),
                    )
                if path.is_dir():
                    raise PlainMLError(f"'{source}' is a folder; point to a data file inside it.")
                source = str(path)
            fmt = _format_for(source)
            try:
                df = _read_file(source, fmt, sheet, engine)
            except PlainMLError:
                raise
            except (pd.errors.EmptyDataError, pd.errors.ParserError, ValueError, OSError) as exc:
                raise PlainMLError(f"Couldn't read '{source}': {exc}") from exc
            except ImportError as exc:
                raise PlainMLError(f"Couldn't read '{source}': {exc}") from exc

    if df.empty or df.shape[1] == 0:
        raise PlainMLError("The dataset is empty (no rows or no columns).")
    df.columns = [str(c) for c in df.columns]
    duplicated = df.columns[df.columns.duplicated()].tolist()
    if duplicated:
        raise PlainMLError(
            f"The dataset has duplicate column names: {', '.join(sorted(set(duplicated)))}.",
            hint="Rename the columns so each name is unique.",
        )

    if sample:
        n = int(sample * len(df)) if isinstance(sample, float) and sample < 1 else int(sample)
        if 0 < n < len(df):
            df = df.sample(n=n, random_state=seed).sort_index()
    return df


def _nearby_files_hint(path: Path) -> str | None:
    folder = path.parent if path.parent.exists() else Path.cwd()
    candidates = [p.name for p in folder.iterdir() if p.suffix.lower() in READERS][:50]
    if not candidates:
        return None
    from plainml.errors import did_you_mean

    suggestion = did_you_mean(path.name, candidates)
    if suggestion:
        return suggestion.strip()
    return f"Data files in {folder}: {', '.join(sorted(candidates)[:10])}"


def describe_source(source: str | Path | pd.DataFrame) -> str:
    if isinstance(source, pd.DataFrame):
        return "<DataFrame>"
    source = str(source)
    if is_sql_url(source):
        parsed = urlparse(source)
        return f"{parsed.scheme}://{parsed.hostname or ''}{parsed.path}"  # never echo passwords
    if _is_url(source):
        return source
    return str(Path(source).expanduser().resolve())


def source_stem(source: str | Path | pd.DataFrame) -> str:
    if isinstance(source, pd.DataFrame):
        return "dataframe"
    source = str(source)
    if is_sql_url(source):
        return "database"
    path = urlparse(source).path if _is_url(source) else source
    stem = Path(path).name.split(".")[0]
    return stem or "data"


def save_table(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    """Write a DataFrame, choosing the format from the file extension (default CSV)."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xlsm"):
        df.to_excel(path, index=index)
    elif suffix in (".parquet", ".pq"):
        require("pyarrow", "Writing Parquet files")
        df.to_parquet(path, index=index)
    elif suffix == ".json":
        df.to_json(path, orient="records", indent=2, date_format="iso")
    elif suffix in (".jsonl", ".ndjson"):
        df.to_json(path, orient="records", lines=True, date_format="iso")
    elif suffix in (".tsv", ".tab"):
        df.to_csv(path, sep="\t", index=index)
    else:
        df.to_csv(path, index=index)
    return path


def fingerprint(df: pd.DataFrame) -> str:
    """A stable hash of the data, stored with each run so reruns can detect changed data."""
    digest = hashlib.sha256()
    digest.update(",".join(map(str, df.columns)).encode())
    digest.update(pd.util.hash_pandas_object(df, index=False).values.tobytes())
    return digest.hexdigest()[:16]
