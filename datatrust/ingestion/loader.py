"""
datatrust/ingestion/loader.py

Safe CSV ingestion.

Responsibilities (and ONLY these - single responsibility principle):
  1. Validate the file: exists, is a file, allowed extension, not empty, not too large.
  2. Detect text encoding (utf-8 / utf-8 with BOM / Windows cp1252 / latin-1).
  3. Detect the delimiter (, ; tab |).
  4. Parse into a DataFrame, recovering from malformed rows instead of crashing.
  5. Report structural problems in the header (duplicate / blank / padded names).

What it deliberately does NOT do: fix data values. A data-quality tool must see the
data as it really is, so cleaning is never silently applied at load time.
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from datatrust.config import DEFAULT_SETTINGS, Settings
from datatrust.exceptions import DataLoadError, EmptyDatasetError, FileValidationError
from datatrust.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LoadResult:
    """Everything the rest of the pipeline needs to know about a loaded file."""

    df: pd.DataFrame
    file_name: str
    file_size_bytes: int
    encoding: str
    delimiter: str
    malformed_rows_skipped: int = 0
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        """JSON-friendly metadata (excludes the DataFrame itself)."""
        return {
            "file_name": self.file_name,
            "file_size_bytes": self.file_size_bytes,
            "encoding": self.encoding,
            "delimiter": self.delimiter,
            "rows": int(self.df.shape[0]),
            "columns": int(self.df.shape[1]),
            "malformed_rows_skipped": self.malformed_rows_skipped,
            "warnings": list(self.warnings),
        }


class CSVLoader:
    """Load a CSV file defensively. Usage: CSVLoader().load("data.csv")"""

    def __init__(self, settings: Settings = DEFAULT_SETTINGS) -> None:
        self.settings = settings

    # ------------------------------------------------------------------ public
    def load(self, path: str | Path) -> LoadResult:
        path = Path(path)
        logger.info("Loading file: %s", path)

        file_size = self._validate_file(path)
        raw_sample = self._read_sample_bytes(path)
        encoding = self._detect_encoding(path)
        text_sample = raw_sample.decode(encoding, errors="replace")
        delimiter = self._detect_delimiter(text_sample)
        logger.info("Detected encoding=%s delimiter=%r", encoding, delimiter)

        warnings: list[str] = []
        warnings += self._check_header(text_sample, delimiter)

        df, skipped = self._parse(path, encoding, delimiter)
        if skipped:
            warnings.append(
                f"{skipped} malformed row(s) had the wrong number of fields and were skipped."
            )

        df = self._clean_column_names(df, warnings)
        self._validate_dataframe(df)

        for w in warnings:
            logger.warning(w)
        logger.info("Loaded %d rows x %d columns from %s", df.shape[0], df.shape[1], path.name)

        return LoadResult(
            df=df,
            file_name=path.name,
            file_size_bytes=file_size,
            encoding=encoding,
            delimiter=delimiter,
            malformed_rows_skipped=skipped,
            warnings=warnings,
        )

    # -------------------------------------------------------------- validation
    def _validate_file(self, path: Path) -> int:
        """Checks that do not require reading content. Returns file size in bytes."""
        if not path.exists():
            raise FileValidationError(f"File not found: {path}")
        if not path.is_file():
            raise FileValidationError(f"Not a file: {path}")
        if path.suffix.lower() not in self.settings.allowed_extensions:
            raise FileValidationError(
                f"Unsupported file type '{path.suffix}'. "
                f"Allowed: {', '.join(self.settings.allowed_extensions)}"
            )
        size = path.stat().st_size
        if size == 0:
            raise FileValidationError(f"File is empty (0 bytes): {path.name}")
        if size > self.settings.max_file_size_bytes:
            raise FileValidationError(
                f"File is {size / 1024 / 1024:.1f} MB; "
                f"limit is {self.settings.max_file_size_mb} MB."
            )
        return size

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        if df.shape[1] == 0:
            raise EmptyDatasetError("No columns found in the file.")
        if df.shape[0] == 0:
            raise EmptyDatasetError("The file has a header but no data rows.")
        if df.isna().all(axis=None):
            raise EmptyDatasetError("Every cell in the dataset is empty.")

    # --------------------------------------------------------------- detection
    def _read_sample_bytes(self, path: Path) -> bytes:
        with path.open("rb") as fh:
            return fh.read(self.settings.sniff_sample_bytes)

    def _detect_encoding(self, path: Path) -> str:
        """
        Try each candidate encoding on the WHOLE file (streamed in chunks).
        Checking only a sample can miss a bad byte near the end of the file.
        A UTF-8 byte-order mark (BOM, common in Excel exports) is checked first so it
        is stripped from the first column name and reported accurately.
        """
        with path.open("rb") as fh:
            has_bom = fh.read(3) == b"\xef\xbb\xbf"
        candidates = ("utf-8-sig",) if has_bom else tuple(
            e for e in self.settings.encodings_to_try if e != "utf-8-sig"
        )
        for enc in candidates:
            try:
                with path.open("r", encoding=enc) as fh:
                    while fh.read(1024 * 1024):
                        pass
                return enc
            except (UnicodeDecodeError, UnicodeError):
                continue
        # latin-1 decodes any byte sequence, so this is reached only if it was removed from config.
        raise DataLoadError("Could not determine the file's text encoding.")

    def _detect_delimiter(self, text_sample: str) -> str:
        """Use csv.Sniffer on the first lines; fall back to comma."""
        lines = text_sample.splitlines()[:50]
        if len(lines) > 1 and text_sample and not text_sample.endswith(("\n", "\r")):
            lines = lines[:-1]  # last line may be cut mid-row by the byte sample
        sample = "\n".join(lines)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=self.settings.candidate_delimiters)
            return dialect.delimiter
        except csv.Error:
            logger.debug("Delimiter sniffing failed; defaulting to ','")
            return ","

    def _check_header(self, text_sample: str, delimiter: str) -> list[str]:
        """
        Inspect the raw header line. pandas silently renames duplicates
        ('age', 'age' -> 'age', 'age.1'), so we must look before pandas does.
        """
        warnings: list[str] = []
        try:
            header = next(csv.reader(io.StringIO(text_sample), delimiter=delimiter))
        except StopIteration:
            return warnings

        stripped = [h.strip() for h in header]
        seen: set[str] = set()
        dupes: list[str] = []
        for name in stripped:
            if name and name in seen and name not in dupes:
                dupes.append(name)
            seen.add(name)
        if dupes:
            warnings.append(f"Duplicate column names in header: {dupes}. Pandas renamed them with suffixes (.1, .2).")
        blanks = sum(1 for h in stripped if h == "")
        if blanks:
            warnings.append(f"{blanks} column(s) have a blank header name.")
        return warnings

    # ----------------------------------------------------------------- parsing
    def _parse(self, path: Path, encoding: str, delimiter: str) -> tuple[pd.DataFrame, int]:
        """
        Fast path: pandas C engine. If rows have inconsistent field counts the C
        engine raises ParserError; we then retry with the Python engine and a
        callback that counts (and skips) the bad rows, so one broken line does
        not make the whole upload fail.
        """
        common = dict(sep=delimiter, encoding=encoding, skipinitialspace=False)
        try:
            df = pd.read_csv(path, low_memory=False, **common)
            return df, 0
        except pd.errors.EmptyDataError as exc:
            raise EmptyDatasetError("The file contains no parsable data.") from exc
        except pd.errors.ParserError as exc:
            logger.warning("C parser failed (%s). Retrying with tolerant parser.", exc)

        bad_rows: list[list[str]] = []

        def _on_bad_line(fields: list[str]) -> None:
            bad_rows.append(fields)
            return None  # returning None tells pandas to skip the row

        try:
            df = pd.read_csv(path, engine="python", on_bad_lines=_on_bad_line, **common)
        except (pd.errors.ParserError, ValueError) as exc:
            raise DataLoadError(f"Could not parse CSV: {exc}") from exc
        return df, len(bad_rows)

    @staticmethod
    def _clean_column_names(df: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
        """Strip surrounding whitespace from header names (a common Excel export artefact)."""
        original = [str(c) for c in df.columns]
        cleaned = [c.strip() for c in original]
        changed = [o for o, c in zip(original, cleaned) if o != c]
        if changed and len(set(cleaned)) < len(cleaned):
            # e.g. 'age' and ' age' would collide after stripping; keep originals.
            warnings.append("Some column names differ only by surrounding spaces; names left unchanged.")
            return df
        if changed:
            warnings.append(f"Stripped leading/trailing spaces from {len(changed)} column name(s).")
            df = df.rename(columns=dict(zip(original, cleaned)))
        return df
