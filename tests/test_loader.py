"""
tests/test_loader.py

Unit tests for datatrust.ingestion.loader.CSVLoader.
Each test targets one behaviour: a valid load, a rejection, or a recovery.
"""
from __future__ import annotations

import json

import pytest

from datatrust.config import Settings
from datatrust.exceptions import EmptyDatasetError, FileValidationError
from datatrust.ingestion import CSVLoader


@pytest.fixture
def loader() -> CSVLoader:
    return CSVLoader()


# ----------------------------------------------------------------- happy path
def test_loads_clean_sample(loader, sample_dir):
    result = loader.load(sample_dir / "customers_clean.csv")
    assert result.df.shape == (500, 11)
    assert result.encoding == "utf-8"
    assert result.delimiter == ","
    assert result.malformed_rows_skipped == 0
    assert result.warnings == []


def test_loads_dirty_sample_with_expected_shape(loader, sample_dir, dirty_manifest):
    result = loader.load(sample_dir / "customers_dirty.csv")
    assert result.df.shape == (dirty_manifest["total_rows"], 12)


def test_accepts_uppercase_extension(loader, write_csv):
    path = write_csv("a,b\n1,2\n", name="DATA.CSV")
    assert loader.load(path).df.shape == (1, 2)


def test_quoted_field_containing_delimiter(loader, write_csv):
    path = write_csv('name,city\n"Kumar, R",Chennai\nAsha,Pune\n')
    df = loader.load(path).df
    assert df.loc[0, "name"] == "Kumar, R"
    assert df.shape == (2, 2)


@pytest.mark.parametrize("delim", [";", "\t", "|"])
def test_detects_alternative_delimiters(loader, write_csv, delim):
    content = delim.join(["id", "name", "score"]) + "\n" + "\n".join(
        delim.join([str(i), f"user{i}", str(i * 10)]) for i in range(1, 6)
    ) + "\n"
    result = loader.load(write_csv(content))
    assert result.delimiter == delim
    assert list(result.df.columns) == ["id", "name", "score"]
    assert result.df.shape == (5, 3)


# ------------------------------------------------------------------ encodings
def test_detects_windows_cp1252_encoding(loader, write_csv):
    path = write_csv("name,city\nRené,Chennai\nZoë,Pune\n", encoding="cp1252")
    result = loader.load(path)
    assert result.encoding == "cp1252"
    assert result.df.loc[0, "name"] == "René"


def test_utf8_bom_is_stripped_from_first_column(loader, write_csv):
    path = write_csv("﻿id,name\n1,Asha\n", encoding="utf-8")
    result = loader.load(path)
    assert result.encoding == "utf-8-sig"
    assert list(result.df.columns) == ["id", "name"]


def test_utf8_with_tamil_text(loader, write_csv):
    path = write_csv("id,name\n1,கலை\n2,Asha\n")
    result = loader.load(path)
    assert result.encoding == "utf-8"
    assert result.df.loc[0, "name"] == "கலை"


# ------------------------------------------------------------- file rejection
def test_missing_file_raises(loader, tmp_path):
    with pytest.raises(FileValidationError, match="not found"):
        loader.load(tmp_path / "nope.csv")


def test_directory_raises(loader, tmp_path):
    folder = tmp_path / "folder.csv"
    folder.mkdir()
    with pytest.raises(FileValidationError, match="Not a file"):
        loader.load(folder)


def test_wrong_extension_raises(loader, write_csv):
    with pytest.raises(FileValidationError, match="Unsupported file type"):
        loader.load(write_csv("a,b\n1,2\n", name="data.xlsx"))


def test_zero_byte_file_raises(loader, write_csv):
    with pytest.raises(FileValidationError, match="empty"):
        loader.load(write_csv(b""))


def test_file_over_size_limit_raises(write_csv):
    tiny_limit = CSVLoader(Settings(max_file_size_mb=0))
    with pytest.raises(FileValidationError, match="limit"):
        tiny_limit.load(write_csv("a,b\n1,2\n"))


# ----------------------------------------------------------- empty datasets
def test_header_only_raises(loader, write_csv):
    with pytest.raises(EmptyDatasetError, match="no data rows"):
        loader.load(write_csv("a,b,c\n"))


def test_blank_lines_only_raises(loader, write_csv):
    with pytest.raises(EmptyDatasetError):
        loader.load(write_csv("\n\n\n"))


def test_all_cells_empty_raises(loader, write_csv):
    with pytest.raises(EmptyDatasetError, match="Every cell"):
        loader.load(write_csv("a,b\n,\n,\n"))


# ------------------------------------------------------ structural warnings
def test_malformed_rows_are_skipped_and_counted(loader, write_csv):
    content = "a,b,c\n1,2,3\n4,5,6,7,8\n9,10,11\n"  # row 2 has too many fields
    result = loader.load(write_csv(content))
    assert result.malformed_rows_skipped == 1
    assert result.df.shape == (2, 3)
    assert any("malformed" in w for w in result.warnings)


def test_duplicate_header_names_warn(loader, write_csv):
    result = loader.load(write_csv("id,age,age\n1,30,31\n"))
    assert any("Duplicate column names" in w for w in result.warnings)
    assert list(result.df.columns) == ["id", "age", "age.1"]


def test_blank_header_name_warns(loader, write_csv):
    result = loader.load(write_csv("id,,score\n1,x,5\n"))
    assert any("blank header" in w for w in result.warnings)


def test_column_name_whitespace_is_stripped(loader, write_csv):
    result = loader.load(write_csv(" id , name \n1,Asha\n"))
    assert list(result.df.columns) == ["id", "name"]
    assert any("Stripped" in w for w in result.warnings)


def test_colliding_names_after_strip_are_left_unchanged(loader, write_csv):
    result = loader.load(write_csv("age, age\n1,2\n"))
    assert list(result.df.columns) == ["age", " age"]


# --------------------------------------------------------------- data integrity
def test_loader_does_not_alter_values(loader, write_csv):
    """A quality tool must see raw data: 'abc' in a numeric column must survive loading."""
    df = loader.load(write_csv("age\n25\nabc\n40\n")).df
    assert "abc" in df["age"].astype(str).tolist()


def test_summary_is_json_serialisable(loader, sample_dir):
    summary = loader.load(sample_dir / "customers_dirty.csv").summary()
    json.dumps(summary)
    assert summary["rows"] == 515
