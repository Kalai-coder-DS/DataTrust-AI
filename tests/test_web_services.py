"""
tests/test_web_services.py

Phase 6 web layer WITHOUT Django: upload checks, analyse + save, the data behind every
page, and the score-history chart. Runs on SQLite like test_storage.py.
"""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

import pytest

from datatrust.config import Settings
from datatrust.storage import AnalysisRepository, Database, DBConfig, StorageError
from datatrust.web import (
    NotFound,
    UploadRejected,
    analyze_upload,
    dashboard_page,
    dataset_page,
    repository,
    reset_repository_state,
    run_page,
    stored_analysis,
    trend_svg,
    validate_upload,
)
from datatrust.web.services import clean_file_name


@pytest.fixture
def repo(tmp_path):
    db = Database(DBConfig(backend="sqlite", sqlite_path=tmp_path / "web.db")).connect()
    r = AnalysisRepository(db)
    r.init_schema()
    yield r
    db.close()


def chunks_of(path: Path, size: int = 4096):
    data = path.read_bytes()
    return [data[i:i + size] for i in range(0, len(data), size)]


def upload(repo, path: Path, name: str | None = None, target: str | None = None) -> int:
    return analyze_upload(chunks_of(path), name or path.name, repo, target=target, size=path.stat().st_size)


# ------------------------------------------------------------ file names
@pytest.mark.parametrize("raw, expected", [
    ("data.csv", "data.csv"),
    (r"C:\Users\kalai\Downloads\insurance.csv", "insurance.csv"),
    ("../../etc/passwd.csv", "passwd.csv"),
    ("bad\x00name\n.csv", "badname.csv"),
    ("  spaced.csv  ", "spaced.csv"),
])
def test_clean_file_name(raw, expected):
    assert clean_file_name(raw) == expected


@pytest.mark.parametrize("raw", ["", "/", "..", "C:\\"])
def test_clean_file_name_rejects_nameless(raw):
    with pytest.raises(UploadRejected):
        clean_file_name(raw)


def test_long_name_fits_column_and_keeps_extension():
    name = clean_file_name("x" * 400 + ".csv")
    assert len(name) == 255 and name.endswith(".csv")


# ------------------------------------------------------------ validation
def test_validate_rejects_wrong_extension():
    with pytest.raises(UploadRejected, match="not a CSV"):
        validate_upload("report.xlsx", 100)


def test_validate_rejects_empty_and_too_large():
    small = Settings(max_file_size_mb=1)
    with pytest.raises(UploadRejected, match="empty"):
        validate_upload("a.csv", 0, small)
    with pytest.raises(UploadRejected, match="limit is 1 MB"):
        validate_upload("a.csv", 2 * 1024 * 1024, small)
    assert validate_upload("A.CSV", 10, small) == "A.CSV"


def test_streamed_size_is_enforced_even_if_declared_size_lies(repo):
    small = Settings(max_file_size_mb=1)
    big = [b"a,b\n" + b"1,2\n" * 100_000] * 3          # ~1.2 MB in total
    with pytest.raises(UploadRejected, match="larger than"):
        analyze_upload(big, "big.csv", repo, size=10, settings=small)
    assert repo.list_datasets() == []


# ------------------------------------------------------------ analyse + save
def _temp_upload_dirs() -> set[str]:
    return {p for p in os.listdir(tempfile.gettempdir()) if p.startswith("datatrust-upload-")}


def test_upload_is_analysed_saved_and_cleaned_up(repo, sample_dir):
    before = _temp_upload_dirs()
    run_id = upload(repo, sample_dir / "customers_dirty.csv", target="churned")
    assert _temp_upload_dirs() == before                      # temporary copy deleted
    analysis = stored_analysis(repo, run_id)
    assert analysis["file"]["file_name"] == "customers_dirty.csv"   # original name, not the temp name
    assert analysis["meta"]["target"] == "churned"
    assert analysis["scores"]["trust"]["score"] == 79.8      # same result as the CLI (ARCHITECTURE.md)


def test_blank_target_means_no_target(repo, sample_dir):
    run_id = analyze_upload(chunks_of(sample_dir / "customers_clean.csv"), "customers_clean.csv", repo, target="   ")
    assert stored_analysis(repo, run_id)["meta"]["target"] is None


def test_unknown_target_is_a_friendly_rejection(repo, sample_dir):
    with pytest.raises(UploadRejected, match="not found. Available"):
        upload(repo, sample_dir / "customers_dirty.csv", target="no_such_column")
    assert repo.list_datasets() == []


def test_unreadable_files_are_rejected(repo):
    with pytest.raises(UploadRejected, match="empty"):
        analyze_upload([], "empty.csv", repo)
    with pytest.raises(UploadRejected):
        analyze_upload([b"only_a_header\n"], "header_only.csv", repo)


def test_storage_failure_is_not_reported_as_bad_upload(sample_dir):
    class BrokenRepo:
        def save(self, analysis):
            raise StorageError("Cannot reach the server")

    with pytest.raises(StorageError) as info:
        upload(BrokenRepo(), sample_dir / "customers_clean.csv")
    assert not isinstance(info.value, UploadRejected)


# ------------------------------------------------------------ pages
def test_dashboard_empty_and_filled(repo, sample_dir):
    assert dashboard_page(repo) == {"datasets": [], "total_datasets": 0, "total_runs": 0}
    upload(repo, sample_dir / "customers_dirty.csv", target="churned")
    upload(repo, sample_dir / "customers_dirty.csv", target="churned")
    upload(repo, sample_dir / "customers_clean.csv")
    page = dashboard_page(repo)
    assert (page["total_datasets"], page["total_runs"]) == (2, 3)
    dirty = next(d for d in page["datasets"] if d["name"] == "customers_dirty.csv")
    assert dirty["runs"] == 2
    assert dirty["trust"] == {"letter": "C", "css": "sev-medium", "meaning": "usable with fixes"}


def test_dataset_page_history_labels(repo, sample_dir, tmp_path):
    src = sample_dir / "customers_dirty.csv"
    upload(repo, src, target="churned")
    upload(repo, src, target="churned")
    # Same name, different data: drop the last 15 rows (the duplicates).
    changed = tmp_path / "customers_dirty.csv"
    changed.write_text("\n".join(src.read_text().splitlines()[:-15]) + "\n")
    upload(repo, changed, target="churned")

    page = dataset_page(repo, "customers_dirty.csv")
    newest, middle, oldest = page["runs"]                    # newest first
    assert oldest["trust_change_label"] == "first run" and oldest["data_label"] == "-"
    assert middle["trust_change_label"] == "no change" and middle["data_label"] == "same data"
    assert newest["data_label"] == "new data"
    assert page["latest"] is newest
    assert page["chart"].startswith("<svg") and page["chart"].count("<title>Run #") == 6   # 3 runs x 2 series


def test_dataset_page_not_found(repo):
    with pytest.raises(NotFound):
        dataset_page(repo, "never-uploaded.csv")


def test_run_page_content(repo, sample_dir):
    run_id = upload(repo, sample_dir / "customers_dirty.csv", target="churned")
    page = run_page(repo, run_id)
    assert page["run_id"] == run_id and page["dataset_name"] == "customers_dirty.csv"
    assert page["trust"]["score"] == 79.8 and page["trust"]["grade"]["letter"] == "C"
    assert [d["label"] for d in page["trust"]["dimensions"]] == ["Completeness", "Uniqueness", "Validity", "Consistency"]
    assert all(0 <= d["width"] <= 100 for d in page["ml"]["dimensions"])
    order = ["critical", "high", "medium", "low"]
    ranks = [order.index(i["severity"]) for i in page["issues"]]
    assert ranks == sorted(ranks)                            # most severe first
    assert sum(s["count"] for s in page["severity_summary"]) == page["total_issues"] == len(page["issues"])
    assert 0 < len(page["top_actions"]) <= 5


def test_run_page_and_analysis_not_found(repo):
    with pytest.raises(NotFound):
        run_page(repo, 999)
    with pytest.raises(NotFound):
        stored_analysis(repo, 999)


# ------------------------------------------------------------ repository per request
def test_repository_creates_schema_once_then_connects(tmp_path):
    reset_repository_state()
    env = {"DATATRUST_DB": "sqlite", "DATATRUST_SQLITE_PATH": str(tmp_path / "req.db")}
    with repository(env) as repo:
        assert repo.list_datasets() == []
        first_db = repo.db
    assert first_db._conn is None                            # closed after the request
    with repository(env) as repo:
        assert repo.db is not first_db and repo.list_datasets() == []
    reset_repository_state()


def test_repository_bad_config_raises_storage_error():
    with pytest.raises(StorageError):
        with repository({"DATATRUST_DB": "oracle"}):
            pass


# ------------------------------------------------------------ chart
def _row(run_id, trust, ml, when="2026-11-16 10:00:00"):
    return {"run_id": run_id, "analyzed_at": when, "trust_score": trust, "trust_grade": "C",
            "ml_score": ml, "ml_grade": "D"}


def test_trend_svg_empty_and_single_run():
    assert trend_svg([]) == ""
    svg = trend_svg([_row(1, 79.8, 65.0)])
    assert "<polyline" not in svg                            # one point: markers only
    assert svg.count("<circle") == 1 and svg.count("<rect") == 1
    assert "Trust 79.8" in svg and "ML Readiness 65.0" in svg


def test_trend_svg_lines_legend_encoding_and_tooltips():
    svg = trend_svg([_row(1, 70, 60), _row(2, 80, 75), _row(3, 90, 88)], "insurance.csv")
    assert svg.count("<polyline") == 2
    assert 'stroke-dasharray="6 4"' in svg                   # ML line is dashed (not colour alone)
    assert svg.count("<title>Run #") == 6
    assert "var(--series-1)" in svg and "var(--series-2)" in svg
    assert "Score history for insurance.csv" in svg


def test_trend_svg_escapes_dataset_name():
    svg = trend_svg([_row(1, 50, 40)], '<script>alert("x")</script>.csv')
    assert "<script>" not in svg and "&lt;script&gt;" in svg


def test_trend_svg_direct_labels_do_not_overlap_when_scores_are_equal():
    svg = trend_svg([_row(1, 80, 80)])
    ys = [float(y) for y in re.findall(r'class="direct" x="[\d.]+" y="([\d.]+)"', svg)]
    assert len(ys) == 2 and abs(ys[0] - ys[1]) >= 13.9   # 14px apart (SVG coords rounded to 0.1)


def test_trend_svg_clamps_out_of_range_scores():
    svg = trend_svg([_row(1, 120, -5), _row(2, 50, 50)])
    ys = [float(y) for y in re.findall(r'cy="([\d.]+)"', svg)]
    assert all(16 <= y <= 240 - 36 for y in ys)
