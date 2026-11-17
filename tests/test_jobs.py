"""
tests/test_jobs.py

Phase 10 background analyses, without Django: real threads, SQLite through datatrust.storage.
"""
from __future__ import annotations

import os
import tempfile
import threading

import pytest

from datatrust.config import Settings
from datatrust.storage import AnalysisRepository, Database, DBConfig, StorageError
from datatrust.web import UploadRejected, reset_repository_state
from datatrust.web.jobs import MAX_KEPT, JobManager
from datatrust.web.services import start_upload_job


@pytest.fixture
def manager():
    m = JobManager(workers=2)
    yield m
    m.shutdown()


@pytest.fixture
def env(tmp_path):
    reset_repository_state()
    yield {"DATATRUST_DB": "sqlite", "DATATRUST_SQLITE_PATH": str(tmp_path / "jobs.db")}
    reset_repository_state()


def stored_runs(env) -> list[dict]:
    db = Database(DBConfig.from_env(env)).connect()
    try:
        return AnalysisRepository(db).list_datasets()
    finally:
        db.close()


# ------------------------------------------------------------------ manager
def test_states_from_queued_to_done(manager):
    release = threading.Event()
    job_id = manager.submit(lambda: release.wait(5) and 42, label="a.csv")
    assert len(job_id) == 32 and manager.get(job_id)["state"] in ("queued", "running")
    release.set()
    job = manager.wait(job_id, 5)
    assert (job["state"], job["run_id"], job["error"]) == ("done", 42, None)
    assert job["elapsed"] >= 0


@pytest.mark.parametrize("error, kind, text", [
    (UploadRejected("Target column 'x' not found."), "user", "Target column 'x' not found."),
    (StorageError("Cannot reach the server."), "database", "Cannot reach the server. Check the database settings"),
    (RuntimeError("boom with secret path C:/x"), "internal", "unexpected error"),
])
def test_errors_are_classified(manager, error, kind, text):
    def fail():
        raise error
    job = manager.wait(manager.submit(fail, label="bad.csv"), 5)
    assert job["state"] == "failed" and job["error_kind"] == kind and text in job["error"]
    assert "boom" not in job["error"]                     # internal details stay in the log


def test_unknown_job(manager):
    assert manager.get("0" * 32) is None and manager.wait("nope", 0.1) is None


def test_old_jobs_are_forgotten(manager):
    ids = [manager.submit(lambda: 1, label=f"{i}.csv") for i in range(MAX_KEPT + 5)]
    manager.wait(ids[-1], 10)
    assert manager.get(ids[0]) is None and manager.get(ids[-1]) is not None


def test_two_jobs_run_at_the_same_time(manager):
    both = threading.Barrier(2, timeout=5)
    ids = [manager.submit(lambda: both.wait() + 1, label=f"{i}.csv") for i in range(2)]
    assert all(manager.wait(i, 10)["state"] == "done" for i in ids)


# ------------------------------------------------------------------ uploads
def test_upload_job_end_to_end(manager, env, sample_dir):
    data = (sample_dir / "customers_dirty.csv").read_bytes()
    job_id = start_upload_job(manager, [data], "customers_dirty.csv", target="churned", size=len(data),
                              summary="template", environ=env)
    job = manager.wait(job_id, 60)
    assert job["state"] == "done", job
    runs = stored_runs(env)
    assert runs[0]["name"] == "customers_dirty.csv" and runs[0]["runs"] == 1


def test_fast_checks_happen_before_the_job(manager, env):
    with pytest.raises(UploadRejected, match="not a CSV"):
        start_upload_job(manager, [b"a,b\n1,2\n"], "data.xlsx", environ=env)
    with pytest.raises(UploadRejected, match="empty"):
        start_upload_job(manager, [], "data.csv", environ=env)
    with pytest.raises(UploadRejected, match="larger than"):
        start_upload_job(manager, [b"a,b\n" + b"1,2\n" * 400_000], "big.csv", environ=env,
                         settings=Settings(max_file_size_mb=1))
    assert manager._jobs == {}


def test_bad_target_fails_inside_the_job(manager, env, sample_dir):
    data = (sample_dir / "customers_clean.csv").read_bytes()
    job = manager.wait(start_upload_job(manager, [data], "c.csv", target="nope", environ=env), 30)
    assert job["state"] == "failed" and job["error_kind"] == "user" and "not found. Available" in job["error"]


def test_staged_files_are_removed(manager, env, sample_dir):
    def leftovers():
        return {p for p in os.listdir(tempfile.gettempdir()) if p.startswith("datatrust-job-")}
    before = leftovers()
    data = (sample_dir / "customers_clean.csv").read_bytes()
    manager.wait(start_upload_job(manager, [data], "c.csv", environ=env), 30)
    manager.wait(start_upload_job(manager, [data], "c.csv", target="nope", environ=env), 30)
    with pytest.raises(UploadRejected):
        start_upload_job(manager, [], "e.csv", environ=env)
    assert leftovers() == before


def test_database_down_fails_the_job_not_the_server(manager, sample_dir):
    reset_repository_state()
    data = (sample_dir / "customers_clean.csv").read_bytes()
    job = manager.wait(start_upload_job(manager, [data], "c.csv", environ={"DATATRUST_DB": "oracle"}), 30)
    assert job["state"] == "failed" and job["error_kind"] == "database"
