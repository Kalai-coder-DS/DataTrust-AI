"""
tests/test_storage.py

Phase 5b storage, tested on SQLite (built into Python, no server needed). The MySQL
backend runs the same repository code with `%s` placeholders; it is verified on a real
MySQL server by the steps in README.md.
"""
from __future__ import annotations

import pandas as pd
import pytest

from datatrust.pipeline import analyze_file, file_sha256
from datatrust.reporting import render_html
from datatrust.storage import AnalysisRepository, Database, DBConfig, StorageError, load_env_file, open_repository


@pytest.fixture
def repo(tmp_path):
    db = Database(DBConfig(backend="sqlite", sqlite_path=tmp_path / "test.db")).connect()
    r = AnalysisRepository(db)
    r.init_schema()
    yield r
    db.close()


@pytest.fixture(scope="module")
def dirty(sample_dir):
    return analyze_file(sample_dir / "customers_dirty.csv", target="churned")


# ------------------------------------------------------------ configuration
def test_env_file_parsing(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text('# comment\nDATATRUST_DB=mysql\nDATATRUST_MYSQL_PASSWORD="p@ss=word"\n\nBROKEN LINE\n')
    env = {}
    load_env_file(env_file, environ=env)
    assert env == {"DATATRUST_DB": "mysql", "DATATRUST_MYSQL_PASSWORD": "p@ss=word"}


def test_real_environment_wins_over_env_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("DATATRUST_DB=mysql\n")
    env = {"DATATRUST_DB": "sqlite"}
    load_env_file(env_file, environ=env)
    assert env["DATATRUST_DB"] == "sqlite"


def test_config_from_env():
    c = DBConfig.from_env({"DATATRUST_DB": "MySQL", "DATATRUST_MYSQL_PORT": "3307",
                           "DATATRUST_MYSQL_PASSWORD": "secret"})
    assert (c.backend, c.port, c.password) == ("mysql", 3307, "secret")


def test_password_never_shown():
    c = DBConfig(backend="mysql", password="secret")
    assert "secret" not in repr(c)
    assert "secret" not in c.describe()


@pytest.mark.parametrize("env", [{"DATATRUST_DB": "oracle"}, {"DATATRUST_MYSQL_PORT": "abc"}])
def test_invalid_config_rejected(env):
    with pytest.raises(StorageError):
        DBConfig.from_env(env)


def test_mysql_without_pymysql_gives_clear_message(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pymysql":
            raise ImportError("no pymysql")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(StorageError, match="pip install PyMySQL"):
        Database(DBConfig(backend="mysql")).connect()


def test_mysql_uses_percent_placeholders():
    db = Database(DBConfig(backend="mysql"))
    assert db._sql("SELECT * FROM t WHERE a = ? AND b = ?") == "SELECT * FROM t WHERE a = %s AND b = %s"


def test_unsafe_mysql_database_name_rejected():
    db = Database(DBConfig(backend="mysql", database="x; DROP DATABASE y"))
    with pytest.raises(StorageError, match="Unsafe database name"):
        db.create_mysql_database()


# ------------------------------------------------------------- repository
def test_schema_is_idempotent(repo):
    repo.init_schema()  # running it twice must not fail
    tables = {r["name"] for r in repo.db.query("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"datasets", "analysis_runs", "issues"} <= tables


def test_save_and_read_back(repo, dirty):
    run_id = repo.save(dirty)
    stored = repo.get_analysis(run_id)
    assert stored["scores"]["trust"]["score"] == dirty["scores"]["trust"]["score"]
    assert stored["file"]["sha256"] == dirty["file"]["sha256"]
    assert len(repo.issues(run_id)) == dirty["quality"]["total_issues"]


def test_stored_run_rerenders_identically(repo, dirty):
    run_id = repo.save(dirty)
    assert render_html(repo.get_analysis(run_id)) == render_html(dirty)


def test_dataset_created_once_and_history_ordered(repo, dirty):
    ids = [repo.save(dirty) for _ in range(3)]
    assert len(repo.db.query("SELECT * FROM datasets")) == 1
    history = repo.history("customers_dirty.csv")
    assert [h["run_id"] for h in history] == ids
    assert history[0]["trust_change"] is None
    assert history[1]["trust_change"] == 0.0
    assert history[1]["data_changed"] is False


def test_history_detects_changed_data(repo, dirty, sample_dir, tmp_path):
    repo.save(dirty)
    cleaned = tmp_path / "customers_dirty.csv"   # same name, different content
    pd.read_csv(sample_dir / "customers_dirty.csv").drop_duplicates().to_csv(cleaned, index=False)
    repo.save(analyze_file(cleaned, target="churned"))
    second = repo.history("customers_dirty.csv")[1]
    assert second["data_changed"] is True
    assert second["uniqueness"] == 100.0


def test_list_datasets_shows_latest_scores(repo, dirty, sample_dir):
    repo.save(dirty)
    repo.save(analyze_file(sample_dir / "customers_clean.csv"))
    rows = {r["name"]: r for r in repo.list_datasets()}
    assert rows["customers_clean.csv"]["trust_grade"] == "A"
    assert rows["customers_dirty.csv"]["runs"] == 1


def test_sql_injection_is_stored_as_text(repo, dirty):
    evil = dict(dirty, file={**dirty["file"], "file_name": "x'; DROP TABLE issues; --.csv"})
    repo.save(evil)
    assert repo.history("x'; DROP TABLE issues; --.csv")
    assert repo.db.query("SELECT COUNT(*) AS n FROM issues")[0]["n"] > 0  # table still there


def test_failed_save_leaves_nothing_behind(repo, dirty):
    broken = dict(dirty, quality={"issues": [{"severity": "low"}]})  # issue missing required fields
    with pytest.raises(KeyError):
        repo.save(broken)
    assert repo.db.query("SELECT COUNT(*) AS n FROM analysis_runs")[0]["n"] == 0  # rolled back
    assert repo.db.query("SELECT COUNT(*) AS n FROM datasets")[0]["n"] == 0


def test_incomplete_analysis_rejected_clearly(repo):
    with pytest.raises(StorageError, match="missing the field"):
        repo.save({"file": {}})


def test_unknown_run_id(repo):
    with pytest.raises(StorageError, match="No analysis run"):
        repo.get_analysis(999)


def test_deleting_a_dataset_cascades(repo, dirty):
    repo.save(dirty)
    with repo.db.transaction():
        repo.db.execute("DELETE FROM datasets")
    assert repo.db.query("SELECT COUNT(*) AS n FROM issues")[0]["n"] == 0


# ------------------------------------------------------------ end to end
def test_file_fingerprint_is_stable(sample_dir):
    path = sample_dir / "customers_clean.csv"
    assert file_sha256(path) == file_sha256(path)
    assert len(file_sha256(path)) == 64


def test_open_repository_and_cli_save(tmp_path, sample_dir, monkeypatch):
    db_path = tmp_path / "cli.db"
    monkeypatch.setenv("DATATRUST_DB", "sqlite")
    monkeypatch.setenv("DATATRUST_SQLITE_PATH", str(db_path))
    from main import run
    assert run([str(sample_dir / "customers_clean.csv"), "--save"]) == 0
    repo = open_repository()
    assert repo.list_datasets()[0]["name"] == "customers_clean.csv"
    repo.db.close()


def test_db_script_commands(tmp_path, sample_dir, monkeypatch, capsys):
    monkeypatch.setenv("DATATRUST_DB", "sqlite")
    monkeypatch.setenv("DATATRUST_SQLITE_PATH", str(tmp_path / "s.db"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("db_script", sample_dir.parent.parent / "scripts" / "db.py")
    db_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(db_script)
    from main import run
    run([str(sample_dir / "customers_dirty.csv"), "--target", "churned", "--save"])
    capsys.readouterr()
    assert db_script.main(["history", "customers_dirty.csv"]) == 0
    assert "run #1" in capsys.readouterr().out
    out_html = tmp_path / "r.html"
    assert db_script.main(["report", "1", "--html", str(out_html)]) == 0
    assert "Data Trust Score" in out_html.read_text(encoding="utf-8")
    assert db_script.main(["issues", "999"]) == 0          # no issues: prints nothing, no crash
    assert db_script.main(["report", "999", "--html", str(out_html)]) == 1
