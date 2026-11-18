"""
tests/test_deploy.py  (Deployment)

Everything about the public demo that can be checked WITHOUT Django:
  - production settings rules (datatrust/web/deploy.py), and that local mode is unchanged;
  - visitor cookies, owner tags, upload rate limits, public upload checks (datatrust/web/demo.py);
  - private runs in the repository: visibility, same 404 for "not yours", pruning;
  - idempotent sample-data loading and start-up upkeep;
  - render.yaml / requirements.txt / .gitignore keep secrets out of GitHub.
The page-level checks (two browsers, 429, 413, banner) are Django tests in webapp/dashboard/tests.py.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from datatrust.pipeline import analyze_file
from datatrust.storage import AnalysisRepository, Database, DBConfig, RunNotFound
from datatrust.web import UploadRejected, analyze_upload, dashboard_page, dataset_page, run_page
from datatrust.web.demo import (
    UploadRateLimiter,
    check_public_upload,
    client_address,
    new_visitor_cookie,
    owner_tag,
    read_visitor_cookie,
    seed_sample_data,
    startup_upkeep,
)
from datatrust.web.deploy import DEV_SECRET_KEY, DemoLimits, DeploymentConfigError, django_settings
from datatrust.web.jobs import JobManager
from datatrust.web.services import NotFound, start_upload_job

ROOT = Path(__file__).resolve().parent.parent
STRONG = "k3y-" + "a1B2c3D4e5" * 6                     # 64 characters
SECRET = "test-secret-" + "x" * 40


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)      # stored times are naive UTC


# ======================================================== production settings
def test_local_mode_is_unchanged():
    s = django_settings({})
    assert s["DEBUG"] is True and s["SECRET_KEY"] == DEV_SECRET_KEY
    assert s["ALLOWED_HOSTS"] == ["localhost", "127.0.0.1"]
    assert s["DATATRUST_PUBLIC_DEMO"] is False
    assert not s["SECURE_SSL_REDIRECT"] and s["SECURE_HSTS_SECONDS"] == 0
    assert s["SECURE_PROXY_SSL_HEADER"] is None and s["CSRF_COOKIE_SECURE"] is False
    assert s["CSRF_TRUSTED_ORIGINS"] == [] and s["SILENCED_SYSTEM_CHECKS"] == []


def test_production_needs_a_secret_key():
    with pytest.raises(DeploymentConfigError, match="Set DATATRUST_SECRET_KEY"):
        django_settings({"DATATRUST_DEBUG": "0", "DATATRUST_WEB_PASSWORD": "pw"})


@pytest.mark.parametrize("weak", ["short", DEV_SECRET_KEY])
def test_production_rejects_a_weak_secret_key(weak):
    with pytest.raises(DeploymentConfigError, match="too weak"):
        django_settings({"DATATRUST_DEBUG": "0", "DATATRUST_SECRET_KEY": weak, "DATATRUST_WEB_PASSWORD": "pw"})


def test_private_production_still_needs_a_password():
    with pytest.raises(DeploymentConfigError, match="Set DATATRUST_WEB_PASSWORD"):
        django_settings({"DATATRUST_DEBUG": "0", "DATATRUST_SECRET_KEY": STRONG})


def test_public_demo_needs_no_password_and_turns_https_on():
    s = django_settings({"DATATRUST_DEBUG": "0", "DATATRUST_PUBLIC_DEMO": "1", "DATATRUST_SECRET_KEY": STRONG,
                         "RENDER_EXTERNAL_HOSTNAME": "datatrust-ai.onrender.com"})
    assert s["DEBUG"] is False and s["DATATRUST_PUBLIC_DEMO"] is True
    assert s["SECURE_SSL_REDIRECT"] is True and s["SECURE_HSTS_SECONDS"] > 0
    assert s["SECURE_PROXY_SSL_HEADER"] == ("HTTP_X_FORWARDED_PROTO", "https")
    assert s["CSRF_COOKIE_SECURE"] is True
    assert s["SECURE_HSTS_INCLUDE_SUBDOMAINS"] is False and s["SECURE_HSTS_PRELOAD"] is False
    assert "datatrust-ai.onrender.com" in s["ALLOWED_HOSTS"]
    assert s["CSRF_TRUSTED_ORIGINS"] == ["https://datatrust-ai.onrender.com"]


def test_public_demo_refuses_debug_mode():
    with pytest.raises(DeploymentConfigError, match="needs DATATRUST_DEBUG=0"):
        django_settings({"DATATRUST_PUBLIC_DEMO": "1"})


def test_render_hostname_is_not_duplicated_and_https_can_be_switched_off():
    s = django_settings({"DATATRUST_DEBUG": "0", "DATATRUST_PUBLIC_DEMO": "1", "DATATRUST_SECRET_KEY": STRONG,
                         "DATATRUST_ALLOWED_HOSTS": "a.example", "RENDER_EXTERNAL_HOSTNAME": "a.example",
                         "DATATRUST_HTTPS": "0"})
    assert s["ALLOWED_HOSTS"] == ["a.example"]
    assert not s["SECURE_SSL_REDIRECT"] and s["SECURE_PROXY_SSL_HEADER"] is None


def test_demo_limits_defaults_overrides_and_bad_values():
    assert DemoLimits.from_env({}) == DemoLimits()
    limits = DemoLimits.from_env({"DATATRUST_DEMO_MAX_ROWS": "100", "DATATRUST_DEMO_UPLOADS_PER_HOUR": "3"})
    assert limits.max_rows == 100 and limits.uploads_per_hour == 3 and limits.max_columns == 50
    with pytest.raises(DeploymentConfigError, match="whole number"):
        DemoLimits.from_env({"DATATRUST_DEMO_MAX_ROWS": "lots"})
    with pytest.raises(DeploymentConfigError, match="at least 1"):
        DemoLimits.from_env({"DATATRUST_DEMO_MAX_PENDING": "0"})


# ============================================================ visitors
def test_visitor_cookie_round_trip_and_tampering():
    visitor, cookie = new_visitor_cookie(SECRET)
    assert read_visitor_cookie(cookie, SECRET) == visitor
    other, _ = new_visitor_cookie(SECRET)
    assert other != visitor                                          # random
    forged = other + "." + cookie.split(".")[1]                      # someone else's id, my signature
    for bad in [None, "", "abc", forged, cookie + "x", cookie.replace(".", ".."), "Z" * 32 + ".00"]:
        assert read_visitor_cookie(bad, SECRET) is None, bad
    assert read_visitor_cookie(cookie, "another-secret") is None     # new SECRET_KEY = new visitors


def test_owner_tag_is_stable_private_and_per_visitor():
    a, cookie = new_visitor_cookie(SECRET)
    b, _ = new_visitor_cookie(SECRET)
    assert owner_tag(a, SECRET) == owner_tag(a, SECRET)
    assert owner_tag(a, SECRET) != owner_tag(b, SECRET)
    assert owner_tag(a, SECRET) not in cookie and a not in owner_tag(a, SECRET)
    assert len(owner_tag(a, SECRET)) == 32


def test_client_address():
    assert client_address({"HTTP_X_FORWARDED_FOR": "203.0.113.9, 10.0.0.1", "REMOTE_ADDR": "10.0.0.1"}) == "203.0.113.9"
    assert client_address({"REMOTE_ADDR": "127.0.0.1"}) == "127.0.0.1"
    assert client_address({}) == "unknown"


def test_rate_limiter_per_key_total_window_and_refusals_not_counted():
    limiter = UploadRateLimiter()
    t = 1000.0
    assert limiter.check_and_record(["visitor:a"], per_key=2, total=3, now=t) == 0
    assert limiter.check_and_record(["visitor:a"], per_key=2, total=3, now=t + 1) == 0
    wait = limiter.check_and_record(["visitor:a"], per_key=2, total=3, now=t + 2)
    assert 3500 < wait <= 3601                                       # until the first upload leaves the window
    assert limiter.check_and_record(["visitor:a"], per_key=2, total=3, now=t + 3) > 0   # refusal not counted...
    assert limiter.check_and_record(["visitor:b"], per_key=2, total=3, now=t + 4) == 0  # ...so b still fits
    assert limiter.check_and_record(["visitor:c"], per_key=2, total=3, now=t + 5) > 0   # server-wide total
    assert limiter.check_and_record(["visitor:a"], per_key=2, total=3, now=t + 3601) == 0  # window moved on


# ======================================================= public upload checks
def test_public_upload_checks(tmp_path):
    limits = DemoLimits(max_rows=5, max_columns=3)
    ok = tmp_path / "ok.csv"
    ok.write_text("a,b\n1,2\n3,4\n")
    check_public_upload(ok, "ok.csv", limits)                        # no exception

    binary = tmp_path / "b.csv"
    binary.write_bytes(b"PK\x03\x04\x00\x00 zip pretending to be csv")
    with pytest.raises(UploadRejected, match="binary data"):
        check_public_upload(binary, "report.csv", limits)

    rows = tmp_path / "rows.csv"
    rows.write_text("a,b\n" + "1,2\n" * 6)
    with pytest.raises(UploadRejected, match="6 rows; the public demo accepts up to 5"):
        check_public_upload(rows, "rows.csv", limits)

    cols = tmp_path / "cols.csv"
    cols.write_text("a,b,c,d\n1,2,3,4\n")
    with pytest.raises(UploadRejected, match="4 columns; the public demo accepts up to 3"):
        check_public_upload(cols, "cols.csv", limits)

    empty = tmp_path / "empty.csv"
    empty.write_text("")
    check_public_upload(empty, "empty.csv", limits)   # left to the pipeline, which explains it


# ===================================================== private runs (storage)
@pytest.fixture
def repo(tmp_path):
    db = Database(DBConfig(backend="sqlite", sqlite_path=tmp_path / "demo.db")).connect()
    r = AnalysisRepository(db)
    r.init_schema()
    yield r
    db.close()


@pytest.fixture(scope="module")
def small_analysis(tmp_path_factory):
    path = tmp_path_factory.mktemp("a") / "people.csv"
    path.write_text("id,age,city\n1,30,Pune\n2,41,Delhi\n3,,Pune\n4,29,Chennai\n")
    return analyze_file(path)


def test_runs_are_private_to_their_owner(repo, small_analysis):
    public = repo.save(small_analysis)
    mine = repo.save(small_analysis, owner="alice")
    theirs = repo.save(small_analysis, owner="bob")
    assert [r["run_id"] for r in repo.history("people.csv", visible_to="alice")] == [public, mine]
    assert [r["run_id"] for r in repo.history("people.csv")] == [public, mine, theirs]   # local: everything
    assert repo.list_datasets(visible_to="alice")[0]["runs"] == 2
    repo.get_analysis(mine, visible_to="alice")
    repo.get_analysis(public, visible_to="alice")
    with pytest.raises(RunNotFound) as other:
        repo.get_analysis(theirs, visible_to="alice")
    with pytest.raises(RunNotFound) as missing:
        repo.get_analysis(9999, visible_to="alice")
    assert str(other.value).replace(str(theirs), "N") == str(missing.value).replace("9999", "N")


def test_a_dataset_only_someone_else_uploaded_is_invisible(repo, small_analysis):
    repo.save(small_analysis, owner="bob")
    assert repo.list_datasets(visible_to="alice") == []
    assert dashboard_page(repo, visible_to="alice")["total_runs"] == 0
    with pytest.raises(NotFound):
        dataset_page(repo, "people.csv", visible_to="alice")
    assert dataset_page(repo, "people.csv", visible_to="bob")["name"] == "people.csv"


def test_pages_respect_visibility(repo, small_analysis):
    theirs = repo.save(small_analysis, owner="bob")
    with pytest.raises(NotFound):
        run_page(repo, theirs, visible_to="alice")
    assert run_page(repo, theirs, visible_to="bob")["run_id"] == theirs
    assert run_page(repo, theirs)["run_id"] == theirs                # local app: no restriction


def test_prune_removes_old_and_excess_visitor_runs_only(repo, small_analysis):
    public = repo.save(small_analysis)
    owned = [repo.save(small_analysis, owner=f"v{i}") for i in range(4)]
    later = _utcnow() + timedelta(hours=1)
    assert repo.prune_owned_runs(max_age_hours=24, max_runs=2, now=later) == 2     # keeps the newest 2
    assert [r["run_id"] for r in repo.history("people.csv")] == [public, owned[2], owned[3]]
    assert repo.prune_owned_runs(max_age_hours=24, now=later + timedelta(hours=25)) == 2   # expired
    assert [r["run_id"] for r in repo.history("people.csv")] == [public]
    assert repo.db.query("SELECT COUNT(*) AS n FROM issues WHERE run_id != ?", (public,))[0]["n"] == 0
    assert repo.db.query("SELECT COUNT(*) AS n FROM run_owners")[0]["n"] == 0
    assert repo.prune_owned_runs(max_age_hours=24) == 0


def test_prune_removes_datasets_left_empty(repo, small_analysis):
    repo.save(small_analysis, owner="bob")
    repo.prune_owned_runs(max_age_hours=0, now=_utcnow() + timedelta(seconds=5))
    assert repo.db.query("SELECT COUNT(*) AS n FROM datasets")[0]["n"] == 0


def test_find_public_run_ignores_visitor_runs(repo, small_analysis):
    sha = small_analysis["file"]["sha256"]
    repo.save(small_analysis, owner="bob")
    assert repo.find_public_run("people.csv", sha) is None
    run_id = repo.save(small_analysis)
    assert repo.find_public_run("people.csv", sha) == run_id
    assert repo.find_public_run("people.csv", "0" * 64) is None


# =========================================================== sample data
def _samples(tmp_path):
    d = tmp_path / "samples"
    d.mkdir()
    (d / "a.csv").write_text("id,x,y\n1,2,yes\n2,3,no\n3,4,yes\n")
    (d / "b.csv").write_text("id,v\n1,10\n2,20\n")
    return d, ({"file": "a.csv", "target": "y"}, {"file": "b.csv"}, {"file": "gone.csv"})


def test_sample_loading_is_idempotent(repo, tmp_path):
    folder, samples = _samples(tmp_path)
    assert seed_sample_data(repo, samples, folder) == [("a.csv", "added"), ("b.csv", "added"), ("gone.csv", "missing")]
    assert seed_sample_data(repo, samples, folder) == [("a.csv", "present"), ("b.csv", "present"), ("gone.csv", "missing")]
    assert repo.db.query("SELECT COUNT(*) AS n FROM analysis_runs")[0]["n"] == 2
    assert repo.db.query("SELECT COUNT(*) AS n FROM run_owners")[0]["n"] == 0     # samples are public
    (folder / "b.csv").write_text("id,v\n1,10\n2,20\n3,30\n")                      # new sample version
    assert seed_sample_data(repo, samples, folder)[1] == ("b.csv", "added")
    assert len(repo.history("b.csv")) == 2


def test_a_visitor_uploading_the_sample_does_not_stop_it_loading(repo, tmp_path):
    folder, samples = _samples(tmp_path)
    analyze_upload([(folder / "b.csv").read_bytes()], "b.csv", repo, owner="bob")
    assert seed_sample_data(repo, samples[1:2], folder) == [("b.csv", "added")]


def test_startup_upkeep_seeds_prunes_and_never_raises(tmp_path, monkeypatch):
    from datatrust.web import demo, reset_repository_state
    folder, samples = _samples(tmp_path)
    monkeypatch.setattr(demo, "SAMPLES", samples[:2])
    monkeypatch.setattr(demo, "SAMPLE_DIR", folder)
    monkeypatch.setattr(demo.seed_sample_data, "__defaults__", (samples[:2], folder))
    env = {"DATATRUST_DB": "sqlite", "DATATRUST_SQLITE_PATH": str(tmp_path / "up.db")}
    reset_repository_state()
    try:
        startup_upkeep(env)
        startup_upkeep(env)                                          # second start: nothing new
        db = Database(DBConfig.from_env(env)).connect()
        assert db.query("SELECT COUNT(*) AS n FROM analysis_runs")[0]["n"] == 2
        db.close()
        startup_upkeep({"DATATRUST_DB": "oracle"})                   # broken config: logged, not raised
    finally:
        reset_repository_state()


# ============================================================ uploads with owners
def test_upload_with_owner_and_extra_check(repo, tmp_path):
    data = b"id,x\n1,2\n2,3\n"
    run_id = analyze_upload([data], "mine.csv", repo, owner="alice")
    with pytest.raises(RunNotFound):
        repo.get_analysis(run_id, visible_to="bob")

    def refuse(path, name):
        raise UploadRejected(f"{name} refused")
    with pytest.raises(UploadRejected, match="mine.csv refused"):
        analyze_upload([data], "mine.csv", repo, check=refuse)


def test_start_upload_job_checks_at_once_and_records_the_owner(tmp_path, monkeypatch):
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))         # see what gets left behind
    jobs = JobManager(workers=1)
    env = {"DATATRUST_DB": "sqlite", "DATATRUST_SQLITE_PATH": str(tmp_path / "jobs.db")}

    def refuse(path, name):
        assert path.is_file()
        raise UploadRejected("too many rows")
    with pytest.raises(UploadRejected, match="too many rows"):
        start_upload_job(jobs, [b"a,b\n1,2\n"], "x.csv", environ=env, check=refuse)
    assert not list(tmp_path.glob("datatrust-job-*"))               # staged copy deleted
    assert jobs.pending() == 0

    job_id = start_upload_job(jobs, [b"a,b\n1,2\n3,4\n"], "x.csv", environ=env, owner="alice")
    assert jobs.get(job_id)["owner"] == "alice"
    done = jobs.wait(job_id, 60)
    assert done["state"] == "done", done
    assert jobs.pending() == 0 and jobs.pending("alice") == 0
    jobs.shutdown()


def test_jobs_pending_counts_per_owner():
    import threading
    jobs = JobManager(workers=1)
    release = threading.Event()
    a = jobs.submit(lambda: release.wait(10) and 1, label="a.csv", owner="alice")
    b = jobs.submit(lambda: 2, label="b.csv", owner="bob")
    try:
        assert jobs.pending() == 2 and jobs.pending("alice") == 1 and jobs.pending("carol") == 0
    finally:
        release.set()
        jobs.wait(a, 10)
        jobs.wait(b, 10)
        jobs.shutdown()
    assert jobs.pending() == 0


# ===================================================== deployment files
def test_render_blueprint_is_safe_and_complete():
    text = (ROOT / "render.yaml").read_text(encoding="utf-8")
    assert "key: DATATRUST_SECRET_KEY\n        generateValue: true" in text   # Render makes the secret
    assert "DATATRUST_WEB_PASSWORD" not in text                               # open demo: no password at all
    for needle in ['key: DATATRUST_DEBUG\n        value: "0"', 'key: DATATRUST_PUBLIC_DEMO\n        value: "1"',
                   "--workers 1", "python scripts/seed_demo.py", "plan: free", "$PORT",
                   'key: DATATRUST_MAX_FILE_MB\n        value: "2"']:
        assert needle in text, needle
    for secret_word in ("ANTHROPIC_API_KEY", "MYSQL_PASSWORD", "sk-ant"):
        assert secret_word not in text


def test_gunicorn_is_linux_only_and_secrets_stay_ignored():
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert 'gunicorn>=23.0; sys_platform != "win32"' in req
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").split()
    assert ".env" in ignored and "*.db" in ignored


def test_settings_defaults_unchanged_when_no_demo_variables(monkeypatch):
    """The lighter model settings only apply when render.yaml sets them."""
    from datatrust.config import Settings
    for name in ("DATATRUST_ML_CV_REPEATS", "DATATRUST_ML_IMPORTANCE_REPEATS"):
        monkeypatch.delenv(name, raising=False)
    s = Settings()
    assert (s.ml_cv_repeats, s.ml_importance_repeats) == (3, 5)
    monkeypatch.setenv("DATATRUST_ML_CV_REPEATS", "1")
    assert Settings().ml_cv_repeats == 1
