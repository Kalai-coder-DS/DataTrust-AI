"""
Django tests for every page of the web app.

    python webapp/manage.py test dashboard

Each test gets its own empty SQLite database (through datatrust.storage, the same code the
app uses with MySQL), so nothing touches your real MySQL data. SimpleTestCase is used
because there is no Django-managed database (Option B).
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest import mock

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, SimpleTestCase
from django.urls import reverse

from datatrust.storage import StorageError
from datatrust.web import reset_repository_state

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"


def csv_upload(path: Path | None = None, name: str | None = None, content: bytes | None = None) -> SimpleUploadedFile:
    data = content if content is not None else path.read_bytes()
    return SimpleUploadedFile(name or path.name, data, content_type="text/csv")


class WebTestCase(SimpleTestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="datatrust-webtest-")
        patcher = mock.patch.dict(os.environ, {
            "DATATRUST_DB": "sqlite",
            "DATATRUST_SQLITE_PATH": str(Path(self.tmp) / "web.db"),
            "DATATRUST_WEB_BACKGROUND": "0",       # analyse inside the request (Phase 10 tests switch it on)
            "DATATRUST_WEB_PASSWORD": "",          # login off, even if your .env sets a password
        })
        patcher.start()
        self.addCleanup(patcher.stop)
        reset_repository_state()
        self.addCleanup(reset_repository_state)
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def upload(self, path=None, name=None, target="", content=None, baseline=False):
        data = {"file": csv_upload(path, name, content), "target": target}
        if baseline:
            data["baseline"] = "on"
        return self.client.post(reverse("dashboard:upload"), data)

    def upload_dirty(self) -> int:
        response = self.upload(SAMPLE_DIR / "customers_dirty.csv", target="churned")
        self.assertEqual(response.status_code, 302)
        return int(response["Location"].rstrip("/").split("/")[-1])


class HomePageTests(WebTestCase):
    def test_empty_dashboard(self):
        r = self.client.get(reverse("dashboard:home"))
        self.assertEqual(r.status_code, 200)
        self.assertTemplateUsed(r, "dashboard/home.html")
        self.assertContains(r, "No datasets yet")
        self.assertContains(r, reverse("dashboard:upload"))

    def test_dashboard_lists_datasets_with_latest_scores(self):
        self.upload_dirty()
        self.upload_dirty()
        self.upload(SAMPLE_DIR / "customers_clean.csv")
        r = self.client.get(reverse("dashboard:home"))
        self.assertContains(r, "customers_dirty.csv")
        self.assertContains(r, "customers_clean.csv")
        self.assertContains(r, reverse("dashboard:dataset", args=["customers_dirty.csv"]))
        self.assertContains(r, "79.8")
        self.assertContains(r, "C · usable with fixes")
        self.assertEqual(r.context["total_runs"], 3)
        self.assertEqual(r.context["total_datasets"], 2)

    def test_post_not_allowed(self):
        self.assertEqual(self.client.post(reverse("dashboard:home")).status_code, 405)


class UploadTests(WebTestCase):
    def test_form_page(self):
        r = self.client.get(reverse("dashboard:upload"))
        self.assertEqual(r.status_code, 200)
        self.assertContains(r, 'enctype="multipart/form-data"')
        self.assertContains(r, "csrfmiddlewaretoken")
        self.assertContains(r, "Target column (optional)")
        self.assertContains(r, f"{r.context['limit_mb']} MB")

    def test_upload_analyses_saves_and_redirects_to_run(self):
        r = self.upload(SAMPLE_DIR / "customers_dirty.csv", target="churned")
        self.assertEqual(r.status_code, 302)
        self.assertEqual(r["Location"], reverse("dashboard:run", args=[1]))
        page = self.client.get(r["Location"])
        self.assertEqual(page.status_code, 200)
        self.assertContains(page, "customers_dirty.csv")
        self.assertContains(page, "79.8")
        self.assertContains(page, "churned")

    def test_wrong_file_type(self):
        r = self.upload(name="numbers.xlsx", content=b"a,b\n1,2\n")
        self.assertEqual(r.status_code, 400)
        self.assertContains(r, "is not a CSV file", status_code=400)
        self.assertContains(self.client.get(reverse("dashboard:home")), "No datasets yet")

    def test_unknown_target_keeps_form_values(self):
        r = self.upload(SAMPLE_DIR / "customers_dirty.csv", target="no_such_column")
        self.assertEqual(r.status_code, 400)
        self.assertContains(r, "not found. Available", status_code=400)
        self.assertContains(r, 'value="no_such_column"', status_code=400)

    def test_missing_file(self):
        r = self.client.post(reverse("dashboard:upload"), {"target": ""})
        self.assertEqual(r.status_code, 400)
        self.assertContains(r, "Choose a CSV file to analyse.", status_code=400)

    def test_unreadable_csv(self):
        r = self.upload(name="header_only.csv", content=b"only_a_header\n")
        self.assertEqual(r.status_code, 400)
        self.assertContains(r, 'role="alert"', status_code=400)

    def test_csrf_is_enforced(self):
        client = Client(enforce_csrf_checks=True)
        r = client.post(reverse("dashboard:upload"),
                        {"file": csv_upload(SAMPLE_DIR / "customers_clean.csv"), "target": ""})
        self.assertEqual(r.status_code, 403)

    def test_database_down_during_upload(self):
        with mock.patch("dashboard.views.repository", side_effect=StorageError("Cannot reach the server.")):
            r = self.upload(SAMPLE_DIR / "customers_clean.csv")
        self.assertEqual(r.status_code, 503)
        self.assertContains(r, "Database unavailable", status_code=503)


class BaselineModelTests(WebTestCase):
    def test_form_offers_the_baseline_ticked(self):
        r = self.client.get(reverse("dashboard:upload"))
        self.assertContains(r, 'name="baseline"')
        self.assertContains(r, "checked")

    def test_run_page_shows_the_baseline_model(self):
        r = self.upload(SAMPLE_DIR / "customers_dirty.csv", target="churned", baseline=True)
        self.assertEqual(r.status_code, 302)
        page = self.client.get(r["Location"])
        self.assertTemplateUsed(page, "dashboard/_baseline.html")
        self.assertTemplateUsed(page, "dashboard/_importance.html")
        self.assertContains(page, "What the model relies on")
        self.assertContains(page, "tenure_months")
        for text in ["Baseline model", "Guessing", "Raw data", "After cleaning", "Balanced accuracy",
                     "Removed 15 duplicate row(s)", "within the run-to-run noise"]:
            self.assertContains(page, text)
        report = self.client.get(reverse("dashboard:report", args=[1]))
        self.assertContains(report, "Baseline model")
        self.assertContains(report, "What the model relies on")
        data = json.loads(self.client.get(reverse("dashboard:json", args=[1])).content)
        self.assertEqual(data["ml"]["status"], "ok")

    def test_without_baseline_no_section(self):
        r = self.upload(SAMPLE_DIR / "customers_dirty.csv", target="churned")
        self.assertNotContains(self.client.get(r["Location"]), "Baseline model")


class SummaryTests(WebTestCase):
    def test_form_offers_the_summary_choice(self):
        r = self.client.get(reverse("dashboard:upload"))
        self.assertContains(r, 'name="summary"')
        self.assertContains(r, '<option value="auto" selected>')

    def test_template_summary_on_run_page_and_report(self):
        r = self.client.post(reverse("dashboard:upload"),
                             {"file": csv_upload(SAMPLE_DIR / "customers_dirty.csv"), "target": "churned",
                              "summary": "template"})
        self.assertEqual(r.status_code, 302)
        page = self.client.get(r["Location"])
        self.assertTemplateUsed(page, "dashboard/_summary.html")
        self.assertContains(page, "Template summary (offline)")
        self.assertContains(self.client.get(reverse("dashboard:report", args=[1])), "Template summary (offline)")

    def test_claude_without_key_falls_back_visibly(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}):
            r = self.client.post(reverse("dashboard:upload"),
                                 {"file": csv_upload(SAMPLE_DIR / "customers_clean.csv"), "summary": "claude"})
        self.assertContains(self.client.get(r["Location"]), "ANTHROPIC_API_KEY is not set")

    def test_no_summary_choice(self):
        r = self.client.post(reverse("dashboard:upload"),
                             {"file": csv_upload(SAMPLE_DIR / "customers_clean.csv"), "summary": ""})
        self.assertTemplateNotUsed(self.client.get(r["Location"]), "dashboard/_summary.html")


class LoginTests(WebTestCase):
    """Phase 10: single-user login."""

    def setUp(self):
        super().setUp()
        from datatrust.web.auth import LoginThrottle
        for patcher in (mock.patch.dict(os.environ, {"DATATRUST_WEB_PASSWORD": "s3cret"}),
                        mock.patch("dashboard.views.throttle", LoginThrottle())):
            patcher.start()
            self.addCleanup(patcher.stop)

    def login(self, password="s3cret", next_url="/"):
        return self.client.post(reverse("dashboard:login"), {"password": password, "next": next_url})

    def test_pages_redirect_to_login(self):
        r = self.client.get(reverse("dashboard:run", args=[1]))
        self.assertEqual(r.status_code, 302)
        self.assertEqual(r["Location"], "/login/?next=%2Fruns%2F1%2F")
        self.assertEqual(self.client.get(reverse("dashboard:upload")).status_code, 302)
        self.assertEqual(self.client.get(reverse("dashboard:login")).status_code, 200)

    def test_wrong_then_right_password(self):
        r = self.login("nope")
        self.assertEqual(r.status_code, 401)
        self.assertContains(r, "Wrong password.", status_code=401)
        r = self.login(next_url="/upload/")
        self.assertRedirects(r, "/upload/")
        cookie = r.cookies["datatrust_auth"]
        self.assertTrue(cookie["httponly"])
        self.assertEqual(cookie["samesite"], "Lax")
        home = self.client.get(reverse("dashboard:home"))
        self.assertEqual(home.status_code, 200)
        self.assertContains(home, "Log out")

    def test_changing_the_password_logs_out(self):
        self.login()
        with mock.patch.dict(os.environ, {"DATATRUST_WEB_PASSWORD": "new-password"}):
            self.assertEqual(self.client.get(reverse("dashboard:home")).status_code, 302)

    def test_forged_cookie_is_rejected(self):
        self.client.cookies["datatrust_auth"] = "9999999999." + "0" * 64
        self.assertEqual(self.client.get(reverse("dashboard:home")).status_code, 302)

    def test_no_open_redirect(self):
        self.assertRedirects(self.login(next_url="https://evil.example/"), "/", fetch_redirect_response=False)

    def test_lockout_after_five_wrong_passwords(self):
        for _ in range(5):
            self.login("nope")
        r = self.login()                                   # even the right password waits
        self.assertEqual(r.status_code, 429)
        self.assertContains(r, "Too many wrong passwords", status_code=429)

    def test_logout(self):
        self.login()
        self.assertEqual(self.client.get(reverse("dashboard:logout")).status_code, 405)   # POST only (CSRF-safe)
        r = self.client.post(reverse("dashboard:logout"))
        self.assertRedirects(r, reverse("dashboard:login"))
        self.assertEqual(self.client.get(reverse("dashboard:home")).status_code, 302)

    def test_login_needs_csrf(self):
        client = Client(enforce_csrf_checks=True)
        self.assertEqual(client.post(reverse("dashboard:login"), {"password": "s3cret"}).status_code, 403)

    def test_login_page_without_password_set_goes_home(self):
        with mock.patch.dict(os.environ, {"DATATRUST_WEB_PASSWORD": ""}):
            self.assertRedirects(self.client.get(reverse("dashboard:login")), "/")


class BackgroundJobTests(WebTestCase):
    """Phase 10: uploads analysed in a background thread."""

    def setUp(self):
        super().setUp()
        patcher = mock.patch.dict(os.environ, {"DATATRUST_WEB_BACKGROUND": "1"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def job_id(self, response):
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response["Location"].startswith("/jobs/"), response["Location"])
        return response["Location"].rstrip("/").split("/")[-1]

    def test_upload_returns_at_once_then_opens_the_run(self):
        from dashboard.views import jobs
        job_id = self.job_id(self.upload(SAMPLE_DIR / "customers_dirty.csv", target="churned"))
        self.assertEqual(jobs.wait(job_id, 60)["state"], "done")
        r = self.client.get(reverse("dashboard:job", args=[job_id]))
        self.assertRedirects(r, reverse("dashboard:run", args=[1]))
        self.assertContains(self.client.get(r["Location"]), "79.8")

    def test_running_page_refreshes_itself(self):
        import threading
        from dashboard.views import jobs
        release = threading.Event()
        job_id = jobs.submit(lambda: release.wait(10) and 1, label="slow.csv")
        try:
            r = self.client.get(reverse("dashboard:job", args=[job_id]))
            self.assertEqual(r.status_code, 200)
            self.assertContains(r, '<meta http-equiv="refresh" content="3">')
            self.assertContains(r, "slow.csv")
        finally:
            release.set()
            jobs.wait(job_id, 10)

    def test_bad_target_shows_the_message_on_the_job_page(self):
        from dashboard.views import jobs
        job_id = self.job_id(self.upload(SAMPLE_DIR / "customers_clean.csv", target="nope"))
        jobs.wait(job_id, 30)
        r = self.client.get(reverse("dashboard:job", args=[job_id]))
        self.assertEqual(r.status_code, 400)
        self.assertContains(r, "not found. Available", status_code=400)
        self.assertNotContains(r, "http-equiv", status_code=400)

    def test_wrong_file_type_is_still_rejected_on_the_form(self):
        r = self.upload(name="numbers.xlsx", content=b"a,b\n1,2\n")
        self.assertEqual(r.status_code, 400)
        self.assertContains(r, "is not a CSV file", status_code=400)

    def test_unknown_or_malformed_job_is_404(self):
        self.assertEqual(self.client.get(reverse("dashboard:job", args=["0" * 32])).status_code, 404)
        self.assertEqual(self.client.get("/jobs/../../etc/").status_code, 404)
        self.assertEqual(self.client.get(reverse("dashboard:job", args=["x" * 32])).status_code, 404)


class DatasetPageTests(WebTestCase):
    def test_history_chart_and_table(self):
        self.upload_dirty()
        self.upload_dirty()
        r = self.client.get(reverse("dashboard:dataset", args=["customers_dirty.csv"]))
        self.assertEqual(r.status_code, 200)
        self.assertTemplateUsed(r, "dashboard/dataset.html")
        self.assertContains(r, "<svg", count=1)
        self.assertContains(r, "Score history for customers_dirty.csv")
        self.assertContains(r, "first run")
        self.assertContains(r, "same data")
        self.assertContains(r, "no change")
        self.assertContains(r, reverse("dashboard:run", args=[2]))
        self.assertContains(r, reverse("dashboard:report", args=[1]))

    def test_unknown_dataset_is_404(self):
        r = self.client.get(reverse("dashboard:dataset", args=["never.csv"]))
        self.assertEqual(r.status_code, 404)
        self.assertContains(r, "No dataset called", status_code=404)

    def test_awkward_names_round_trip_through_the_url(self):
        for name in ["x'; DROP TABLE issues; --.csv", "my data #1?.csv", "données é.csv"]:
            self.upload(SAMPLE_DIR / "customers_clean.csv", name=name)
            url = reverse("dashboard:dataset", args=[name])
            r = self.client.get(url)
            self.assertEqual(r.status_code, 200, (name, url))
            self.assertEqual(r.context["name"], name)


class RunPageTests(WebTestCase):
    def test_run_summary(self):
        run_id = self.upload_dirty()
        r = self.client.get(reverse("dashboard:run", args=[run_id]))
        self.assertEqual(r.status_code, 200)
        self.assertTemplateUsed(r, "dashboard/run.html")
        self.assertTemplateUsed(r, "dashboard/_score_card.html")
        for text in ["Data Trust Score", "ML Readiness Score", "Completeness", "Top actions",
                     "Data quality issues", "Recommendation", "515 rows"]:
            self.assertContains(r, text)
        self.assertContains(r, reverse("dashboard:report", args=[run_id]))
        self.assertContains(r, reverse("dashboard:json", args=[run_id]))
        self.assertContains(r, reverse("dashboard:dataset", args=["customers_dirty.csv"]))
        self.assertContains(r, 'class="fill" style="width:76.0%"')     # completeness bar

    def test_unknown_run_is_404(self):
        r = self.client.get(reverse("dashboard:run", args=[999]))
        self.assertEqual(r.status_code, 404)
        self.assertContains(r, "No analysis run with id 999", status_code=404)


class ReportAndJsonTests(WebTestCase):
    def test_full_report(self):
        run_id = self.upload_dirty()
        r = self.client.get(reverse("dashboard:report", args=[run_id]))
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.content.startswith(b"<!DOCTYPE html>"))
        self.assertContains(r, "DataTrust AI report")
        self.assertNotIn("Content-Disposition", r)

    def test_report_download(self):
        run_id = self.upload_dirty()
        r = self.client.get(reverse("dashboard:report", args=[run_id]) + "?download=1")
        self.assertEqual(r["Content-Disposition"], f'attachment; filename="datatrust-run-{run_id}.html"')

    def test_json_download(self):
        run_id = self.upload_dirty()
        r = self.client.get(reverse("dashboard:json", args=[run_id]))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r["Content-Type"], "application/json; charset=utf-8")
        self.assertIn("attachment", r["Content-Disposition"])
        data = json.loads(r.content)
        self.assertEqual(data["scores"]["trust"]["score"], 79.8)
        self.assertEqual(data["file"]["file_name"], "customers_dirty.csv")

    def test_unknown_run_report_and_json_are_404(self):
        self.assertEqual(self.client.get(reverse("dashboard:report", args=[5])).status_code, 404)
        self.assertEqual(self.client.get(reverse("dashboard:json", args=[5])).status_code, 404)


class ErrorAndSecurityTests(WebTestCase):
    def test_unmatched_url_uses_friendly_404(self):
        r = self.client.get("/no/such/page/")
        self.assertEqual(r.status_code, 404)
        self.assertContains(r, "There is no page at this address.", status_code=404)

    def test_database_unavailable_page(self):
        with mock.patch("dashboard.views.repository",
                        side_effect=StorageError("Cannot reach the server: is the MySQL80 service running?")):
            r = self.client.get(reverse("dashboard:home"))
        self.assertEqual(r.status_code, 503)
        self.assertContains(r, "MySQL80 service", status_code=503)

    def test_real_mysql_misconfiguration_is_503_not_500(self):
        with mock.patch.dict(os.environ, {"DATATRUST_DB": "mysql", "DATATRUST_MYSQL_HOST": "127.0.0.1",
                                          "DATATRUST_MYSQL_PORT": "1", "DATATRUST_MYSQL_PASSWORD": "s3cret-pw"}):
            reset_repository_state()
            r = self.client.get(reverse("dashboard:home"))
        self.assertEqual(r.status_code, 503)
        self.assertNotContains(r, "s3cret-pw", status_code=503)

    def test_uploaded_names_and_columns_are_escaped(self):
        evil_name = "<img src=x onerror=alert(1)>.csv"      # no '/', so it survives as the dataset name
        evil_column = "<script>alert(1)</script>"
        content = f"id,{evil_column},value\n1,a,10\n2,b,20\n3,c,30\n".encode()
        r = self.upload(name=evil_name, content=content)
        self.assertEqual(r.status_code, 302)
        for url in [reverse("dashboard:home"), r["Location"],
                    reverse("dashboard:dataset", args=[evil_name]),
                    reverse("dashboard:report", args=[1])]:
            page = self.client.get(url)
            self.assertEqual(page.status_code, 200, url)
            self.assertNotIn(b"<img src=x", page.content, url)
            self.assertNotIn(evil_column.encode(), page.content, url)
            self.assertIn(b"&lt;img src=x", page.content, url)

    def test_security_headers(self):
        r = self.client.get(reverse("dashboard:home"))
        self.assertEqual(r["X-Frame-Options"], "DENY")
        self.assertEqual(r["X-Content-Type-Options"], "nosniff")
        self.assertIn("default-src 'none'", r["Content-Security-Policy"])
        self.assertIn("frame-ancestors 'none'", r["Content-Security-Policy"])


# ============================================================= deployment: public demo
from django.conf import settings as django_settings  # noqa: E402
from django.test import override_settings  # noqa: E402

TINY_CSV = b"id,amount\n1,10\n2,20\n3,30\n"


@override_settings(DATATRUST_PUBLIC_DEMO=True)
class PublicDemoTests(WebTestCase):
    """Deployment: the open demo. Every browser is its own visitor; uploads are private and limited."""

    def setUp(self):
        super().setUp()
        from datatrust.web.demo import UploadRateLimiter
        patcher = mock.patch("dashboard.views.upload_limiter", UploadRateLimiter())
        patcher.start()
        self.addCleanup(patcher.stop)
        self.other = Client(REMOTE_ADDR="10.0.0.2")             # a second visitor: own cookies, own address

    def upload_as(self, client, path=None, name=None, target="", content=None, summary=""):
        return client.post(reverse("dashboard:upload"),
                           {"file": csv_upload(path, name, content), "target": target, "summary": summary})

    def owner_of(self, client) -> str:
        from datatrust.web.demo import VISITOR_COOKIE, owner_tag, read_visitor_cookie
        client.get(reverse("dashboard:home"))
        visitor = read_visitor_cookie(client.cookies[VISITOR_COOKIE].value, django_settings.SECRET_KEY)
        return owner_tag(visitor, django_settings.SECRET_KEY)

    def seed_public_sample(self):
        from datatrust.web import repository
        from datatrust.web.demo import seed_sample_data
        with repository() as repo:
            seed_sample_data(repo, ({"file": "customers_clean.csv"},))

    def test_banner_and_limits_are_shown(self):
        r = self.client.get(reverse("dashboard:home"))
        self.assertContains(r, "Public demo.")
        self.assertContains(r, "visible only to you")
        page = self.client.get(reverse("dashboard:upload"))
        self.assertContains(page, "Public demo limits")
        self.assertContains(page, "50 columns")
        self.assertNotContains(page, "Log out")                  # open demo: no login

    def test_visitor_cookie_is_set_once_and_protected(self):
        first = self.client.get(reverse("dashboard:home"))
        cookie = first.cookies["datatrust_visitor"]
        self.assertTrue(cookie["httponly"])
        self.assertEqual(cookie["samesite"], "Lax")
        self.assertNotIn("datatrust_visitor", self.client.get(reverse("dashboard:home")).cookies)  # kept, not renewed

    def test_uploads_are_private_but_samples_are_shared(self):
        self.seed_public_sample()
        r = self.upload_as(self.client, SAMPLE_DIR / "customers_dirty.csv", target="churned")
        self.assertEqual(r.status_code, 302)
        run_url = r["Location"]
        run_id = int(run_url.rstrip("/").split("/")[-1])
        self.assertContains(self.client.get(run_url), "79.8")
        mine = self.client.get(reverse("dashboard:home"))
        self.assertContains(mine, "customers_dirty.csv")
        self.assertContains(mine, "customers_clean.csv")

        theirs = self.other.get(reverse("dashboard:home"))
        self.assertNotContains(theirs, "customers_dirty.csv")
        self.assertContains(theirs, "customers_clean.csv")       # the public sample
        for url in (run_url, reverse("dashboard:report", args=[run_id]), reverse("dashboard:json", args=[run_id]),
                    reverse("dashboard:dataset", args=["customers_dirty.csv"])):
            self.assertEqual(self.other.get(url).status_code, 404, url)

    def test_a_forged_visitor_cookie_gets_a_new_identity(self):
        r = self.upload_as(self.client, content=TINY_CSV, name="mine.csv")
        stranger = Client()
        genuine = self.client.cookies["datatrust_visitor"].value
        stranger.cookies["datatrust_visitor"] = genuine[:-1] + ("1" if genuine.endswith("0") else "0")
        self.assertEqual(stranger.get(r["Location"]).status_code, 404)

    def test_summary_is_always_the_offline_template(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}):
            r = self.upload_as(self.client, SAMPLE_DIR / "customers_clean.csv", summary="claude")
        page = self.client.get(r["Location"])
        self.assertContains(page, "Template summary (offline)")
        self.assertNotContains(page, "ANTHROPIC_API_KEY is not set")   # no failed attempt, no waiting

    def test_rows_columns_and_binary_files_are_refused_on_the_form(self):
        with mock.patch.dict(os.environ, {"DATATRUST_DEMO_MAX_ROWS": "10"}):
            r = self.upload_as(self.client, SAMPLE_DIR / "customers_dirty.csv")
        self.assertContains(r, "the public demo accepts up to 10", status_code=400)
        with mock.patch.dict(os.environ, {"DATATRUST_DEMO_MAX_COLUMNS": "3"}):
            r = self.upload_as(self.client, SAMPLE_DIR / "customers_clean.csv")
        self.assertContains(r, "the public demo accepts up to 3", status_code=400)
        r = self.upload_as(self.client, name="sheet.csv", content=b"PK\x03\x04\x00\x00\x00binary")
        self.assertContains(r, "binary data", status_code=400)

    def test_upload_rate_limit(self):
        with mock.patch.dict(os.environ, {"DATATRUST_DEMO_UPLOADS_PER_HOUR": "2"}):
            for i in range(2):
                self.assertEqual(self.upload_as(self.client, content=TINY_CSV, name=f"t{i}.csv").status_code, 302)
            r = self.upload_as(self.client, content=TINY_CSV, name="t3.csv")
            self.assertContains(r, "Upload limit reached", status_code=429)
            self.assertEqual(self.upload_as(self.other, content=TINY_CSV, name="o.csv").status_code, 302)

    def test_one_running_analysis_per_visitor_and_a_server_wide_queue_cap(self):
        import threading
        from dashboard.views import jobs
        release = threading.Event()
        mine = jobs.submit(lambda: release.wait(10) and 1, label="slow.csv", owner=self.owner_of(self.client))
        try:
            r = self.upload_as(self.client, content=TINY_CSV, name="t.csv")
            self.assertContains(r, "already have an analysis running", status_code=429)
            with mock.patch.dict(os.environ, {"DATATRUST_DEMO_MAX_PENDING": "1"}):
                r = self.upload_as(self.other, content=TINY_CSV, name="o.csv")
            self.assertContains(r, "busy", status_code=429)
        finally:
            release.set()
            jobs.wait(mine, 10)

    def test_another_visitors_job_page_is_404(self):
        from dashboard.views import jobs
        with mock.patch.dict(os.environ, {"DATATRUST_WEB_BACKGROUND": "1"}):
            r = self.upload_as(self.client, content=TINY_CSV, name="t.csv")
        job_id = r["Location"].rstrip("/").split("/")[-1]
        jobs.wait(job_id, 60)
        self.assertEqual(self.other.get(r["Location"]).status_code, 404)
        self.assertEqual(self.client.get(r["Location"]).status_code, 302)   # the owner is sent to the run

    def test_oversized_request_is_refused_before_reading(self):
        r = self.client.generic("POST", reverse("dashboard:upload"), b"x", content_type="multipart/form-data",
                                CONTENT_LENGTH=str(10 ** 12))
        self.assertEqual(r.status_code, 413)
        self.assertContains(r, "File too large", status_code=413)


class LocalModeUnchangedTests(WebTestCase):
    """Deployment changes nothing on your own computer: no banner, no visitor cookie, one shared history."""

    def test_no_banner_no_cookie_and_everyone_sees_every_run(self):
        r = self.client.get(reverse("dashboard:home"))
        self.assertNotContains(r, "Public demo.")
        self.assertNotIn("datatrust_visitor", r.cookies)
        self.assertNotContains(self.client.get(reverse("dashboard:upload")), "Public demo limits")
        upload = self.upload(content=TINY_CSV, name="shared.csv")
        self.assertEqual(Client().get(upload["Location"]).status_code, 200)

    def test_oversized_request_is_413_here_too(self):
        r = self.client.generic("POST", reverse("dashboard:upload"), b"x", content_type="multipart/form-data",
                                CONTENT_LENGTH=str(10 ** 12))
        self.assertEqual(r.status_code, 413)
