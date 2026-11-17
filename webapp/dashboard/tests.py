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
