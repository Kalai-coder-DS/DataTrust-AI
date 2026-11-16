"""
tests/test_reporting.py

Phase 5a: the pipeline entry point and the self-contained HTML report.
"""
from __future__ import annotations

import json
import re

import pytest

from datatrust.exceptions import FileValidationError
from datatrust.pipeline import analyze_file
from datatrust.reporting import render_html
from main import run


@pytest.fixture(scope="module")
def dirty_analysis(sample_dir):
    return analyze_file(sample_dir / "customers_dirty.csv", target="churned")


# ------------------------------------------------------------------ pipeline
def test_pipeline_returns_every_section(dirty_analysis):
    assert set(dirty_analysis) == {"meta", "file", "profile", "quality", "statistics", "scores"}
    assert dirty_analysis["meta"]["target"] == "churned"
    assert dirty_analysis["scores"]["trust"]["score"] == pytest.approx(79.8)


def test_pipeline_output_is_json_serialisable(dirty_analysis):
    json.dumps(dirty_analysis)


def test_pipeline_propagates_errors(tmp_path):
    with pytest.raises(FileValidationError):
        analyze_file(tmp_path / "missing.csv")


def test_pipeline_rejects_bad_target(sample_dir):
    with pytest.raises(ValueError):
        analyze_file(sample_dir / "customers_dirty.csv", target="nope")


# ------------------------------------------------------------------ html
def test_html_contains_scores_and_sections(dirty_analysis):
    page = render_html(dirty_analysis)
    assert page.startswith("<!DOCTYPE html>")
    for text in ["customers_dirty.csv", "Data Trust Score", "ML Readiness Score", "79.8",
                 "Top actions", "Data quality issues", "Missing values by column",
                 "Target: churned", "Numeric columns", "Relationships between columns", "Column profile"]:
        assert text in page, text


def test_html_is_self_contained(dirty_analysis):
    """No external scripts, stylesheets, fonts or images: works offline and when emailed."""
    page = render_html(dirty_analysis)
    assert "<script" not in page
    assert not re.search(r'(src|href)\s*=\s*"https?://', page)


def test_severity_never_shown_by_colour_alone(dirty_analysis):
    page = render_html(dirty_analysis)
    assert "! Medium" in page and "• Low" in page   # icon + text label


def test_user_data_is_escaped(tmp_path):
    """A malicious column name or value must appear as text, never as live HTML (XSS)."""
    evil = '<script>alert("x")</script>'
    csv = tmp_path / "evil.csv"
    csv.write_text(f'name,"{evil.replace(chr(34), chr(34) * 2)}"\nA,{1}\nB,2\nC,3\n', encoding="utf-8")
    page = render_html(analyze_file(csv))
    assert "<script>" not in page
    assert "&lt;script&gt;" in page


def test_report_without_target_or_issues(sample_dir):
    page = render_html(analyze_file(sample_dir / "customers_clean.csv"))
    assert "Target:" not in page.split("<main>")[1].split("</header>")[1]  # no target section
    assert "Data Trust Score" in page


def test_cli_writes_html_and_json(sample_dir, tmp_path):
    html_out, json_out = tmp_path / "r" / "report.html", tmp_path / "r" / "report.json"
    code = run([str(sample_dir / "customers_dirty.csv"), "--target", "churned",
                "--html", str(html_out), "--json", str(json_out)])
    assert code == 0
    assert "Data Trust Score" in html_out.read_text(encoding="utf-8")
    assert json.loads(json_out.read_text(encoding="utf-8"))["scores"]["trust"]["grade"] == "C"
