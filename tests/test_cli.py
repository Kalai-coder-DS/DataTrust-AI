"""
tests/test_cli.py

End-to-end tests of main.py: file in -> report out, plus exit codes.
"""
from __future__ import annotations

import json

from main import run


def test_cli_success_prints_report(sample_dir, capsys):
    code = run([str(sample_dir / "customers_clean.csv")])
    out = capsys.readouterr().out
    assert code == 0
    assert "customers_clean.csv" in out
    assert "customer_id" in out


def test_cli_writes_json(sample_dir, tmp_path):
    out_file = tmp_path / "reports" / "profile.json"
    assert run([str(sample_dir / "customers_dirty.csv"), "--json", str(out_file)]) == 0
    data = json.loads(out_file.read_text())
    assert data["profile"]["n_rows"] == 515
    assert data["file"]["file_name"] == "customers_dirty.csv"
    assert data["quality"]["total_issues"] > 0


def test_cli_prints_quality_section(sample_dir, capsys):
    assert run([str(sample_dir / "customers_dirty.csv")]) == 0
    out = capsys.readouterr().out
    assert "DATA QUALITY ISSUES" in out
    assert "impossible" in out


def test_cli_missing_file_returns_error_code(tmp_path, capsys):
    code = run([str(tmp_path / "nope.csv")])
    assert code == 1
    assert "ERROR" in capsys.readouterr().err
