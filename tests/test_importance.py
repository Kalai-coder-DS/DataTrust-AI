"""
tests/test_importance.py

Phase 8: what the model relies on, joined with the data problems. Files with KNOWN causes:
  - the dirty sample: churned was generated from tenure_months ONLY (scripts/generate_sample_data.py)
  - y = 2a + b + noise: 'a' must rank above 'b', and an unrelated column must not be relied on
  - an important column full of -999 placeholders -> "Fix first"; an unrelated column with
    missing values -> "Can wait"
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from datatrust.config import Settings
from datatrust.ingestion import CSVLoader
from datatrust.ml import run_baseline
from datatrust.ml.importance import column_problems, explain
from datatrust.ml.present import importance_view
from datatrust.pipeline import analyze_file
from datatrust.reporting import render_html

FAST = Settings(ml_cv_repeats=1, ml_importance_repeats=3)


def run(path, target, settings=FAST):
    a = analyze_file(path, target=target)
    return a, run_baseline(CSVLoader().load(path).df, a, target, settings)


def write(tmp_path, name, df):
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def dirty(sample_dir):
    return run(sample_dir / "customers_dirty.csv", "churned")


def test_finds_the_one_real_cause_in_the_sample(dirty):
    _, ml = dirty
    cols = ml["importance"]["columns"]
    assert cols[0]["column"] == "tenure_months" and cols[0]["relied_on"]
    assert [c["column"] for c in cols if c["relied_on"]] == ["tenure_months"]
    assert "customer_id" not in [c["column"] for c in cols]           # IDs are never features
    assert "country" not in [c["column"] for c in cols]               # dropped by cleaning (constant)
    assert "one column only, 'tenure_months'" in ml["importance"]["verdict"][0]


def test_problems_are_joined_from_the_report(dirty):
    a, ml = dirty
    by_name = {c["column"]: c for c in ml["importance"]["columns"]}
    assert [p["issue_type"] for p in by_name["gender"]["problems"]] == ["disguised_missing"]
    assert {p["issue_type"] for p in by_name["city"]["problems"]} == {"inconsistent_format", "missing_values"}
    assert by_name["tenure_months"]["problems"] == []
    can_wait = next(v for v in ml["importance"]["verdict"] if v.startswith("Can wait"))
    assert "'gender'" in can_wait or "more" in can_wait


def test_column_problems_skip_dataset_and_column_level_issues(dirty):
    a, _ = dirty
    assert column_problems(a, "country") == []        # constant_column is about the column, not its values
    sev = [p["severity"] for p in column_problems(a, "age")]
    order = ["critical", "high", "medium", "low"]
    assert sev == sorted(sev, key=order.index)


def test_ranks_stronger_cause_first(tmp_path):
    rng = np.random.default_rng(10)
    n = 400
    a, b = rng.normal(size=n), rng.normal(size=n)
    df = pd.DataFrame({"a": a.round(3), "b": b.round(3), "unrelated": rng.normal(size=n).round(3),
                       "y": (2 * a + b + rng.normal(scale=1.5, size=n)).round(3)})
    _, ml = run(write(tmp_path, "rank.csv", df), "y")
    cols = ml["importance"]["columns"]
    assert [c["column"] for c in cols[:2]] == ["a", "b"]
    assert cols[0]["importance"] > cols[1]["importance"] > 0
    assert not next(c for c in cols if c["column"] == "unrelated")["relied_on"]
    assert "2 of 3 columns: 'a', 'b'" in ml["importance"]["verdict"][0]


def test_fix_first_and_can_wait(tmp_path):
    rng = np.random.default_rng(11)
    n = 400
    x = rng.uniform(0, 10, n)
    y = (x + rng.normal(scale=2.5, size=n)).round(2)
    x = x.round(2)
    x[rng.choice(n, 40, replace=False)] = -999                  # 10% placeholders in the IMPORTANT column
    noise = rng.normal(size=n).round(3).astype(object)
    noise[rng.choice(n, 60, replace=False)] = None               # missing values in an UNRELATED column
    df = pd.DataFrame({"x": x, "noise": noise, "y": y})
    a, ml = run(write(tmp_path, "fix.csv", df), "y")
    by_name = {c["column"]: c for c in ml["importance"]["columns"]}
    assert by_name["x"]["relied_on"] and by_name["x"]["problems"]
    assert not by_name["noise"]["relied_on"] and by_name["noise"]["problems"]
    verdict = " ".join(ml["importance"]["verdict"])
    assert "Fix first" in verdict and "'x' (#1" in verdict
    assert "Can wait (for this model): 'noise'" in verdict
    view = importance_view(ml)
    assert view["rows"][0]["fix_first"] and not view["rows"][1]["fix_first"]


def test_suspected_leak_on_top_is_called_out(tmp_path):
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 10, 300)
    df = pd.DataFrame({"x": x.round(3), "group": rng.choice(["a", "b", "c"], 300),
                       "price": (3 * x + rng.normal(scale=1, size=300)).round(2)})
    _, ml = run(write(tmp_path, "leak.csv", df), "price")
    assert ml["importance"]["columns"][0]["suspect"]
    assert "flagged as possible target leakage" in ml["importance"]["verdict"][-1]


def test_explain_when_nothing_is_relied_on():
    imp = {"folds": 5, "repeats": 5,
           "columns": [{"column": "a", "importance": 0.001, "se": 0.002, "relied_on": False}]}
    out = explain(imp, {"quality": {"issues": []}}, "R²", [])
    assert out["verdict"][0].startswith("No single column makes a measurable difference")


def test_view_scales_bars_and_hides_the_tail():
    imp = {"metric_label": "R²", "folds": 5, "repeats": 5, "verdict": [], "note": "",
           "columns": [{"rank": i + 1, "column": f"c{i}", "importance": 0.5 - i * 0.03, "se": 0.0,
                        "relied_on": True, "problems": [], "suspect": False} for i in range(15)]
                      + [{"rank": 16, "column": "neg", "importance": -0.01, "se": 0.0,
                          "relied_on": False, "problems": [], "suspect": False}]}
    v = importance_view({"importance": imp}, top=12)
    assert v["rows"][0]["width"] == 100.0 and len(v["rows"]) == 12 and v["hidden"] == 4
    full = importance_view({"importance": imp}, top=20)
    assert full["rows"][-1]["width"] == 0.0 and full["rows"][-1]["text"] == "-0.010"
    assert importance_view(None) is None and importance_view({"status": "skipped"}) is None


def test_report_and_cli(dirty, sample_dir, capsys):
    a, ml = dirty
    html = render_html({**a, "ml": ml})
    assert "What the model relies on" in html and "tenure_months" in html
    assert "What the model relies on" not in render_html(a)
    from main import run as cli
    assert cli([str(sample_dir / "customers_clean.csv"), "--target", "churned", "--baseline"]) == 0
    assert "WHAT THE MODEL RELIES ON" in capsys.readouterr().out


def test_same_input_same_importance(dirty, sample_dir):
    _, again = run(sample_dir / "customers_dirty.csv", "churned")
    assert again["importance"]["columns"] == dirty[1]["importance"]["columns"]


@pytest.mark.parametrize("seed", [1, 2])
def test_random_columns_are_not_called_relied_on_whatever_the_seed(sample_dir, seed):
    """Found in testing: with a 2-standard-error rule, 'plan' (drawn at random) was 'relied on'."""
    _, ml = run(sample_dir / "customers_dirty.csv", "churned", Settings(ml_random_state=seed))
    assert [c["column"] for c in ml["importance"]["columns"] if c["relied_on"]] == ["tenure_months"]
