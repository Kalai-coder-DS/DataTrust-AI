"""
tests/test_ml.py

Phase 7 baseline model. Synthetic files with KNOWN answers:
  - every row repeated 4x with a pure-noise target  -> raw score inflated, cleaned ~ guessing
  - a column that copies the target                 -> raw ~1.0, cleaned drops it
  - y = 3x + noise                                  -> strong R², guessing ~0
  - pure noise                                      -> "barely better than guessing"
and the dirty sample file, whose defects are listed exactly in dirty_manifest.json.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from datatrust.config import Settings
from datatrust.ingestion import CSVLoader
from datatrust.ml import prepare, run_baseline
from datatrust.ml.present import baseline_view, f3
from datatrust.pipeline import analyze_file
from datatrust.reporting import render_html

FAST = Settings(ml_cv_repeats=1)       # one round of folds: enough for effects this large


def analysed(path, target, settings=FAST):
    a = analyze_file(path, target=target)
    return a, CSVLoader().load(path).df


@pytest.fixture(scope="module")
def dirty(sample_dir):
    path = sample_dir / "customers_dirty.csv"
    return analysed(path, "churned")


@pytest.fixture(scope="module")
def dirty_ml(dirty):
    a, df = dirty
    return run_baseline(df, a, "churned")          # default settings: 5 folds x 3 repeats


def write(tmp_path, name, df):
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


# ------------------------------------------------------------ preparation
def test_raw_keeps_defects_cleaned_fixes_exactly_what_the_report_found(dirty, dirty_manifest):
    a, df = dirty
    raw = prepare(df, a, "churned", clean=False)
    cleaned = prepare(df, a, "churned", clean=True)
    assert raw.rows_used == 515 and raw.steps == []
    assert "Unknown" in set(raw.X["gender"].dropna())
    assert (raw.X["age"] > 120).sum() == 2 and (raw.X["age"] < 0).sum() == 3
    assert {"chennai", "Chennai"} <= set(raw.X["city"].dropna())

    assert cleaned.rows_used == 515 - 15
    text = " ".join(cleaned.steps)
    for expected in ["country", "Removed 15 duplicate row(s)", "Set 8 placeholder", "Set 5 impossible",
                     "Merged 12 label(s)"]:
        assert expected in text, (expected, cleaned.steps)
    assert "Unknown" not in set(cleaned.X["gender"].dropna())
    assert cleaned.X["age"].dropna().between(0, 120).all()
    assert "chennai" not in set(cleaned.X["city"].dropna())
    assert "country" not in cleaned.X and "country" in raw.X


def test_no_pandas_NA_reaches_scikit_learn(dirty):
    """Found on Windows (pandas 3.1.1): merged spellings kept pd.NA -> 'boolean value of NA is ambiguous'."""
    a, df = dirty
    for clean in (False, True):
        p = prepare(df, a, "churned", clean=clean)
        for col in p.categorical:
            assert p.X[col].dtype == object
            assert not any(v is pd.NA for v in p.X[col]), col
            assert all(isinstance(v, str) or (isinstance(v, float) and np.isnan(v)) for v in p.X[col]), col
        assert p.y.dtype == object and all(isinstance(v, str) for v in p.y)
    # The column that triggered it: 'city' after merging spellings still has real missing values.
    cleaned = prepare(df, a, "churned", clean=True)
    assert cleaned.X["city"].isna().sum() > 0


def test_ids_and_target_never_used_as_features(dirty):
    a, df = dirty
    for clean in (False, True):
        p = prepare(df, a, "churned", clean=clean)
        assert "customer_id" not in p.X and "churned" not in p.X
        assert {"column": "customer_id", "reason": "identifier (unique per row, no pattern to learn)"} in p.excluded


def test_rows_without_target_are_dropped_not_imputed(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=60).round(3), "y": rng.choice(["a", "b"], 60).astype(object)})
    df.loc[:4, "y"] = None
    df.loc[5:6, "y"] = "Unknown"
    a, data = analysed(write(tmp_path, "t.csv", df), "y")
    p = prepare(data, a, "y", clean=False)
    assert p.rows_used == 60 - 7 and set(p.y) == {"a", "b"}


# ------------------------------------------------------------ known effects
def test_duplicates_inflate_the_raw_score(tmp_path):
    rng = np.random.default_rng(1)
    base = pd.DataFrame({"a": rng.normal(size=150).round(4), "b": rng.normal(size=150).round(4),
                         "c": rng.choice(list("xyz"), 150), "target": rng.choice(["yes", "no"], 150)})
    df = pd.concat([base] * 4, ignore_index=True).sample(frac=1, random_state=0)
    a, data = analysed(write(tmp_path, "dups.csv", df), "target")
    r = run_baseline(data, a, "target", FAST)
    assert r["raw"]["score"] > 0.9                 # memorised: copies of each test row were in training
    assert r["cleaned"]["score"] < 0.65            # pure-noise target: close to guessing once de-duplicated
    assert r["change"] < -r["noise"]
    assert any("duplicate rows" in v and "honest" in v for v in r["verdict"])
    assert "barely better than guessing" in r["verdict"][0]


def test_leaking_column_inflates_the_raw_score(tmp_path):
    rng = np.random.default_rng(2)
    x = rng.normal(size=300)
    y = (x + rng.normal(scale=2, size=300) > 0).astype(int)
    df = pd.DataFrame({"x": x.round(4), "noise": rng.normal(size=300).round(4), "outcome_code": y, "target": y})
    a, data = analysed(write(tmp_path, "leak.csv", df), "target")
    r = run_baseline(data, a, "target", FAST)
    assert r["raw"]["score"] > 0.98
    # A suspected leak is NOT dropped by cleaning (a person must confirm); it is measured instead.
    assert r["cleaned"]["score"] > 0.98
    w = r["without_suspects"]
    assert w["columns"] == ["outcome_code"] and w["score"] < 0.8
    assert {"column": "outcome_code", "reason": "left out to measure its effect: possible target leakage"} in w["excluded"]
    assert "Most of the model's skill comes from 'outcome_code'" in r["verdict"][-1]
    assert "known BEFORE 'target'" in r["verdict"][-1]


def test_regression_signal(tmp_path):
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 10, 300)
    df = pd.DataFrame({"x": x.round(3), "group": rng.choice(["a", "b", "c"], 300),
                       "price": (3 * x + rng.normal(scale=1, size=300)).round(2)})
    a, data = analysed(write(tmp_path, "reg.csv", df), "price")
    r = run_baseline(data, a, "price", FAST)
    assert r["task"] == "regression" and r["metric"]["key"] == "r2"
    assert r["cleaned"]["score"] > 0.85
    assert abs(r["cleaned"]["guess"]) < 0.05
    assert 0 < r["cleaned"]["secondary"] < 2                         # MAE in price units, positive
    assert r["cleaned"]["secondary"] < r["cleaned"]["guess_secondary"]
    assert "The report found nothing to clean, so both runs use the same data." in r["verdict"]


def test_strong_genuine_predictor_is_not_thrown_away(tmp_path):
    """Found in testing: x in price = 3x + noise is flagged as possible leakage (association >= 0.95).
    Dropping it automatically took R² from 0.97 to 0. Cleaning keeps it; the third run shows the effect."""
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 10, 300)
    df = pd.DataFrame({"x": x.round(3), "group": rng.choice(["a", "b", "c"], 300),
                       "price": (3 * x + rng.normal(scale=1, size=300)).round(2)})
    a, data = analysed(write(tmp_path, "reg.csv", df), "price")
    r = run_baseline(data, a, "price", FAST)
    assert r["without_suspects"]["columns"] == ["x"]
    assert r["cleaned"]["score"] > 0.85 and r["without_suspects"]["score"] < 0.2
    assert "If 'x' is known BEFORE 'price' is decided, this is genuine signal" in r["verdict"][-1]


def test_pure_noise_is_not_called_signal(tmp_path):
    rng = np.random.default_rng(4)
    df = pd.DataFrame({"a": rng.normal(size=200).round(3), "b": rng.normal(size=200).round(3),
                       "target": rng.choice(["yes", "no"], 200)})
    a, data = analysed(write(tmp_path, "noise.csv", df), "target")
    r = run_baseline(data, a, "target")
    assert "barely better than guessing" in r["verdict"][0]


# ------------------------------------------------------------ dirty sample (default settings)
def test_dirty_sample_verdict_is_honest_about_noise(dirty_ml):
    r = dirty_ml
    assert r["status"] == "ok" and r["task"] == "classification"
    assert r["raw"]["repeats"] == 3 and len(r["raw"]["fold_scores"]) == 15
    assert 0.6 < r["cleaned"]["score"] < 0.8 and r["cleaned"]["guess"] == 0.5
    assert abs(r["change"]) <= r["noise"]             # stable across seeds: see next test
    assert "within the run-to-run noise" in r["verdict"][1]
    assert "duplicate rows" in r["verdict"][2]


@pytest.mark.parametrize("seed", [1, 2])
def test_dirty_sample_verdict_does_not_depend_on_the_seed(dirty, seed):
    a, df = dirty
    r = run_baseline(df, a, "churned", Settings(ml_random_state=seed))
    assert "within the run-to-run noise" in r["verdict"][1]


def test_same_input_same_scores(dirty, dirty_ml):
    a, df = dirty
    again = run_baseline(df, a, "churned")
    assert again["raw"]["fold_scores"] == dirty_ml["raw"]["fold_scores"]
    assert again["cleaned"]["score"] == dirty_ml["cleaned"]["score"]


def test_big_files_are_sampled_by_class(dirty):
    a, df = dirty
    r = run_baseline(df, a, "churned", Settings(ml_max_rows=200, ml_cv_repeats=1))
    assert abs(r["sampled_rows"] - 200) <= 2 and r["rows_in_file"] == 515
    assert r["raw"]["rows"] == r["sampled_rows"]
    assert r["raw"]["repeats"] == 1


# ------------------------------------------------------------ skips (never crash)
def test_no_target_is_skipped(dirty):
    a, df = dirty
    assert run_baseline(df, a, None) == {"status": "skipped", "reason": "A baseline model needs a target column."}


def test_too_few_rows_is_skipped(tmp_path):
    df = pd.DataFrame({"x": np.random.default_rng(6).normal(size=20).round(3), "y": ["a", "b"] * 10})
    a, data = analysed(write(tmp_path, "small.csv", df), "y")
    r = run_baseline(data, a, "y", FAST)
    assert r["status"] == "skipped" and "at least 30" in r["reason"]


def test_no_usable_features_is_skipped(tmp_path):
    df = pd.DataFrame({"id": [f"ID{i:04d}" for i in range(80)], "y": ["a", "b"] * 40})
    a, data = analysed(write(tmp_path, "ids.csv", df), "y")
    r = run_baseline(data, a, "y", FAST)
    assert r["status"] == "skipped" and "No usable feature" in r["reason"]


# ------------------------------------------------------------ pipeline, report, storage, CLI
def test_pipeline_option_and_backward_compatibility(sample_dir):
    path = sample_dir / "customers_clean.csv"
    assert "ml" not in analyze_file(path)                         # unchanged default: no model
    named = analyze_file(path, name="renamed.csv")
    assert named["file"]["file_name"] == "renamed.csv"


def test_report_section(dirty, dirty_ml):
    a, _ = dirty
    html = render_html({**a, "ml": dirty_ml})
    assert "Baseline model" in html and "After cleaning" in html and "Removed 15 duplicate row(s)" in html
    assert "Baseline model" not in render_html(a)
    skipped = render_html({**a, "ml": {"status": "skipped", "reason": "A baseline model needs a target column."}})
    assert "Not trained: A baseline model needs a target column." in skipped


def test_report_escapes_column_names_in_the_model_section(tmp_path):
    rng = np.random.default_rng(5)
    evil = "<script>alert(1)</script>"
    y = rng.choice([0, 1], 120)
    df = pd.DataFrame({"x": rng.normal(size=120).round(3), evil: y, "target": y})   # evil column leaks
    a, data = analysed(write(tmp_path, "evil.csv", df), "target")
    html = render_html({**a, "ml": run_baseline(data, a, "target", FAST)})
    assert evil not in html and "&lt;script&gt;" in html


def test_view_formats_negative_r2_and_skips():
    assert f3(-0.0003) == "0.000" and f3(-0.25) == "-0.250"
    assert baseline_view(None) is None
    assert baseline_view({"status": "skipped", "reason": "x"}) == {"status": "skipped", "reason": "x"}


def test_stored_analysis_keeps_the_model_result(dirty, dirty_ml, tmp_path):
    from datatrust.storage import AnalysisRepository, Database, DBConfig
    a, _ = dirty
    full = {**a, "ml": dirty_ml}
    db = Database(DBConfig(backend="sqlite", sqlite_path=tmp_path / "ml.db")).connect()
    repo = AnalysisRepository(db)
    repo.init_schema()
    run_id = repo.save(json.loads(json.dumps(full, default=str)))
    assert repo.get_analysis(run_id)["ml"]["cleaned"]["score"] == dirty_ml["cleaned"]["score"]
    assert render_html(repo.get_analysis(run_id)) == render_html(json.loads(json.dumps(full, default=str)))
    db.close()


def test_cli_baseline(sample_dir, capsys):
    from main import run
    assert run([str(sample_dir / "customers_clean.csv"), "--baseline"]) == 2
    assert "--baseline needs --target" in capsys.readouterr().err
    assert run([str(sample_dir / "customers_clean.csv"), "--target", "churned", "--baseline"]) == 0
    out = capsys.readouterr().out
    assert "BASELINE MODEL" in out and "After cleaning" in out
