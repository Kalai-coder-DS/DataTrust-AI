"""
datatrust/config.py

Central configuration for DataTrust AI.

Why a single config module?
- Every threshold (file size limit, categorical cut-off, etc.) lives in ONE place,
  so behaviour is tunable without hunting through code.
- Values can be overridden by environment variables (12-factor style), which is
  how the app will later be configured in Django / Docker without code changes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Project root = folder that contains the `datatrust` package.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable, falling back safely on bad input."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable application settings. Create a custom instance in tests."""

    # ---------- Ingestion ----------
    max_file_size_mb: int = field(default_factory=lambda: _env_int("DATATRUST_MAX_FILE_MB", 200))
    allowed_extensions: tuple[str, ...] = (".csv",)
    # Tried in order; latin-1 never fails, so it is the last-resort fallback.
    encodings_to_try: tuple[str, ...] = ("utf-8-sig", "utf-8", "cp1252", "latin-1")
    candidate_delimiters: str = ",;\t|"
    sniff_sample_bytes: int = 64 * 1024

    # ---------- Profiling ----------
    # Max distinct values for a text column to be treated as categorical.
    categorical_max_unique: int = field(default_factory=lambda: _env_int("DATATRUST_CAT_MAX_UNIQUE", 50))
    # ...or if distinct values are at most this share of non-null rows.
    categorical_max_unique_ratio: float = field(default_factory=lambda: _env_float("DATATRUST_CAT_MAX_RATIO", 0.05))
    # Share of values that must parse as numbers/dates to infer that type for a text column.
    type_inference_threshold: float = 0.90
    # Rows sampled per column for expensive parsing checks (keeps profiling fast on big files).
    inference_sample_size: int = 10_000
    # Average string length above which an all-unique text column is "free text", not an ID.
    free_text_min_avg_length: int = 30
    sample_values_count: int = 5
    # A column named like an ID (customer_id) stays an identifier down to this uniqueness.
    id_min_unique_ratio: float = 0.95

    # ---------- Quality checks (Phase 2) ----------
    # Placeholder strings that mean "missing" but are not blank. pandas already converts
    # '', 'NA', 'N/A', 'null', 'NaN', 'None' to NaN on load, so these are the ones it misses.
    disguised_missing_tokens: tuple[str, ...] = (
        "unknown", "?", "-", "--", "missing", "not available", "not applicable",
        "undefined", "nil", "none given", "tbd", "#n/a", "na.", "n.a.",
    )
    # Numeric codes commonly used to mean "no value" in exported systems.
    sentinel_numbers: tuple[float, ...] = (-999.0, -9999.0, -99999.0)
    # Severity bands by % of rows affected: <5 LOW, <20 MEDIUM, <50 HIGH, else CRITICAL.
    severity_thresholds: tuple[float, float, float] = (5.0, 20.0, 50.0)
    # Domain range rules: (name token, min, max). Applied when a column name contains the token.
    range_rules: tuple[tuple[str, float, float], ...] = (
        ("age", 0, 120),
        ("percent", 0, 100),
        ("percentage", 0, 100),
        ("pct", 0, 100),
    )
    # Columns whose names contain these tokens can never be negative.
    non_negative_tokens: tuple[str, ...] = (
        "age", "income", "salary", "price", "cost", "amount", "spend", "revenue", "sales",
        "quantity", "qty", "count", "tenure", "weight", "height", "length", "width",
        "distance", "duration", "years", "months", "days", "bmi",
    )
    # Outliers: classic Tukey fence and the Iglewicz-Hoaglin modified z-score cut-off.
    iqr_multiplier: float = 1.5
    modified_z_threshold: float = 3.5
    min_rows_for_outliers: int = 10
    # A column whose most common value covers at least this share is "near-constant".
    near_constant_ratio: float = 0.99
    max_examples: int = 5

    # ---------- Statistics (Phase 3) ----------
    # Associations at or above this strength are listed in the report.
    association_report_threshold: float = 0.5
    # Two numeric columns this correlated carry almost the same information.
    redundancy_threshold: float = 0.9
    # Caps the number of columns compared pairwise (pairs grow as n^2).
    max_columns_for_associations: int = 50
    # A numeric target with more distinct values than this is treated as regression.
    max_classes_for_classification: int = 20
    # Categories below this share of rows are "rare" (models learn little from them).
    rare_category_pct: float = 1.0

    # ---------- Scoring (Phase 4) ----------
    # Points deducted from a dimension per issue, by severity.
    severity_points: tuple[tuple[str, float], ...] = (
        ("low", 4.0), ("medium", 10.0), ("high", 20.0), ("critical", 35.0),
    )
    # Data Trust Score dimension weights (must sum to 1).
    trust_weights: tuple[tuple[str, float], ...] = (
        ("completeness", 0.30), ("uniqueness", 0.20), ("validity", 0.30), ("consistency", 0.20),
    )
    # ML Readiness Score dimension weights (must sum to 1; renormalised if no target is given).
    ml_weights: tuple[tuple[str, float], ...] = (
        ("data_trust", 0.40), ("sample_size", 0.15), ("feature_quality", 0.20), ("target", 0.25),
    )
    # A feature associated with the target at least this strongly may be leaking the answer.
    leakage_threshold: float = 0.95

    # ---------- Phase 7: baseline model ----------
    # Rows used for training (a random, class-stratified sample above this): keeps big files fast.
    ml_max_rows: int = field(default_factory=lambda: _env_int("DATATRUST_ML_MAX_ROWS", 20_000))
    ml_cv_folds: int = 5            # cross-validation folds
    # Repeat the k folds with new shuffles (small files only): steadier scores. The public demo
    # server sets 1 (DATATRUST_ML_CV_REPEATS) because its CPU is about 10x slower than a laptop.
    ml_cv_repeats: int = field(default_factory=lambda: _env_int("DATATRUST_ML_CV_REPEATS", 3))
    ml_repeat_max_rows: int = 5_000 # above this, one round of k folds (big files are steady and slow)
    ml_min_rows: int = 30           # fewer usable rows than this: no model is trained
    ml_random_state: int = 0        # same data -> same scores, every run
    ml_max_categories: int = 20     # one-hot columns per categorical feature (rarer labels grouped)
    # Phase 8: shuffles per column per test fold (the demo server sets 2: DATATRUST_ML_IMPORTANCE_REPEATS).
    ml_importance_repeats: int = field(default_factory=lambda: _env_int("DATATRUST_ML_IMPORTANCE_REPEATS", 5))

    # ---------- Logging ----------
    log_dir: Path = PROJECT_ROOT / "logs"
    log_level: str = field(default_factory=lambda: os.getenv("DATATRUST_LOG_LEVEL", "INFO").upper())

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


# Default shared instance. Modules accept a Settings argument so tests can inject their own.
DEFAULT_SETTINGS = Settings()
