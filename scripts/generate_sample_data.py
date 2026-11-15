"""
scripts/generate_sample_data.py

Generates realistic, reproducible test datasets into data/sample/.

    python scripts/generate_sample_data.py

Files produced:
  customers_clean.csv  - 500 rows, well-formed. Baseline: should score HIGH in later phases.
  customers_dirty.csv  - same schema with KNOWN, documented defects injected,
                         so later phases can be tested against exact expected counts.
  dirty_manifest.json  - the ground truth: exactly which defects were injected where.

Using a fixed random seed means everyone running this gets byte-identical files.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_ROWS = 500
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"

CITIES = ["Chennai", "Bengaluru", "Mumbai", "Delhi", "Hyderabad", "Pune", "Kolkata"]
PLANS = ["Basic", "Standard", "Premium"]


def build_clean(rng: np.random.Generator) -> pd.DataFrame:
    signup = pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 900, N_ROWS), unit="D")
    age = rng.normal(38, 11, N_ROWS).clip(18, 75).round().astype(int)
    income = rng.lognormal(mean=13.2, sigma=0.35, size=N_ROWS).round(-2)  # ~INR 5-6 lakh median
    tenure = rng.integers(1, 60, N_ROWS)
    monthly_spend = (income / 12 * rng.uniform(0.02, 0.08, N_ROWS)).round(2)
    # Churn probability rises with low tenure -> gives later ML phases a real signal.
    churn_p = 1 / (1 + np.exp(0.08 * (tenure - 20)))
    churned = (rng.uniform(size=N_ROWS) < churn_p).astype(int)

    return pd.DataFrame({
        "customer_id": [f"C{10000 + i}" for i in range(N_ROWS)],
        "age": age,
        "gender": rng.choice(["Male", "Female"], N_ROWS),
        "city": rng.choice(CITIES, N_ROWS),
        "plan": rng.choice(PLANS, N_ROWS, p=[0.5, 0.35, 0.15]),
        "annual_income": income,
        "monthly_spend": monthly_spend,
        "tenure_months": tenure,
        "signup_date": signup.strftime("%Y-%m-%d"),
        "is_active": rng.choice(["Yes", "No"], N_ROWS, p=[0.8, 0.2]),
        "churned": churned,
    })


def build_dirty(clean: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    df = clean.copy().astype(object)  # object so we can inject wrong-type values
    manifest: dict = {}

    used: set[int] = set()

    def pick(n: int) -> list[int]:
        """Pick n rows not touched by any earlier defect, so every defect is counted exactly
        once (e.g. a city can't be both blanked AND lower-cased, and duplicated rows
        never copy a planted defect)."""
        pool = np.setdiff1d(np.arange(N_ROWS), sorted(used))
        rows = sorted(int(i) for i in rng.choice(pool, n, replace=False))
        used.update(rows)
        return rows

    # Missing values (true NaN)
    miss_income = pick(40)
    df.loc[miss_income, "annual_income"] = np.nan
    miss_city = pick(25)
    df.loc[miss_city, "city"] = np.nan
    manifest["missing"] = {"annual_income": len(miss_income), "city": len(miss_city)}

    # Disguised missing values (placeholders that are NOT parsed as NaN by default)
    disguised = pick(8)
    df.loc[disguised, "gender"] = "Unknown"
    manifest["disguised_missing"] = {"gender": {"value": "Unknown", "count": len(disguised)}}

    # Invalid values: wrong type and impossible ranges
    bad_age_text = pick(5)
    df.loc[bad_age_text, "age"] = "abc"
    bad_age_neg = pick(3)
    df.loc[bad_age_neg, "age"] = -5
    bad_age_high = pick(2)
    df.loc[bad_age_high, "age"] = 210
    bad_dates = pick(6)
    df.loc[bad_dates, "signup_date"] = "2023-02-30"  # impossible date
    manifest["invalid"] = {
        "age_non_numeric": len(bad_age_text),
        "age_negative": len(bad_age_neg),
        "age_over_120": len(bad_age_high),
        "signup_date_impossible": len(bad_dates),
    }

    # Inconsistent categories (case / whitespace)
    inconsistent = pick(12)
    df.loc[inconsistent, "city"] = df.loc[inconsistent, "city"].str.lower()
    manifest["inconsistent_case_city"] = len(inconsistent)

    # Outliers in monthly_spend (extreme but numerically valid)
    outliers = pick(7)
    df.loc[outliers, "monthly_spend"] = [250000.0, 310000.0, 275500.0, 400000.0, 520000.0, 290000.0, 333333.0]
    manifest["outliers"] = {"monthly_spend": len(outliers)}
    manifest["outlier_rows"] = {"monthly_spend": outliers}

    # Constant column (zero information for ML)
    df["country"] = "India"
    manifest["constant_columns"] = ["country"]

    # Exact duplicate rows appended at the end (copied from untouched rows only)
    dup_src = pick(15)
    df = pd.concat([df, df.loc[dup_src]], ignore_index=True)
    manifest["exact_duplicate_rows"] = len(dup_src)
    manifest["total_rows"] = int(len(df))
    return df, manifest


def main() -> None:
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clean = build_clean(rng)
    dirty, manifest = build_dirty(clean, rng)

    clean.to_csv(OUT_DIR / "customers_clean.csv", index=False)
    dirty.to_csv(OUT_DIR / "customers_dirty.csv", index=False)
    (OUT_DIR / "dirty_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(clean)} clean rows and {len(dirty)} dirty rows to {OUT_DIR}")


if __name__ == "__main__":
    main()
