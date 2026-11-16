"""
datatrust/ml/present.py

Turns the stored baseline result (analysis["ml"]) into display-ready values, so the HTML report
and the Django run page show it identically. Pure data in, plain values out: no HTML here.
"""
from __future__ import annotations


def f3(value: float) -> str:
    """3 decimals, and never '-0.000' (a tiny negative R² like -0.0003 is just 0)."""
    text = f"{value:.3f}"
    return "0.000" if text == "-0.000" else text


def _fmt(value, key: str) -> str:
    if value is None:
        return "-"
    if key == "neg_mean_absolute_error":
        return f"{value:,.2f}"
    return f3(value)


def _width(value: float | None) -> float:
    """Bar length in % for a 0-1 metric. R² can be negative: the bar is empty and the number says so."""
    if value is None:
        return 0.0
    return round(max(0.0, min(1.0, float(value))) * 100, 1)


def baseline_view(ml: dict | None) -> dict | None:
    if not ml:
        return None
    if ml.get("status") != "ok":
        return {"status": "skipped", "reason": ml.get("reason", "The baseline model was not trained.")}

    key, skey = ml["metric"]["key"], ml["secondary_metric"]["key"]
    raw, cleaned, without = ml["raw"], ml["cleaned"], ml.get("without_suspects")
    runs = [("Raw data", raw), ("After cleaning", cleaned)]
    if without:
        runs.append(("Without " + ", ".join(without["columns"]), without))
    trusted = cleaned if cleaned["status"] == "ok" else raw

    bars = [{"label": "Guessing", "value": trusted["guess"], "text": _fmt(trusted["guess"], key),
             "width": _width(trusted["guess"]), "note": "a model that ignores every feature"}]
    for label, run in runs:
        if run["status"] == "ok":
            bars.append({"label": label, "value": run["score"], "text": _fmt(run["score"], key),
                         "width": _width(run["score"]),
                         "note": f"± {run['score_std']:.3f} across {len(run['fold_scores'])} test folds"})

    def cell(run: dict, fmt) -> str:
        return fmt(run) if run["status"] == "ok" else f"not trained: {run['reason']}"

    rows = [
        ("Rows used", lambda r: f"{r['rows']:,}", False),
        ("Feature columns", lambda r: str(r["features"]), False),
        (ml["metric"]["label"], lambda r: f"{_fmt(r['score'], key)} ± {f3(r['score_std'])}", True),
        (ml["secondary_metric"]["label"], lambda r: _fmt(r["secondary"], skey), True),
        ("Score when guessing", lambda r: _fmt(r["guess"], key), True),
    ]
    columns = [label for label, _ in runs]
    table = [{"label": label,
              "cells": [cell(run, fmt) if needs_ok else fmt(run) for _, run in runs]}
             for label, fmt, needs_ok in rows]

    cv = ""
    if trusted.get("folds"):
        cv = f"{trusted['folds']}-fold cross-validation" + (
            f", repeated {trusted['repeats']}×" if trusted.get("repeats", 1) > 1 else "")
    footnote = " · ".join(p for p in [
        ml["model"], cv,
        f"trained on a random sample of {ml['sampled_rows']:,} of {ml['rows_in_file']:,} rows"
        if ml.get("sampled_rows") else "",
        f"scikit-learn {ml.get('sklearn_version', '?')}",
    ] if p)

    change = ml.get("change")
    if not cleaned.get("steps") and not any(e["reason"].startswith("dropped by cleaning")
                                            for e in cleaned.get("excluded", [])):
        change = None                      # nothing was cleaned: "+0.000" would only add noise
    return {
        "status": "ok",
        "reason": None,
        "target": ml["target"],
        "task": ml["task"],
        "metric_label": ml["metric"]["label"],
        "metric_about": ml["metric"]["about"],
        "bars": bars,
        "columns": columns,
        "table": table,
        "change_text": (None if change is None else f"{'+' if change >= 0 else '-'}{f3(abs(change))} (noise ±{f3(ml['noise'])})"),
        "verdict": ml.get("verdict", []),
        "steps": cleaned.get("steps", []),
        "excluded": [e for e in cleaned.get("excluded", []) if not e["reason"].startswith("dropped by cleaning")],
        "dropped": [e for e in cleaned.get("excluded", []) if e["reason"].startswith("dropped by cleaning")],
        "notes": sorted(set(raw.get("notes", []) + cleaned.get("notes", []))),
        "footnote": footnote,
    }
