"""
datatrust/genai/facts.py

The ONLY input every summary provider receives: a small, JSON-serialisable set of facts copied
from the analysis the Python engine produced (Phases 1-8). Nothing here is computed by an AI,
and no provider sees the raw data - only these numbers and messages.

Every number in the facts is also collected into `allowed_numbers`, which the AI output is
checked against (see guard.py): an AI text that mentions a number not in the facts is rejected.
"""
from __future__ import annotations

import re

MAX_ISSUES = 8
MAX_ACTIONS = 5
SEVERITY_ORDER = ("critical", "high", "medium", "low")


GRADE_MEANING = {"A": "excellent", "B": "good", "C": "usable with fixes", "D": "poor", "F": "not usable as-is"}


def _score(s: dict) -> dict:
    return {"score": s["score"], "grade": s["grade"], "grade_means": GRADE_MEANING.get(s["grade"], ""),
            "dimensions": {d["name"]: d["score"] for d in s["dimensions"]}}


def build_facts(analysis: dict) -> dict:
    q = analysis["quality"]
    issues = sorted(q["issues"], key=lambda i: SEVERITY_ORDER.index(i["severity"]))[:MAX_ISSUES]
    facts = {
        "dataset": analysis["file"]["file_name"],
        "rows": analysis["file"]["rows"],
        "columns": analysis["file"]["columns"],
        "target": analysis["meta"].get("target"),
        "data_trust": _score(analysis["scores"]["trust"]),
        "ml_readiness": _score(analysis["scores"]["ml_readiness"]),
        "issue_counts": {**{s: q["severity_counts"].get(s, 0) for s in SEVERITY_ORDER}, "total": q["total_issues"]},
        "issues_listed_below": len(issues),
        "top_issues": [{"severity": i["severity"], "column": i["column"], "message": i["message"],
                        "affected_rows": i["affected_rows"]} for i in issues],
        "top_actions": [{"gain": a["impact"], "problem": a["reason"], "action": a["recommendation"]}
                        for a in analysis["scores"]["ml_readiness"]["top_actions"][:MAX_ACTIONS]],
    }
    ml = analysis.get("ml")
    if ml and ml.get("status") == "ok":
        cleaned = ml["cleaned"] if ml["cleaned"]["status"] == "ok" else ml["raw"]
        facts["model"] = {
            "metric": ml["metric"]["label"],
            "guessing": cleaned["guess"],
            "beats_guessing_by": cleaned["lift"],
            "raw_data": ml["raw"].get("score"),
            "after_cleaning": ml["cleaned"].get("score"),
            "change_after_cleaning": ml.get("change"),
            "noise": ml.get("noise"),
            "verdict": ml.get("verdict", []),
        }
        imp = ml.get("importance")
        if imp:
            facts["model"]["relied_on"] = [c["column"] for c in imp["columns"] if c["relied_on"]]
            facts["model"]["fix_first"] = [c["column"] for c in imp["columns"] if c["relied_on"] and c["problems"]]
            facts["model"]["importance_verdict"] = imp["verdict"]
    return facts


_NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")


def numbers_in(text: str) -> set[float]:
    out = set()
    for m in _NUMBER.findall(text):
        try:
            out.add(round(float(m.replace(",", "")), 4))
        except ValueError:
            pass
    return out


def allowed_numbers(facts) -> set[float]:
    """Every number that appears anywhere in the facts, as written and as a percentage of 1."""
    found: set[float] = set()

    def walk(v):
        if isinstance(v, bool) or v is None:
            return
        if isinstance(v, (int, float)):
            for x in (v, v * 100, abs(v), abs(v) * 100):
                for digits in (0, 1, 2, 3, 4):
                    found.add(round(float(x), digits))
        elif isinstance(v, str):
            found.update(numbers_in(v))
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)

    walk(facts)
    return found
