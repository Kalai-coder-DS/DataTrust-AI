"""
datatrust/genai/present.py

Display values for a stored summary (analysis["summary"]), shared by the HTML report, the CLI and
the web run page.
"""
from __future__ import annotations


def summary_view(summary: dict | None) -> dict | None:
    if not summary or not summary.get("text"):
        return None
    source = summary.get("label", summary.get("provider", ""))
    if summary.get("model"):
        source += f", model {summary['model']}"
    fallback = None
    if summary.get("fallback_reason"):
        fallback = f"{summary['fallback_reason']}. The template summary was used instead."
    return {
        "paragraphs": [p.strip() for p in summary["text"].split("\n\n") if p.strip()],
        "source": source,
        "is_ai": summary.get("provider") in ("ollama", "claude"),
        "fallback": fallback,
        "rejected_text": summary.get("rejected_text"),
        "note": "The scores are calculated by the DataTrust engine; this summary only explains them. "
                "AI-written text is checked so it cannot contain numbers that are not in the analysis.",
    }
