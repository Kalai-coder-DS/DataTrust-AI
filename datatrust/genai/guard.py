"""
datatrust/genai/guard.py

Checks an AI-written summary before it is accepted. The scores themselves are never taken from
AI text (the report and pages always show the engine's numbers); this guard stops the AI text
from contradicting them or adding numbers of its own.

Rejected, so the template summary is used instead:
  - empty or absurdly long output
  - any number that is not in the facts (an invented or recalculated score, count or percentage).
    Small whole numbers 0-10 and 100 are allowed ("the top 3 problems", "out of 100").
"""
from __future__ import annotations

from datatrust.genai.facts import allowed_numbers, numbers_in

MAX_CHARS = 4000
ALWAYS_ALLOWED = {float(n) for n in range(0, 11)} | {100.0}


class SummaryRejected(ValueError):
    """The AI output failed a check; the message says which (safe to show and log)."""


def check_summary(text: str, facts: dict) -> str:
    text = (text or "").strip()
    if not text:
        raise SummaryRejected("the AI returned an empty text")
    if len(text) > MAX_CHARS:
        raise SummaryRejected(f"the AI text is too long ({len(text)} characters)")
    allowed = allowed_numbers(facts) | ALWAYS_ALLOWED
    unknown = sorted(n for n in numbers_in(text) if n not in allowed)
    if unknown:
        shown = ", ".join(f"{n:g}" for n in unknown[:5])
        raise SummaryRejected(f"the AI text contains number(s) not in the verified facts: {shown}")
    return text
